#!/usr/bin/env python3
"""
Charter-AI — Freight Forecasting Model Training CLI.

Trains and persists dry-bulk freight forecasting models under models/
with rigorous walk-forward cross-validation and JSON metadata.

Usage:
    python scripts/train_freight_model.py --route AUS_NEW_IND_GVM --vessel-class Capesize --model all
    python scripts/train_freight_model.py --origin AUS_NEW --destination IND_GVM --vessel-class Capesize --model xgboost
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Ensure charter-ai root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.models.baseline_forecaster import NaiveBaselineForecaster, MovingAverageForecaster, SeasonalBaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import FreightFeatureBuilder
from src.models.model_evaluation import walk_forward_cv, evaluate_forecast
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_route(route_arg: str):
    """Parse route formatted as AUS_NEW_IND_GVM or AUS_NEW->IND_GVM."""
    if "->" in route_arg:
        parts = route_arg.split("->")
    elif "_" in route_arg:
        # e.g. AUS_NEW_IND_GVM: first 2 parts origin, last 2 dest
        tokens = route_arg.split("_")
        if len(tokens) == 4:
            return f"{tokens[0]}_{tokens[1]}", f"{tokens[2]}_{tokens[3]}"
        elif len(tokens) == 2:
            return tokens[0], tokens[1]
        else:
            mid = len(tokens) // 2
            return "_".join(tokens[:mid]), "_".join(tokens[mid:])
    else:
        parts = route_arg.split("-")
    return parts[0].strip(), parts[1].strip()


def train_and_persist(
    origin: str,
    destination: str,
    vessel_class: str,
    cargo_type: str = "thermal_coal",
    model_choice: str = "all",
    data_dir: str = "data/processed",
    output_dir: str = "models",
    version: str = "1.0.0"
):
    data_path = Path(data_dir)
    freight_file = data_path / "freight_rates.csv"
    if not freight_file.exists():
        data_path = Path("data/demo")
        freight_file = data_path / "freight_rates.csv"

    print("=" * 80)
    print(f" CharterAI V2 — Freight Model Training Engine")
    print(f" Route: {origin} -> {destination} | Vessel: {vessel_class} | Cargo: {cargo_type}")
    print(f" Dataset: {freight_file} | Model: {model_choice.upper()} | Version: {version}")
    print("=" * 80)

    # 1. Load data
    df = pd.read_csv(freight_file)
    df["date"] = pd.to_datetime(df["date"])
    mask = (
        (df["origin"].str.upper() == origin.upper()) &
        (df["destination"].str.upper() == destination.upper()) &
        (df["vessel_class"].str.lower() == vessel_class.lower())
    )
    route_df = df[mask].sort_values("date").reset_index(drop=True)

    if route_df.empty:
        print(f"[ERROR] No historical records found for {origin} -> {destination} ({vessel_class}).")
        sys.exit(1)

    print(f"Loaded {len(route_df)} historical daily records ({route_df['date'].iloc[0].date()} to {route_df['date'].iloc[-1].date()}).")

    # 2. Build multi-domain features
    feature_builder = FreightFeatureBuilder(data_dir=str(data_path))
    feat_df = feature_builder.build_features(
        freight_df=route_df,
        target_col="freight_rate",
        date_col="date",
        destination_port=destination
    )
    feature_cols = feature_builder.get_feature_columns(feat_df, target_col="freight_rate")
    print(f"Engineered {len(feature_cols)} predictor features across freight, market, commodity, energy, macro, and operational domains.")

    # 3. Model definitions
    available_models = {
        "naive": lambda: NaiveBaselineForecaster(),
        "moving_average": lambda: MovingAverageForecaster(window_size=7),
        "seasonal": lambda: SeasonalBaselineForecaster(seasonal_period=7),
        "arima": lambda: ARIMAForecaster(order=(1, 1, 1)),
        "xgboost": lambda: XGBoostForecaster(),
        "ensemble": lambda: EnsembleForecaster(weighting_strategy="inverse_rmse")
    }

    if model_choice == "all":
        models_to_train = list(available_models.keys())
    elif model_choice in available_models:
        models_to_train = [model_choice]
    else:
        print(f"[ERROR] Invalid model choice '{model_choice}'. Choose from: {list(available_models.keys()) + ['all']}")
        sys.exit(1)

    # 4. Walk-Forward Cross-Validation
    print("\nRunning Time-Series Walk-Forward Cross-Validation (Horizon: 7 days, Initial: 180 days, Step: 30 days)...")
    cv_results = {}

    for m_name in models_to_train:
        print(f" -> Evaluating {m_name.upper()} via Walk-Forward CV...", end="", flush=True)
        try:
            res = walk_forward_cv(
                model_factory=available_models[m_name],
                df=feat_df,
                target_col="freight_rate",
                date_col="date",
                horizon_days=7,
                initial_train_size=min(180, len(feat_df) - 60),
                step_size=30,
                window_type="expanding"
            )
            cv_results[m_name] = res
            print(f" Done! [MAE: {res.mae:.3f}, RMSE: {res.rmse:.3f}, MAPE: {res.overall_metrics.get('MAPE', 0):.2f}%, MASE: {res.overall_metrics.get('MASE', 0):.3f}]")
        except Exception as e:
            print(f" FAILED: {e}")

    # 5. Persist models and metadata
    route_key = f"{origin}_{destination}_{vessel_class}".lower()
    base_out = Path(output_dir) / route_key
    base_out.mkdir(parents=True, exist_ok=True)

    print("\nFitting final models on full history & saving artifacts:")
    for m_name in models_to_train:
        model_dir = base_out / m_name
        model_dir.mkdir(parents=True, exist_ok=True)

        final_model = available_models[m_name]()
        final_model.fit(feat_df, target_col="freight_rate", date_col="date")

        artifact_path = model_dir / "model.pkl"
        final_model.save(str(artifact_path))

        # Compile metadata
        metrics = cv_results[m_name].overall_metrics if m_name in cv_results else {}
        metadata = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "dataset_version": "v2.0",
            "route": f"{origin}->{destination}",
            "origin": origin,
            "destination": destination,
            "vessel_class": vessel_class,
            "cargo_type": cargo_type,
            "model_name": m_name,
            "model_version": version,
            "features": feature_cols,
            "metrics": metrics,
            "horizons": [3, 7, 14, 30]
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f" [SAVED] {m_name.upper():<15} -> {artifact_path} & metadata.json")

    print(f"\nTraining pipeline complete! Model artifacts stored in {base_out}/")


def main():
    parser = argparse.ArgumentParser(description="Train CharterAI Freight Forecasting Models")
    parser.add_argument("--origin", type=str, default=None, help="Origin port code, e.g. AUS_NEW")
    parser.add_argument("--destination", type=str, default=None, help="Destination port code, e.g. IND_GVM")
    parser.add_argument("--route", type=str, default="AUS_NEW_IND_GVM", help="Route format AUS_NEW_IND_GVM")
    parser.add_argument("--vessel-class", type=str, default="Capesize", help="Vessel class, e.g. Capesize, Panamax")
    parser.add_argument("--cargo-type", type=str, default="thermal_coal", help="Cargo type, default thermal_coal")
    parser.add_argument("--model", type=str, default="all", help="Model: naive, moving_average, seasonal, arima, xgboost, ensemble, all")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed CSV files")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for trained models")
    parser.add_argument("--version", type=str, default="1.0.0", help="Model version string")

    args = parser.parse_args()

    if args.origin and args.destination:
        origin = args.origin
        destination = args.destination
    else:
        origin, destination = parse_route(args.route)

    train_and_persist(
        origin=origin,
        destination=destination,
        vessel_class=args.vessel_class,
        cargo_type=args.cargo_type,
        model_choice=args.model.lower(),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        version=args.version
    )


if __name__ == "__main__":
    main()

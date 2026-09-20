#!/usr/bin/env python3
"""
DockInsights — Multi-Horizon Freight Forecasting Model Evaluation CLI.

Executes rigorous time-series walk-forward cross-validation across all 6 model
families for horizons: 3 days, 7 days, 14 days, and 30 days.

Metrics evaluated: MAE, RMSE, MAPE, sMAPE, MASE.
Objectively ranks models per horizon without assuming XGBoost is superior.

Usage:
    python scripts/evaluate_freight_models.py --route AUS_NEW_IND_GVM --vessel-class Capesize
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.models.baseline_forecaster import NaiveBaselineForecaster, MovingAverageForecaster, SeasonalBaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import FreightFeatureBuilder
from src.models.model_evaluation import walk_forward_cv, compare_models


def parse_route(route_arg: str):
    if "->" in route_arg:
        parts = route_arg.split("->")
    elif "_" in route_arg:
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


def run_evaluation(
    origin: str,
    destination: str,
    vessel_class: str,
    cargo_type: str = "thermal_coal",
    data_dir: str = "data/processed",
    horizons: list = None
):
    if horizons is None:
        horizons = [3, 7, 14, 30]

    data_path = Path(data_dir)
    freight_file = data_path / "freight_rates.csv"
    if not freight_file.exists():
        data_path = Path("data/demo")
        freight_file = data_path / "freight_rates.csv"

    print("=" * 92)
    print(" DockInsights V2 — Multi-Model Walk-Forward Freight Evaluation Benchmark")
    print(f" Route: {origin} -> {destination} | Vessel Class: {vessel_class} | Cargo: {cargo_type}")
    print(f" Horizons: {horizons} days | Validation: Chronological Walk-Forward")
    print("=" * 92)

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

    print(f"Historical Sample: {len(route_df)} records ({route_df['date'].iloc[0].date()} to {route_df['date'].iloc[-1].date()})\n")

    # Feature engineering
    feature_builder = FreightFeatureBuilder(data_dir=str(data_path))
    feat_df = feature_builder.build_features(
        freight_df=route_df,
        target_col="freight_rate",
        date_col="date",
        destination_port=destination
    )

    models = {
        "Naive Baseline": lambda: NaiveBaselineForecaster(),
        "Moving Average (7d)": lambda: MovingAverageForecaster(window_size=7),
        "Seasonal Baseline (7d)": lambda: SeasonalBaselineForecaster(seasonal_period=7),
        "ARIMA (1,1,1)": lambda: ARIMAForecaster(order=(1, 1, 1)),
        "XGBoost": lambda: XGBoostForecaster(),
        "Ensemble (Weighted)": lambda: EnsembleForecaster(weighting_strategy="inverse_rmse")
    }

    all_results = []

    for h in horizons:
        print(f"\n" + "-" * 92)
        print(f" EVALUATION HORIZON: {h} DAYS (Walk-Forward Cross-Validation)")
        print("-" * 92)
        print(f"{'Model':<25} | {'MAE ($/t)':<10} | {'RMSE ($/t)':<11} | {'MAPE (%)':<10} | {'sMAPE (%)':<10} | {'MASE':<8}")
        print("-" * 92)

        horizon_results = []
        for m_name, m_factory in models.items():
            try:
                res = walk_forward_cv(
                    model_factory=m_factory,
                    df=feat_df,
                    target_col="freight_rate",
                    date_col="date",
                    horizon_days=h,
                    initial_train_size=min(180, len(feat_df) - 60),
                    step_size=30,
                    window_type="expanding"
                )
                m = res.overall_metrics
                horizon_results.append({
                    "Horizon": f"{h}d",
                    "Model": m_name,
                    "MAE": m.get("MAE", np.nan),
                    "RMSE": m.get("RMSE", np.nan),
                    "MAPE": m.get("MAPE", np.nan),
                    "sMAPE": m.get("sMAPE", np.nan),
                    "MASE": m.get("MASE", np.nan)
                })
                print(f"{m_name:<25} | {m.get('MAE', 0.0):<10.3f} | {m.get('RMSE', 0.0):<11.3f} | {m.get('MAPE', 0.0):<9.2f}% | {m.get('sMAPE', 0.0):<9.2f}% | {m.get('MASE', 0.0):<8.3f}")
            except Exception as e:
                print(f"{m_name:<25} | ERROR: {e}")

        # Determine winner for this horizon
        df_h = compare_models(horizon_results)
        if not df_h.empty:
            winner = df_h.iloc[0]["Model"]
            win_mae = df_h.iloc[0]["MAE"]
            print(f" -> Best Model for {h}-Day Horizon: {winner} (MAE: {win_mae:.3f} $/t)")
        all_results.extend(horizon_results)

    print("\n" + "=" * 92)
    print(" Benchmark Evaluation Complete.")
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Freight Forecasting Models")
    parser.add_argument("--origin", type=str, default=None)
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--route", type=str, default="AUS_NEW_IND_GVM")
    parser.add_argument("--vessel-class", type=str, default="Capesize")
    parser.add_argument("--cargo-type", type=str, default="thermal_coal")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 7, 14, 30])

    args = parser.parse_args()

    if args.origin and args.destination:
        origin, destination = args.origin, args.destination
    else:
        origin, destination = parse_route(args.route)

    run_evaluation(
        origin=origin,
        destination=destination,
        vessel_class=args.vessel_class,
        cargo_type=args.cargo_type,
        data_dir=args.data_dir,
        horizons=args.horizons
    )


if __name__ == "__main__":
    main()

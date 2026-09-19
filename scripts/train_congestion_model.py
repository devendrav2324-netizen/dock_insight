#!/usr/bin/env python3
"""
Charter-AI — Port Congestion & Waiting Time Model Training CLI.

Trains Queuing Baseline, Moving Average, and XGBoost Quantile Regression models
using chronological time-series cross-validation (no random splits).
Persists model artifacts and metadata under models/congestion/.

Usage:
    python scripts/train_congestion_model.py --data-dir data/processed --output-dir models/congestion
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.models.congestion_predictor import CongestionPredictor, CongestionFeatureBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def time_series_congestion_cv(df_feat: pd.DataFrame, feature_cols: list, n_splits: int = 4):
    """
    Strict chronological time-series validation for port congestion.
    """
    df_sorted = df_feat.sort_values("date").reset_index(drop=True)
    dates = df_sorted["date"].drop_duplicates().sort_values().reset_index(drop=True)
    total_dates = len(dates)
    fold_size = total_dates // (n_splits + 1)

    import xgboost as xgb
    metrics_list = []

    for fold in range(n_splits):
        train_cutoff_date = dates.iloc[(fold + 1) * fold_size]
        test_cutoff_date = dates.iloc[min(total_dates - 1, (fold + 2) * fold_size)]

        train_mask = df_sorted["date"] < train_cutoff_date
        test_mask = (df_sorted["date"] >= train_cutoff_date) & (df_sorted["date"] < test_cutoff_date)

        train_df = df_sorted[train_mask]
        test_df = df_sorted[test_mask]

        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["average_waiting_days"].values.astype(float)
        X_test = test_df[feature_cols]
        y_test = test_df["average_waiting_days"].values.astype(float)

        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.50, n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_train)

        preds = m.predict(X_test)
        mae = float(np.mean(np.abs(y_test - preds)))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        mape = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-4))) * 100.0)

        metrics_list.append({"fold": fold, "MAE": mae, "RMSE": rmse, "MAPE": mape})

    overall_mae = float(np.mean([m["MAE"] for m in metrics_list]))
    overall_rmse = float(np.mean([m["RMSE"] for m in metrics_list]))
    overall_mape = float(np.mean([m["MAPE"] for m in metrics_list]))
    return {"MAE": round(overall_mae, 3), "RMSE": round(overall_rmse, 3), "MAPE": round(overall_mape, 2), "folds": len(metrics_list)}


def train_congestion(data_dir: str = "data/processed", output_dir: str = "models/congestion", version: str = "1.0.0"):
    d_path = Path(data_dir)
    cong_file = d_path / "congestion.csv"
    weather_file = d_path / "weather.csv"

    if not cong_file.exists():
        d_path = Path("data/demo")
        cong_file = d_path / "congestion.csv"
        weather_file = d_path / "weather.csv"

    print("=" * 80)
    print(" CharterAI V2 — Port Congestion Model Training Engine")
    print(f" Dataset: {cong_file} | Weather: {weather_file} | Version: {version}")
    print("=" * 80)

    df_cong = pd.read_csv(cong_file)
    df_weather = pd.read_csv(weather_file) if weather_file.exists() else None

    ports_list = sorted(df_cong["port"].unique().tolist())
    print(f"Loaded {len(df_cong)} records across {len(ports_list)} ports: {ports_list}")

    # Build features
    fb = CongestionFeatureBuilder(data_dir=str(d_path))
    df_feat = fb.build_features(df_cong, df_weather=df_weather)
    if "cargo_handling_requirement_days" not in df_feat.columns:
        df_feat["cargo_handling_requirement_days"] = 75000.0 / np.maximum(10000.0, df_feat["cargo_handling_rate_mt_day"])

    feature_cols = CongestionPredictor.FEATURE_COLS
    print(f"Built {len(feature_cols)} predictor features (queue depth, berth occupancy, weather, calendar, handling rate).")

    # Time-series cross validation
    print("\nExecuting Time-Series Cross-Validation across chronological folds...")
    cv_metrics = time_series_congestion_cv(df_feat, feature_cols, n_splits=4)
    print(f" -> Cross-Validation Completed ({cv_metrics['folds']} folds):")
    print(f"    MAE:  {cv_metrics['MAE']:.3f} days")
    print(f"    RMSE: {cv_metrics['RMSE']:.3f} days")
    print(f"    MAPE: {cv_metrics['MAPE']:.2f}%")

    # Train final model on full dataset
    print("\nFitting final Quantile Regression (P10, P50, P90) & Delay Probability models on all history...")
    predictor = CongestionPredictor(data_dir=str(d_path))
    predictor.fit(df_cong, df_weather=df_weather)

    # Persist model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_file = out_path / "model.pkl"
    predictor.save(str(artifact_file))

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "v2.0",
        "model_name": "xgboost_quantile_congestion",
        "model_version": version,
        "ports_trained": ports_list,
        "features": feature_cols,
        "metrics": cv_metrics,
        "quantiles": [0.10, 0.50, 0.90]
    }

    metadata_file = out_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SUCCESS] Model artifact saved to: {artifact_file}")
    print(f"[SUCCESS] Metadata saved to:       {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Train CharterAI Port Congestion Model")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models/congestion")
    parser.add_argument("--version", type=str, default="1.0.0")

    args = parser.parse_args()
    train_congestion(data_dir=args.data_dir, output_dir=args.output_dir, version=args.version)


if __name__ == "__main__":
    main()

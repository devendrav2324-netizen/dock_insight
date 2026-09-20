"""
DockInsights — Model Training Orchestration.

Handles cross-validation, hyperparameter tuning, OOS evaluation metrics,
and coordination of the training and model registry persistence pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from src.models.arima_forecaster import ARIMAForecaster
from src.models.baseline_forecaster import BaselineForecaster, MovingAverageForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import build_forecast_features
from src.models.model_evaluation import evaluate_forecast
from src.models.registry import ModelRegistry
from src.models.xgboost_forecaster import XGBoostForecaster
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    model_type: str = "ensemble"  # "xgboost", "arima", "ensemble", "baseline"
    target_column: str = "freight_rate"
    date_column: str = "date"
    test_size_days: int = 30  # Hold out last N days for OOS evaluation
    n_cv_splits: int = 5  # TimeSeriesSplit folds
    random_state: int = 42
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "v1.0.0"
    route: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    vessel_class: Optional[str] = None
    cargo_type: Optional[str] = "thermal_coal"
    data_mode: str = "SYNTHETIC_DEMO"
    provenance_status: str = "SYNTHETIC_DEMO"
    dataset_name: str = "freight_rates"


@dataclass
class TrainingResult:
    """Results from a training run."""
    model_type: str
    model_version: str
    trained_at: datetime
    metrics: Dict[str, float]  # {"MAE": ..., "RMSE": ..., "MAPE": ...}
    hyperparams: Dict[str, Any]
    artifact_path: str
    feature_importances: Optional[Dict[str, float]] = None
    data_mode: str = "SYNTHETIC_DEMO"
    provenance_status: str = "SYNTHETIC_DEMO"
    dataset_name: str = "freight_rates"


class TrainingPipeline:
    """
    Orchestrates model training end-to-end.

    Steps:
    1. Load and validate data
    2. Build features
    3. Chronological TimeSeriesSplit evaluation
    4. Out-of-sample holdout validation
    5. Save model artifact and metadata
    """

    def __init__(
        self,
        config: TrainingConfig,
        registry: Optional[ModelRegistry] = None,
    ):
        self.config = config
        self.registry = registry or ModelRegistry()

    def run(self, df: pd.DataFrame) -> TrainingResult:
        """
        Execute full training workflow on df.
        """
        logger.info(f"Starting training pipeline for {self.config.model_type} ({self.config.model_version})")

        # 1. Feature Engineering
        target_col = self.config.target_column
        date_col = self.config.date_column
        featured_df, feature_cols = build_forecast_features(df, target_col=target_col, date_col=date_col)
        feature_cols = [c for c in featured_df.columns if c not in (date_col, target_col, "origin", "destination", "vessel_class", "cargo_type", "source")]

        # 2. Chronological OOS Split
        if len(featured_df) <= self.config.test_size_days + 14:
            logger.warning(
                f"Insufficient data ({len(featured_df)} rows) for test_size_days={self.config.test_size_days}. Using 80/20 split."
            )
            split_idx = int(len(featured_df) * 0.8)
            train_df = featured_df.iloc[:split_idx].copy()
            eval_df = featured_df.iloc[split_idx:].copy()
        else:
            split_idx = len(featured_df) - self.config.test_size_days
            train_df = featured_df.iloc[:split_idx].copy()
            eval_df = featured_df.iloc[split_idx:].copy()

        # 3. Instantiate & Train Candidate Model
        m_type = self.config.model_type.lower()
        if m_type == "xgboost":
            model = XGBoostForecaster(**self.config.hyperparams)
        elif m_type == "arima":
            model = ARIMAForecaster(**self.config.hyperparams)
        elif m_type == "ensemble":
            model = EnsembleForecaster(**self.config.hyperparams)
        elif m_type == "moving_average":
            model = MovingAverageForecaster(**self.config.hyperparams)
        else:
            model = BaselineForecaster(**self.config.hyperparams)

        # Fit on training split
        model.fit(train_df, target_col=target_col, date_col=date_col)

        # 4. Out-of-Sample Holdout Evaluation
        eval_metrics = model.evaluate(eval_df, target_col=target_col, date_col=date_col)
        metrics = {k: float(v) for k, v in eval_metrics.items() if k != "Model" and not np.isnan(v)}
        metrics["_evaluation"] = "OOS_chronological_holdout"
        metrics["_oos_size"] = len(eval_df)

        # 5. Final Refit on Full Data for Production Artifact
        model.fit(featured_df, target_col=target_col, date_col=date_col)

        # 6. Save & Register Model Artifact
        artifact_path = self.registry.save_model(
            model=model,
            model_name=m_type,
            model_version=self.config.model_version,
            metrics=metrics,
            hyperparameters=self.config.hyperparams,
            route=self.config.route,
            origin=self.config.origin,
            destination=self.config.destination,
            vessel_class=self.config.vessel_class,
            cargo_type=self.config.cargo_type,
            features=feature_cols,
            validation_methodology="OOS_chronological_holdout",
            is_active=True,
            data_mode=self.config.data_mode,
            provenance_status=self.config.provenance_status,
            dataset_name=self.config.dataset_name,
        )

        trained_at = datetime.now(timezone.utc)
        logger.info(
            f"Training complete for {m_type} {self.config.model_version}. OOS MAE={metrics.get('MAE', 0.0):.3f}, RMSE={metrics.get('RMSE', 0.0):.3f}"
        )

        return TrainingResult(
            model_type=m_type,
            model_version=self.config.model_version,
            trained_at=trained_at,
            metrics=metrics,
            hyperparams=self.config.hyperparams,
            artifact_path=artifact_path,
            data_mode=self.config.data_mode,
            provenance_status=self.config.provenance_status,
            dataset_name=self.config.dataset_name,
        )

    def _time_series_split(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create expanding-window chronological train/validation splits.
        Strict temporal ordering: training data always precedes validation data.
        """
        df_sorted = df.sort_values(self.config.date_column).reset_index(drop=True)
        n = len(df_sorted)
        splits = []
        step = max(7, n // (self.config.n_cv_splits + 1))
        initial = max(30, n - (self.config.n_cv_splits * step))

        for i in range(self.config.n_cv_splits):
            cutoff = initial + (i * step)
            if cutoff + step > n:
                break
            train = df_sorted.iloc[:cutoff].copy()
            val = df_sorted.iloc[cutoff:cutoff + step].copy()
            splits.append((train, val))

        return splits

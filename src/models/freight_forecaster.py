"""
Charter-AI — Freight Forecaster Engine & Pipeline Orchestrator.

Provides the end-to-end forecasting pipeline for dry-bulk freight rates across:
  origin -> destination -> vessel_class -> cargo_type
for horizons:
  3 days, 7 days, 14 days, 30 days

Exposes the canonical output dictionary:
{
    "current_rate": float,
    "forecast_rate": float,
    "lower_bound": float,
    "upper_bound": float,
    "trend": "rising" | "falling" | "stable",
    "confidence": float,
    "model_used": str,
    "metrics": {"MAE": float, "RMSE": float, "MAPE": float, "sMAPE": float, "MASE": float}
}
"""

from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json

from src.models.base_forecaster import ForecastModel, ForecastPoint, ForecastResult
from src.models.baseline_forecaster import BaselineForecaster, NaiveBaselineForecaster, MovingAverageForecaster, SeasonalBaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import FreightFeatureBuilder, build_forecast_features
from src.models.model_evaluation import evaluate_forecast, walk_forward_cv
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DataScope(NamedTuple):
    df: pd.DataFrame
    level: str
    n_obs: int
    data_quality: str

# Re-exports for complete backward compatibility
__all__ = [
    "ForecastModel",
    "ForecastPoint",
    "ForecastResult",
    "DataScope",
    "FreightForecaster",
    "BaselineForecaster",
    "NaiveBaselineForecaster",
    "MovingAverageForecaster",
    "SeasonalBaselineForecaster",
    "ARIMAForecaster",
    "XGBoostForecaster",
    "EnsembleForecaster",
]


class FreightForecaster:
    """
    High-level orchestrator for dry-bulk freight forecasting.
    Binds data loading, feature generation, model loading / fitting,
    uncertainty intervals, and canonical API serialization.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        default_model_type: str = "ensemble"
    ):
        settings = get_settings()
        root = Path(__file__).resolve().parent.parent.parent
        self.models_dir = Path(models_dir) if models_dir else (root / "models")
        self.data_dir = Path(data_dir) if data_dir else (root / "data" / "processed" if (root / "data" / "processed").exists() else root / "data" / "demo")
        self.default_model_type = default_model_type.lower()
        self.feature_builder = FreightFeatureBuilder(data_dir=str(self.data_dir))

    def load_historical_data(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        cargo_type: Optional[str] = None
    ) -> DataScope:
        """
        Load historical freight rates matching the route and vessel class
        using a safe hierarchical fallback.
        """
        freight_path = self.data_dir / "freight_rates.csv"
        if not freight_path.exists():
            raise FileNotFoundError(f"Freight rates file not found at {freight_path}")

        df = pd.read_csv(freight_path)
        df["date"] = pd.to_datetime(df["date"])

        MIN_OBS_PREFERRED = 180
        MIN_OBS_FALLBACK = 60

        # LEVEL 1: Exact Match (origin + destination + vessel + cargo)
        mask_exact = (
            (df["origin"].str.upper() == origin.upper()) &
            (df["destination"].str.upper() == destination.upper()) &
            (df["vessel_class"].str.lower() == vessel_class.lower())
        )
        if cargo_type:
            mask_exact = mask_exact & (df["cargo_type"].str.lower() == cargo_type.lower())

        filtered = df[mask_exact].sort_values("date").reset_index(drop=True)
        if len(filtered) >= MIN_OBS_PREFERRED:
            return DataScope(df=filtered, level="EXACT", n_obs=len(filtered), data_quality="HIGH")

        # LEVEL 2: Route + Vessel (drops cargo_type)
        mask_rv = (
            (df["origin"].str.upper() == origin.upper()) &
            (df["destination"].str.upper() == destination.upper()) &
            (df["vessel_class"].str.lower() == vessel_class.lower())
        )
        filtered = df[mask_rv].sort_values("date").reset_index(drop=True)
        if len(filtered) >= MIN_OBS_FALLBACK:
            return DataScope(df=filtered, level="ROUTE_VESSEL", n_obs=len(filtered), data_quality="MEDIUM")

        # LEVEL 3: Route Only (drops vessel_class)
        mask_r = (
            (df["origin"].str.upper() == origin.upper()) &
            (df["destination"].str.upper() == destination.upper())
        )
        filtered = df[mask_r].sort_values("date").reset_index(drop=True)
        if len(filtered) >= MIN_OBS_FALLBACK:
            return DataScope(df=filtered, level="ROUTE", n_obs=len(filtered), data_quality="LOW")

        # LEVEL 4: Route Family (same origin OR same destination region)
        mask_rf = (
            (df["origin"].str.upper() == origin.upper()) |
            (df["destination"].str.upper() == destination.upper())
        )
        filtered = df[mask_rf].sort_values("date").reset_index(drop=True)
        if len(filtered) >= MIN_OBS_FALLBACK:
            return DataScope(df=filtered, level="ROUTE_FAMILY", n_obs=len(filtered), data_quality="LOW")

        # LEVEL 5: Vessel Class Fallback across all routes
        mask_vc = (df["vessel_class"].str.lower() == vessel_class.lower())
        filtered = df[mask_vc].sort_values("date").reset_index(drop=True)
        if len(filtered) >= MIN_OBS_FALLBACK:
            return DataScope(df=filtered, level="VESSEL_CLASS_GLOBAL", n_obs=len(filtered), data_quality="LOW")

        # LEVEL 6: Global Freight Market Fallback
        filtered = df.sort_values("date").reset_index(drop=True)
        if len(filtered) > 0:
            return DataScope(df=filtered, level="GLOBAL_MARKET", n_obs=len(filtered), data_quality="LOW")

        # INSUFFICIENT
        return DataScope(df=pd.DataFrame(), level="INSUFFICIENT", n_obs=0, data_quality="INSUFFICIENT")

    def _get_model_key(self, origin: str, destination: str, vessel_class: str) -> str:
        return f"{origin}_{destination}_{vessel_class}".replace(" ", "_").lower()

    def get_trained_model(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        model_name: Optional[str] = None
    ) -> Tuple[Optional[ForecastModel], Optional[Dict[str, Any]]]:
        """
        Check if a serialized model and its metadata exist under models/.
        Validates model identity metadata before returning artifact.
        """
        m_type = (model_name or self.default_model_type).lower()
        key = self._get_model_key(origin, destination, vessel_class)

        candidate_dirs = [
            self.models_dir / key / m_type,
            self.models_dir / key,
            self.models_dir / m_type,
        ]

        target_dir = None
        for d in candidate_dirs:
            if d.exists():
                if (d / "model.pkl").exists() or (d / "model.joblib").exists():
                    target_dir = d
                    break
                subdirs = [s for s in d.iterdir() if s.is_dir()]
                for s in subdirs:
                    if (s / "model.pkl").exists() or (s / "model.joblib").exists():
                        target_dir = s
                        break
                if target_dir:
                    break

        if not target_dir:
            return None, None

        artifact_file = target_dir / "model.pkl"
        if not artifact_file.exists():
            artifact_file = target_dir / "model.joblib"
            if not artifact_file.exists():
                return None, None

        metadata_file = target_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                req_route = f"{origin}->{destination}".upper()
                meta_route = str(metadata.get("route", "")).upper()
                meta_vc = str(metadata.get("vessel_class", "")).lower()

                if meta_route and meta_route != req_route:
                    logger.warning(
                        f"Model identity mismatch: requested route {req_route}, artifact trained for {meta_route}. Skipping artifact."
                    )
                    return None, None

                if meta_vc and meta_vc != vessel_class.lower():
                    logger.warning(
                        f"Model identity mismatch: requested vessel {vessel_class}, artifact trained for {meta_vc}. Skipping artifact."
                    )
                    return None, None

            except Exception as e:
                logger.warning(f"Failed to validate metadata from {metadata_file}: {e}")

        m_name = metadata.get("model_name", target_dir.name).lower()
        if "xgboost" in m_name:
            model = XGBoostForecaster()
        elif "arima" in m_name:
            model = ARIMAForecaster()
        elif "ensemble" in m_name:
            model = EnsembleForecaster()
        else:
            model = BaselineForecaster()

        try:
            model.load(str(artifact_file))
            return model, metadata
        except Exception as e:
            logger.warning(f"Failed loading model artifact {artifact_file}: {e}")
            return None, None

    def predict_freight(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        cargo_type: str = "thermal_coal",
        horizon_days: int = 7,
        model_type: Optional[str] = None,
        allow_on_the_fly: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate end-to-end freight forecast returning canonical schema.
        """
        valid_horizons = [3, 7, 14, 30]
        if horizon_days not in valid_horizons:
            horizon_days = min(valid_horizons, key=lambda x: abs(x - horizon_days))

        route_str = f"{origin}->{destination}"
        selected_type = (model_type or self.default_model_type).lower()

        # 1. Load historical context data
        data_scope = self.load_historical_data(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type
        )
        
        if data_scope.level == "INSUFFICIENT":
            raise ValueError(
                f"Insufficient historical data for {origin}->{destination} {vessel_class}. "
                f"Required at least 60 observations for fallback."
            )
            
        df_history = data_scope.df
        current_rate = float(df_history["freight_rate"].iloc[-1])

        # 2. Enrich features
        featured_df = self.feature_builder.build_features(
            freight_df=df_history,
            target_col="freight_rate",
            date_col="date",
            destination_port=destination
        )

        # 3. Check for pre-trained model in models/
        model, metadata = self.get_trained_model(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            model_name=selected_type
        )

        metrics: Dict[str, float] = {}
        if model is None:
            if not allow_on_the_fly:
                from src.models.registry import ModelNotFoundError
                raise ModelNotFoundError(
                    f"No trained model artifact found for {route_str} {vessel_class} ({selected_type}) and allow_on_the_fly is False."
                )

            logger.info(f"No pre-trained artifact found for {route_str} {vessel_class}. Training on-the-fly {selected_type} model.")

            if selected_type == "naive":
                model = NaiveBaselineForecaster()
            elif selected_type == "moving_average":
                model = MovingAverageForecaster(window_size=7)
            elif selected_type == "seasonal":
                model = SeasonalBaselineForecaster(seasonal_period=7)
            elif selected_type == "arima":
                model = ARIMAForecaster(order=(1, 1, 1))
            elif selected_type == "xgboost":
                model = XGBoostForecaster()
            else:
                model = EnsembleForecaster(weighting_strategy="inverse_rmse")

            n_feat = len(featured_df)
            oos_h = min(30, max(7, n_feat // 5))

            if n_feat > oos_h + 14:
                fit_df_oos = featured_df.iloc[:-oos_h].copy()
                oos_eval_df = featured_df.iloc[-oos_h:].copy()

                model.fit(fit_df_oos, target_col="freight_rate", date_col="date")
                oos_eval = model.evaluate(oos_eval_df, target_col="freight_rate", date_col="date")
                metrics = {k: v for k, v in oos_eval.items() if k != "Model"}
                metrics["_evaluation"] = "OOS_chronological_holdout"
                metrics["_oos_size"] = oos_h

                model.fit(featured_df, target_col="freight_rate", date_col="date")
            else:
                model.fit(featured_df, target_col="freight_rate", date_col="date")
                metrics = {"_evaluation": "insufficient_data_for_OOS", "_note": "Series too short for holdout"}

        else:
            metrics = metadata.get("metrics", {})


        # 4. Generate forecast
        forecast_res = model.predict(
            horizon_days=horizon_days,
            context_df=featured_df,
            date_col="date",
            target_col="freight_rate",
            route=route_str,
            vessel_class=vessel_class
        )

        last_point = forecast_res.series[-1]
        forecast_rate = float(last_point.predicted_rate)
        lower_bound = float(last_point.lower_ci if last_point.lower_ci is not None else forecast_rate * 0.95)
        upper_bound = float(last_point.upper_ci if last_point.upper_ci is not None else forecast_rate * 1.05)

        # 5. Determine trend
        pct_change = ((forecast_rate - current_rate) / max(0.1, current_rate)) * 100.0
        if pct_change > 2.0:
            trend = "rising"
        elif pct_change < -2.0:
            trend = "falling"
        else:
            trend = "stable"

        # 6. Statistically defensible confidence score
        # Bounded between 0.60 and 0.98, derived from relative uncertainty width
        rel_uncertainty = (upper_bound - lower_bound) / (2.0 * max(1.0, forecast_rate))
        confidence = float(np.clip(1.0 - rel_uncertainty, 0.60, 0.98))

        return {
            "current_rate": round(current_rate, 2),
            "forecast_rate": round(forecast_rate, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "trend": trend,
            "confidence": round(confidence, 2),
            "model_used": forecast_res.model_used,
            "metrics": metrics,
            "route": route_str,
            "vessel_class": vessel_class,
            "cargo_type": cargo_type,
            "data_scope": data_scope.level,
            "fallback_level": data_scope.level,
            "training_observations": data_scope.n_obs,
            "data_quality": data_scope.data_quality,
            "data_mode": metadata.get("data_mode", "SYNTHETIC_DEMO") if isinstance(metadata, dict) else "SYNTHETIC_DEMO",
            "provenance_status": metadata.get("provenance_status", "SYNTHETIC_DEMO") if isinstance(metadata, dict) else "SYNTHETIC_DEMO",
            "is_verified_external": metadata.get("is_verified_external", False) if isinstance(metadata, dict) else False,
        }

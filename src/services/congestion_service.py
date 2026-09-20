"""
DockInsights — Congestion & Idle Time Service.

Wraps the CongestionPredictor and IdleTimePredictor models to serve
port waiting times, delay probabilities, and demurrage exposure to:
- Voyage Economics Engine
- Risk Assessment Engine
- Contract Strategy Optimizer
- Decision Support Engine
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import date
import pandas as pd

from src.models.congestion_predictor import CongestionPredictor, CongestionPrediction
from src.models.idle_time_predictor import IdleTimePredictor, VoyageIdleTimeResult
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CongestionService:
    """
    Centralized service for port congestion and voyage idle time intelligence.
    """

    def __init__(self, data_dir: Optional[str] = None, models_dir: Optional[str] = None):
        self.settings = get_settings()
        root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else (root / "data" / "processed" if (root / "data" / "processed").exists() else root / "data" / "demo")
        self.models_dir = Path(models_dir) if models_dir else (root / "models" / "congestion")

        self.predictor = CongestionPredictor(data_dir=str(self.data_dir))
        self.idle_predictor = IdleTimePredictor(data_dir=str(self.data_dir))
        self._initialize_model()

    def _initialize_model(self):
        """Attempt loading pre-trained artifact or fit on historical data."""
        artifact = self.models_dir / "model.pkl"
        if artifact.exists():
            try:
                self.predictor.load(str(artifact))
                self.idle_predictor.congestion_predictor = self.predictor
                self.idle_predictor.is_fitted = True
                logger.info(f"Loaded trained congestion model from {artifact}")
                return
            except Exception as e:
                logger.warning(f"Failed loading model from {artifact}: {e}")

        # On-the-fly fit from historical CSV
        cong_csv = self.data_dir / "congestion.csv"
        weather_csv = self.data_dir / "weather.csv"
        if cong_csv.exists():
            try:
                df_cong = pd.read_csv(cong_csv)
                df_weather = pd.read_csv(weather_csv) if weather_csv.exists() else None
                self.predictor.fit(df_cong, df_weather=df_weather)
                self.idle_predictor.congestion_predictor = self.predictor
                self.idle_predictor.is_fitted = True
                logger.info(f"CongestionService fitted on-the-fly from {cong_csv}")
            except Exception as e:
                logger.warning(f"Failed fitting on-the-fly congestion model: {e}")

    def predict_congestion(
        self,
        port_id: str,
        target_date: Optional[Union[date, str]] = None,
        vessel_class: Optional[str] = "Panamax",
        cargo_type: Optional[str] = "thermal_coal",
        cargo_quantity: Optional[float] = 75000.0,
    ) -> Dict[str, Any]:
        """
        Produce canonical congestion prediction dictionary:
        {
            "expected_wait_days": float,
            "p10_wait_days": float,
            "p50_wait_days": float,
            "p90_wait_days": float,
            "delay_probability": float,
            "congestion_level": "LOW" | "MODERATE" | "HIGH" | "SEVERE",
            "confidence": float
        }
        """
        pred = self.predictor.predict_congestion(
            port_id=port_id,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )
        return pred.to_dict()

    def predict_voyage_idle(
        self,
        origin_port: str,
        destination_port: str,
        target_date: Optional[Union[date, str]] = None,
        vessel_class: Optional[str] = "Panamax",
        cargo_quantity: Optional[float] = 75000.0,
        cargo_type: Optional[str] = "thermal_coal"
    ) -> VoyageIdleTimeResult:
        """
        Predict combined voyage idle time, laytime exceedance, and demurrage exposure.
        """
        return self.idle_predictor.predict_voyage_idle(
            origin_port=origin_port,
            destination_port=destination_port,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_quantity=cargo_quantity,
            cargo_type=cargo_type
        )

    def get_expected_wait_days(
        self,
        port_id: str,
        target_date: Optional[Union[date, str]] = None,
        vessel_class: Optional[str] = "Panamax"
    ) -> float:
        """Convenience accessor returning float expected waiting days."""
        pred = self.predictor.predict_congestion(
            port_id=port_id,
            target_date=target_date,
            vessel_class=vessel_class
        )
        return float(pred.expected_wait_days)

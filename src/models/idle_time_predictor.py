"""
Charter-AI — Idle Time & Demurrage Risk Predictor.

Predicts expected vessel waiting time and demurrage risk at loading and
discharge ports based on port congestion, berth capacity, weather conditions,
and seasonal patterns.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.congestion_predictor import CongestionPredictor, CongestionPrediction
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IdleTimePrediction:
    """Predicted idle/waiting time at a port."""
    port_id: str
    predicted_date: date
    predicted_waiting_days: float
    prediction_lower: float  # Lower bound (P10)
    prediction_upper: float  # Upper bound (P90)
    confidence: float
    demurrage_risk_level: str  # "low", "moderate", "high", "critical"
    key_factors: dict  # Top contributing factors
    delay_probability: float = 0.0
    congestion_level: str = "MODERATE"


@dataclass
class VoyageIdleTimeResult:
    """Total idle time and operational delay risk for an entire voyage."""
    origin_port: str
    destination_port: str
    origin_wait_days: float
    destination_wait_days: float
    total_idle_days: float
    demurrage_exposure_days: float
    demurrage_risk_level: str
    laytime_allowance_days: float
    delivery_delay_probability: float
    confidence: float


class IdleTimePredictor:
    """
    Predicts operational waiting time and demurrage exposure.
    Wraps the CongestionPredictor engine to assess single-port and voyage-level delays.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.congestion_predictor = CongestionPredictor(data_dir=data_dir)
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, df_weather: Optional[pd.DataFrame] = None) -> None:
        """Fit underlying congestion predictor."""
        self.congestion_predictor.fit(train_df, df_weather=df_weather)
        self.is_fitted = self.congestion_predictor.is_fitted

    def predict(
        self,
        port_id: str,
        target_date: Optional[Union[date, str]] = None,
        vessel_class: Optional[str] = "Panamax",
        cargo_type: Optional[str] = "thermal_coal",
        cargo_quantity: Optional[float] = 75000.0,
    ) -> IdleTimePrediction:
        """
        Predict expected waiting time and demurrage risk at a specific port.
        """
        pred_res = self.congestion_predictor.predict_congestion(
            port_id=port_id,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )

        # Standard laytime threshold (often 2.0 to 2.5 days for bulk ports)
        allowable_laytime = 2.0
        demurrage_days = max(0.0, pred_res.expected_wait_days - allowable_laytime)

        if demurrage_days < 0.5:
            risk_level = "low"
        elif demurrage_days < 2.0:
            risk_level = "moderate"
        elif demurrage_days < 3.5:
            risk_level = "high"
        else:
            risk_level = "critical"

        t_date = target_date if isinstance(target_date, date) else (pd.to_datetime(target_date).date() if target_date else date.today())

        factors = {
            "congestion_index": pred_res.congestion_level,
            "vessel_class": vessel_class,
            "delay_probability_pct": round(pred_res.delay_probability * 100, 1),
            "p90_tail_risk_days": pred_res.p90_wait_days
        }

        return IdleTimePrediction(
            port_id=port_id,
            predicted_date=t_date,
            predicted_waiting_days=pred_res.expected_wait_days,
            prediction_lower=pred_res.p10_wait_days,
            prediction_upper=pred_res.p90_wait_days,
            confidence=pred_res.confidence,
            demurrage_risk_level=risk_level,
            key_factors=factors,
            delay_probability=pred_res.delay_probability,
            congestion_level=pred_res.congestion_level
        )

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
        Predict total idle time across both origin (load) and destination (discharge) ports.
        """
        origin_pred = self.predict(
            port_id=origin_port,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )

        dest_pred = self.predict(
            port_id=destination_port,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )

        total_idle = origin_pred.predicted_waiting_days + dest_pred.predicted_waiting_days
        allowable_laytime = 4.0  # 2 days load + 2 days discharge
        demurrage_days = max(0.0, total_idle - allowable_laytime)

        # Combined delivery delay probability: P(A or B) = 1 - (1 - P(A))*(1 - P(B))
        delay_prob = 1.0 - (1.0 - origin_pred.delay_probability) * (1.0 - dest_pred.delay_probability)

        if demurrage_days < 1.0:
            demurrage_risk = "low"
        elif demurrage_days < 2.5:
            demurrage_risk = "moderate"
        elif demurrage_days < 4.5:
            demurrage_risk = "high"
        else:
            demurrage_risk = "critical"

        avg_confidence = (origin_pred.confidence + dest_pred.confidence) / 2.0

        return VoyageIdleTimeResult(
            origin_port=origin_port,
            destination_port=destination_port,
            origin_wait_days=origin_pred.predicted_waiting_days,
            destination_wait_days=dest_pred.predicted_waiting_days,
            total_idle_days=round(total_idle, 2),
            demurrage_exposure_days=round(demurrage_days, 2),
            demurrage_risk_level=demurrage_risk,
            laytime_allowance_days=allowable_laytime,
            delivery_delay_probability=round(delay_prob, 2),
            confidence=round(avg_confidence, 2)
        )

    def save(self, path: str) -> None:
        self.congestion_predictor.save(path)

    def load(self, path: str) -> None:
        self.congestion_predictor.load(path)
        self.is_fitted = self.congestion_predictor.is_fitted

"""
DockInsights — Freight Forecast Service.

Wraps the underlying FreightForecaster ML/statistical engine to serve
live freight rate forecasts, uncertainty intervals, and metrics.
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from src.models.base_forecaster import ForecastResult, ForecastPoint
from src.models.freight_forecaster import FreightForecaster
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.optimization.contract_optimizer import ForecastDirection, ForecastUncertainty

logger = get_logger(__name__)


class FreightForecastService:
    """
    Service for generating dry-bulk freight rate forecasts.
    Integrates trained models from models/ with on-the-fly forecasting pipeline.
    """

    def __init__(self, data_dir: Optional[str] = None, models_dir: Optional[str] = None):
        self.settings = get_settings()
        self.forecaster = FreightForecaster(data_dir=data_dir, models_dir=models_dir)

    def _parse_route(self, route: str) -> tuple[str, str]:
        """Extract origin and destination from route identifier."""
        if "->" in route:
            parts = route.split("->")
            return parts[0].strip(), parts[1].strip()
        elif "_" in route:
            tokens = route.split("_")
            if len(tokens) == 4:
                return f"{tokens[0]}_{tokens[1]}", f"{tokens[2]}_{tokens[3]}"
            elif len(tokens) == 2:
                return tokens[0], tokens[1]
            else:
                mid = len(tokens) // 2
                return "_".join(tokens[:mid]), "_".join(tokens[mid:])
        return "AUS_NEW", "IND_GVM"

    def predict_freight_api(
        self,
        origin: str,
        destination: str,
        vessel_class: str,
        cargo_type: str = "thermal_coal",
        horizon_days: int = 7,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Produce canonical API forecasting response:
        {
            "current_rate": float,
            "forecast_rate": float,
            "lower_bound": float,
            "upper_bound": float,
            "trend": str,
            "confidence": float,
            "model_used": str,
            "metrics": dict
        }
        """
        return self.forecaster.predict_freight(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            horizon_days=horizon_days,
            model_type=model_type
        )

    def get_forecast(
        self,
        route: str,
        vessel_class: str,
        horizon_days: int = 7,
        cargo_type: str = "thermal_coal"
    ) -> ForecastResult:
        """
        Get a ForecastResult object for internal services and decision engine.
        """
        origin, destination = self._parse_route(route)
        pred_dict = self.predict_freight_api(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            horizon_days=horizon_days
        )

        # Reconstruct series points
        today = datetime.now().date()
        series = []
        current = pred_dict["current_rate"]
        target = pred_dict["forecast_rate"]
        lower = pred_dict["lower_bound"]
        upper = pred_dict["upper_bound"]

        for i in range(1, horizon_days + 1):
            p_date = today + timedelta(days=i)
            # Linear ramp from current to target
            alpha = i / horizon_days
            p_val = current + alpha * (target - current)
            p_low = p_val - (1.0 - alpha * 0.2) * (target - lower)
            p_high = p_val + (1.0 - alpha * 0.2) * (upper - target)
            series.append(ForecastPoint(
                date=p_date,
                predicted_rate=round(p_val, 2),
                lower_ci=round(max(0.0, p_low), 2),
                upper_ci=round(p_high, 2)
            ))

        return ForecastResult(
            route=f"{origin}->{destination}",
            vessel_type=vessel_class,
            horizon_days=horizon_days,
            model_used=pred_dict["model_used"],
            series=series,
            confidence_available=True,
            forecast_date=today,
            metrics=pred_dict.get("metrics")
        )

    def get_legacy_forecast_dict(self) -> Dict[str, Any]:
        """
        Compatibility method for legacy tests and decision engine defaults.
        """
        return {
            "forecast_rate_usd": self.settings.default_forecast_rate_usd,
            "direction": ForecastDirection[self.settings.default_forecast_direction].value,
            "uncertainty": ForecastUncertainty[self.settings.default_forecast_uncertainty].value,
            "status": "SUCCESS"
        }

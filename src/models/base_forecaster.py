"""
Charter-AI — Canonical Forecasting Interface.

Single source of truth for all forecast data structures and the
ForecastModel abstract base class. All forecasters (Baseline, ARIMA,
XGBoost, Ensemble) MUST implement ForecastModel and return ForecastResult.

V2: Consolidated from duplicate definitions in base_forecaster.py and
freight_forecaster.py. The latter is now a backwards-compatibility shim.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ForecastPoint:
    """Single point in a forecast time series."""
    date: date
    predicted_rate: float
    lower_ci: Optional[float] = None  # Lower bound of confidence interval
    upper_ci: Optional[float] = None  # Upper bound of confidence interval


@dataclass
class ForecastResult:
    """
    Complete forecast result returned by any model.

    This is the canonical structure used across:
    - Model predict() methods
    - FreightForecastService
    - DecisionEngine
    - API serialization
    """
    route: str
    vessel_type: str
    horizon_days: int
    model_used: str
    series: List[ForecastPoint] = field(default_factory=list)
    confidence_available: bool = False
    forecast_date: Optional[date] = None  # Date when forecast was generated
    metrics: Optional[Dict[str, float]] = None  # e.g. {"MAE": ..., "RMSE": ...}

    @property
    def predicted_trend(self) -> str:
        """Determine if rates are trending up, down, or flat."""
        if len(self.series) < 2:
            return "insufficient_data"
        first = self.series[0].predicted_rate
        last = self.series[-1].predicted_rate
        if first == 0:
            return "insufficient_data"
        pct_change = (last - first) / first * 100
        if pct_change > 5:
            return "rising"
        elif pct_change < -5:
            return "falling"
        return "stable"

    @property
    def mean_forecast(self) -> Optional[float]:
        """Average predicted rate across the forecast horizon."""
        if not self.series:
            return None
        return sum(p.predicted_rate for p in self.series) / len(self.series)

    # Convenience aliases for canonical V2 naming compatibility
    @property
    def vessel_class(self) -> str:
        return self.vessel_type

    @property
    def horizon(self) -> int:
        return self.horizon_days

    @property
    def model_name(self) -> str:
        return self.model_used

    @property
    def point_forecast(self) -> float:
        return self.series[-1].predicted_rate if self.series else 0.0

    @property
    def lower_bound(self) -> Optional[float]:
        return self.series[-1].lower_ci if self.series and self.series[-1].lower_ci is not None else None

    @property
    def upper_bound(self) -> Optional[float]:
        return self.series[-1].upper_ci if self.series and self.series[-1].upper_ci is not None else None

    @property
    def confidence_level(self) -> float:
        return 0.95 if self.confidence_available else 0.0


class ForecastModel(ABC):
    """
    Abstract base class for all freight forecasting models.

    Enforces a unified interface for model training, prediction,
    evaluation, and persistence.
    """

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Train the model on the provided historical data.
        """
        pass

    @abstractmethod
    def predict(self, horizon_days: int, context_df: pd.DataFrame, date_col: str, **kwargs) -> ForecastResult:
        """
        Generate a forecast for the specified horizon.

        Args:
            horizon_days: Number of days into the future to forecast.
            context_df: Recent historical data required by the model to generate the forecast
                        (e.g., for calculating lags or setting the ARIMA initial state).
            date_col: Name of the date column.
            **kwargs: Additional model-specific parameters (e.g. route, vessel_type).
        """
        pass

    @abstractmethod
    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        """
        Evaluate the model against a test dataset.
        Returns a dictionary of metrics.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Serialize the trained model to disk.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Deserialize a trained model from disk.
        """
        pass

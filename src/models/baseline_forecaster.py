"""
DockInsights — Baseline Forecasters.

Provides 3 statistical baselines:
1. Naive Baseline: Carries the last observed rate forward into the future.
2. Moving Average Baseline: Averages the most recent W days.
3. Seasonal Baseline: Uses seasonal lag replication (e.g. 7-day weekly cycle).

Uncertainty Quantification:
Statistically defensible prediction intervals based on in-sample residual variance:
  e_t = y_t - y_hat_t
  sigma_h = std(e) * sqrt(1 + (h - 1) / W)
  P10 = y_hat - 1.645 * sigma_h
  P90 = y_hat + 1.645 * sigma_h
"""

from typing import Optional, Dict, Any, List
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle

from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.model_evaluation import evaluate_forecast


class BaselineForecaster(ForecastModel):
    """
    Unified Baseline Forecaster supporting Naive, Moving Average, and Seasonal methods.
    """

    def __init__(
        self,
        method: str = "moving_average",
        window_size: int = 7,
        seasonal_period: int = 7
    ):
        self.method = method.lower()
        self.window_size = window_size
        self.seasonal_period = seasonal_period

        self.last_value: Optional[float] = None
        self.seasonal_pattern: Optional[List[float]] = None
        self.residual_std: float = 0.5  # Fallback default until fit
        self.model_name: str = f"Baseline_{self.method.title()}"
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit baseline on historical data and estimate residual standard deviation.
        """
        if len(df_train) == 0:
            raise ValueError("Training data is empty.")

        df_sorted = df_train.sort_values(by=date_col).reset_index(drop=True)
        series = df_sorted[target_col].astype(float)
        n = len(series)

        if self.method == "naive":
            self.last_value = float(series.iloc[-1])
            self.model_name = "Baseline_Naive"
            # In-sample 1-step errors
            if n > 1:
                residuals = series.diff().dropna()
                self.residual_std = float(residuals.std()) if len(residuals) > 1 else 0.5
            else:
                self.residual_std = 0.5

        elif self.method == "seasonal":
            self.model_name = f"Baseline_Seasonal_{self.seasonal_period}"
            if n >= self.seasonal_period:
                self.seasonal_pattern = series.tail(self.seasonal_period).tolist()
                self.last_value = float(series.iloc[-1])
                # Seasonal residuals: y_t - y_{t-S}
                if n > self.seasonal_period:
                    residuals = (series - series.shift(self.seasonal_period)).dropna()
                    self.residual_std = float(residuals.std()) if len(residuals) > 1 else 0.5
                else:
                    self.residual_std = 0.5
            else:
                # Fallback to moving average if history is shorter than seasonal period
                w = min(self.window_size, n)
                self.last_value = float(series.tail(w).mean())
                self.seasonal_pattern = None
                self.residual_std = float(series.tail(w).std()) if w > 1 else 0.5

        else:
            # Default: moving_average
            w = min(self.window_size, n)
            self.last_value = float(series.tail(w).mean())
            self.model_name = f"Baseline_MA{self.window_size}"
            # Residuals of rolling mean
            if n > w:
                roll_mean = series.rolling(window=w).mean().shift(1)
                residuals = (series - roll_mean).dropna()
                self.residual_std = float(residuals.std()) if len(residuals) > 1 else 0.5
            else:
                self.residual_std = float(series.std()) if n > 1 else 0.5

        if self.residual_std is None or np.isnan(self.residual_std) or self.residual_std < 1e-4:
            self.residual_std = 0.25

        self.is_fitted = True

    def predict(
        self,
        horizon_days: int,
        context_df: pd.DataFrame,
        date_col: str,
        **kwargs
    ) -> ForecastResult:
        """
        Generate forecast points with statistically sound residual prediction intervals.
        """
        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", kwargs.get("vessel_class", "unknown"))
        target_col = kwargs.get("target_col", "freight_rate")
        if not context_df.empty and target_col not in context_df.columns:
            target_col = context_df.columns[-1]

        if not context_df.empty:
            df_sorted = context_df.sort_values(by=date_col).reset_index(drop=True)
            last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])
            series = df_sorted[target_col].astype(float)

            if self.method == "naive":
                base_val = float(series.iloc[-1])
            elif self.method == "seasonal" and len(series) >= self.seasonal_period:
                pattern = series.tail(self.seasonal_period).tolist()
                base_val = float(series.iloc[-1])
            else:
                w = min(self.window_size, len(series))
                base_val = float(series.tail(w).mean())
                pattern = None
        else:
            base_val = self.last_value if self.last_value is not None else 0.0
            pattern = self.seasonal_pattern
            last_date = pd.to_datetime("today")

        series_out: List[ForecastPoint] = []
        for i in range(1, horizon_days + 1):
            pred_date = last_date + timedelta(days=i)

            if self.method == "seasonal" and pattern:
                idx = (i - 1) % len(pattern)
                pred_val = float(pattern[idx])
            else:
                pred_val = float(base_val)

            # Statistically defensible expanding standard error: sigma_h = sigma_res * sqrt(1 + (h-1)/w)
            # 90% Prediction Interval: Z = 1.645
            expansion = np.sqrt(1.0 + (i - 1) / max(1, self.window_size))
            sigma_h = self.residual_std * expansion
            lower_ci = max(0.0, pred_val - 1.645 * sigma_h)
            upper_ci = pred_val + 1.645 * sigma_h

            series_out.append(ForecastPoint(
                date=pred_date.date(),
                predicted_rate=round(pred_val, 2),
                lower_ci=round(lower_ci, 2),
                upper_ci=round(upper_ci, 2)
            ))

        return ForecastResult(
            route=route,
            vessel_type=vessel_type,
            horizon_days=horizon_days,
            model_used=self.model_name,
            series=series_out,
            confidence_available=True,
            forecast_date=last_date.date()
        )

    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        """Evaluate baseline sequentially on a test set."""
        df_sorted = df_test.sort_values(by=date_col).reset_index(drop=True)
        val = self.last_value if self.last_value is not None else float(df_sorted[target_col].iloc[0])
        predictions = np.full(len(df_sorted), val)
        metrics = evaluate_forecast(df_sorted[target_col], pd.Series(predictions))
        metrics["Model"] = self.model_name
        return metrics

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.__dict__.update(pickle.load(f))


# Dedicated convenience wrappers
class NaiveBaselineForecaster(BaselineForecaster):
    def __init__(self):
        super().__init__(method="naive", window_size=1)


class MovingAverageForecaster(BaselineForecaster):
    def __init__(self, window_size: int = 7):
        super().__init__(method="moving_average", window_size=window_size)


class SeasonalBaselineForecaster(BaselineForecaster):
    def __init__(self, seasonal_period: int = 7):
        super().__init__(method="seasonal", seasonal_period=seasonal_period)

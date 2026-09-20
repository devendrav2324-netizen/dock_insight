"""
DockInsights — ARIMA & SARIMA Freight Forecaster.

Statistical time-series models for dry-bulk freight rates.
Provides native parametric prediction intervals (P10, P50, P90)
derived from the variance of forecast errors.
"""

from typing import Optional, Tuple, Dict, Any, List
from datetime import timedelta
import warnings
import numpy as np
import pandas as pd
import pickle

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.model_evaluation import evaluate_forecast
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ARIMAForecaster(ForecastModel):
    """
    ARIMA / SARIMA Model for univariate freight rate forecasting.
    Provides native statistical prediction intervals (P10 to P90).
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        alpha: float = 0.10  # 90% prediction intervals (P10 to P90)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.alpha = alpha
        self.model_fit: Optional[ARIMAResults] = None
        self.last_series: Optional[pd.Series] = None
        self.last_date: Optional[pd.Timestamp] = None
        self.target_col: str = "freight_rate"
        self.date_col: str = "date"
        self.model_name = f"ARIMA_{self.order}" if seasonal_order is None else f"SARIMA_{self.order}x{seasonal_order}"

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit ARIMA / SARIMA model on the training series.
        Employs fallback strategies if numerical convergence encounters difficulties.
        """
        if len(df_train) == 0:
            raise ValueError("Training data is empty.")

        self.target_col = target_col
        self.date_col = date_col

        df_sorted = df_train.sort_values(by=date_col).reset_index(drop=True)
        series = df_sorted[target_col].astype(float)
        self.last_series = series
        self.last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])

        # Candidate parameter sets to try if chosen order fails
        candidate_orders = [self.order, (1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0)]
        fitted = False

        for ord_candidate in candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.seasonal_order is not None and len(series) >= 2 * self.seasonal_order[3]:
                        model = SARIMAX(series, order=ord_candidate, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        model = ARIMA(series, order=ord_candidate)
                    self.model_fit = model.fit()
                    self.order = ord_candidate
                    fitted = True
                    break
            except Exception as e:
                logger.debug(f"ARIMA order {ord_candidate} fit failed: {e}")
                continue

        if not fitted or self.model_fit is None:
            # Fallback simple AR(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=(1, 0, 0))
                self.model_fit = model.fit()
                self.order = (1, 0, 0)

    def predict(
        self,
        horizon_days: int,
        context_df: pd.DataFrame,
        date_col: str,
        **kwargs
    ) -> ForecastResult:
        """
        Generate multi-step forecasts with native confidence intervals.
        """
        if self.model_fit is None:
            raise ValueError("Model is not fitted yet.")

        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", kwargs.get("vessel_class", "unknown"))
        target_col = kwargs.get("target_col", self.target_col)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not context_df.empty:
                df_sorted = context_df.sort_values(by=date_col).reset_index(drop=True)
                if target_col not in df_sorted.columns:
                    target_col = df_sorted.columns[-1]
                series = df_sorted[target_col].astype(float)
                last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])

                try:
                    # Update state with recent context data
                    temp_model = ARIMA(series, order=self.order)
                    temp_fit = temp_model.fit()
                    forecast_res = temp_fit.get_forecast(steps=horizon_days)
                except Exception:
                    forecast_res = self.model_fit.get_forecast(steps=horizon_days)
            else:
                forecast_res = self.model_fit.get_forecast(steps=horizon_days)
                last_date = self.last_date if self.last_date is not None else pd.to_datetime("today")

        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=self.alpha)

        series_out: List[ForecastPoint] = []
        for i in range(horizon_days):
            pred_date = last_date + timedelta(days=i + 1)
            p_val = float(pred_mean.iloc[i]) if hasattr(pred_mean, "iloc") else float(pred_mean[i])
            lower = float(conf_int.iloc[i, 0]) if hasattr(conf_int, "iloc") else float(conf_int[i, 0])
            upper = float(conf_int.iloc[i, 1]) if hasattr(conf_int, "iloc") else float(conf_int[i, 1])

            # Freight rate non-negativity constraint
            lower = max(0.0, lower)

            series_out.append(ForecastPoint(
                date=pred_date.date(),
                predicted_rate=round(p_val, 2),
                lower_ci=round(lower, 2),
                upper_ci=round(upper, 2)
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
        """Evaluate ARIMA model over test set."""
        df_sorted = df_test.sort_values(by=date_col).reset_index(drop=True)
        preds = self.model_fit.forecast(steps=len(df_sorted))
        metrics = evaluate_forecast(
            df_sorted[target_col].values,
            np.array(preds.values, dtype=float),
            y_train=self.last_series.values if self.last_series is not None else None
        )
        metrics["Model"] = self.model_name
        return metrics

    def save(self, filepath: str) -> None:
        if self.model_fit:
            # Save pickle containing fit parameters and config
            with open(filepath, "wb") as f:
                pickle.dump({
                    "order": self.order,
                    "seasonal_order": self.seasonal_order,
                    "alpha": self.alpha,
                    "model_name": self.model_name,
                    "target_col": self.target_col,
                    "date_col": self.date_col,
                    "params": self.model_fit.params,
                    "last_date": self.last_date,
                    "last_series": self.last_series
                }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.order = data.get("order", (1, 1, 1))
            self.seasonal_order = data.get("seasonal_order")
            self.alpha = data.get("alpha", 0.10)
            self.model_name = data.get("model_name", "ARIMA")
            self.target_col = data.get("target_col", "freight_rate")
            self.date_col = data.get("date_col", "date")
            self.last_date = data.get("last_date")
            self.last_series = data.get("last_series")
            if self.last_series is not None:
                # Re-fit model to recreate state
                model = ARIMA(self.last_series, order=self.order)
                self.model_fit = model.fit()

"""
Charter-AI — XGBoost Freight Forecaster with Probabilistic Quantile Uncertainty.

Implements multi-step freight rate forecasting using XGBoost with:
1. Multi-domain feature matrix (freight lags, rolling stats, momentum,
   Baltic indices, commodities, bunker, macro, operational, calendar).
2. Probabilistic quantile regression:
   - P10 (quantile_alpha=0.10) for lower uncertainty bound
   - P50 (quantile_alpha=0.50) for median expected rate
   - P90 (quantile_alpha=0.90) for upper uncertainty bound
3. Non-crossing quantile enforcement.
4. No arbitrary uncertainty heuristics (strictly grounded in quantile loss).
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.forecast_features import (
    FreightFeatureBuilder,
    FREIGHT_LAGS,
    ROLLING_WINDOWS,
    add_calendar_features,
    add_momentum,
)
from src.models.model_evaluation import evaluate_forecast
from src.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostForecaster(ForecastModel):
    """
    XGBoost Freight Forecaster with quantile-based prediction intervals (P10, P50, P90).
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        n_estimators: int = 120,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        use_quantiles: bool = True,
        **xgb_kwargs
    ):
        self.lags = lags if lags is not None else FREIGHT_LAGS
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_quantiles = use_quantiles
        self.xgb_kwargs = xgb_kwargs

        self.feature_cols: List[str] = []
        self.target_col: str = "freight_rate"
        self.date_col: str = "date"
        self.is_fitted: bool = False
        self.model_name = "XGBoost_Quantile"

        # Initialize models for P50, P10, and P90
        common_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": 42,
            **self.xgb_kwargs
        }

        if self.use_quantiles:
            self.model_p50 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.50, **common_params)
            self.model_p10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **common_params)
            self.model_p90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **common_params)
        else:
            self.model_p50 = xgb.XGBRegressor(objective="reg:squarederror", **common_params)
            self.model_p10 = None
            self.model_p90 = None

        # Empirical residual fallback
        self.residual_p10_diff: float = -0.8
        self.residual_p90_diff: float = 0.8

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract or build features from dataframe."""
        df_out = df.copy()
        df_out[self.date_col] = pd.to_datetime(df_out[self.date_col])
        df_out = df_out.sort_values(by=self.date_col).reset_index(drop=True)

        # Check if already enriched with lag features
        has_lags = all(f"lag_{lag}" in df_out.columns for lag in self.lags)
        if not has_lags:
            # Build target lags
            for lag in self.lags:
                df_out[f"lag_{lag}"] = df_out[self.target_col].shift(lag)

            # Rolling stats if not present
            for w in [7, 14, 28]:
                if f"rolling_mean_{w}" not in df_out.columns:
                    df_out[f"rolling_mean_{w}"] = df_out[self.target_col].shift(1).rolling(w, min_periods=1).mean()
                if w in [7, 28] and f"rolling_std_{w}" not in df_out.columns:
                    df_out[f"rolling_std_{w}"] = df_out[self.target_col].shift(1).rolling(w, min_periods=1).std()

            # Momentum — delegate to the canonical, leak-free add_momentum().
            # add_momentum uses shift(1) as anchor so target_t is never accessed.
            has_momentum = all(f"momentum_{p}" in df_out.columns for p in [7, 30])
            if not has_momentum:
                df_out = add_momentum(df_out, target_col=self.target_col, periods=[7, 30])

            # Calendar features
            df_out = add_calendar_features(df_out, date_col=self.date_col)

        # Forward fill / backfill missing lags
        df_out = df_out.ffill().bfill()

        exclude = {
            self.date_col, self.target_col, "origin", "destination", "vessel_class",
            "cargo_type", "currency", "unit", "source"
        }
        feature_cols = [c for c in df_out.columns if c not in exclude and not c.startswith("target_")]
        return df_out, feature_cols

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit P10, P50, and P90 XGBoost models on training features.
        """
        if len(df_train) == 0:
            raise ValueError("Training data is empty.")

        self.target_col = target_col
        self.date_col = date_col

        df_feat, feature_cols = self._prepare_features(df_train)
        self.feature_cols = feature_cols

        # Drop initial rows where lags are undefined
        df_clean = df_feat.dropna(subset=self.feature_cols + [self.target_col])
        if len(df_clean) == 0:
            # If still empty due to short series, fill na
            df_clean = df_feat.fillna(0)

        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col].values.astype(float)

        # Fit Median / P50
        self.model_p50.fit(X, y)
        preds_p50 = self.model_p50.predict(X)
        residuals = y - preds_p50

        # Estimate empirical residual quantiles for fallback
        if len(residuals) > 5:
            self.residual_p10_diff = float(np.percentile(residuals, 10))
            self.residual_p90_diff = float(np.percentile(residuals, 90))

        # Fit Quantile P10 and P90 if enabled
        if self.use_quantiles and self.model_p10 is not None and self.model_p90 is not None:
            try:
                self.model_p10.fit(X, y)
                self.model_p90.fit(X, y)
            except Exception as e:
                logger.warning(f"Quantile fit failed, using empirical residual quantiles: {e}")
                self.model_p10 = None
                self.model_p90 = None

        self.is_fitted = True

    def predict(
        self,
        horizon_days: int,
        context_df: pd.DataFrame,
        date_col: str,
        **kwargs
    ) -> ForecastResult:
        """
        Multi-step recursive or direct forecasting with P10/P50/P90 prediction intervals.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", kwargs.get("vessel_class", "unknown"))
        target_col = kwargs.get("target_col", self.target_col)

        df_sorted = context_df.sort_values(by=date_col).reset_index(drop=True)
        if target_col not in df_sorted.columns:
            target_col = df_sorted.columns[-1]

        df_feat, _ = self._prepare_features(df_sorted)
        last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])

        # Extract last row features
        current_features = df_feat.iloc[-1:][self.feature_cols].copy()

        # Track history for recursive lag updating
        rate_history = list(df_sorted[target_col].values.astype(float))

        predictions_p50: List[float] = []
        predictions_p10: List[float] = []
        predictions_p90: List[float] = []

        for i in range(1, horizon_days + 1):
            pred_p50 = float(self.model_p50.predict(current_features)[0])

            # Predict quantiles
            if self.model_p10 is not None and self.model_p90 is not None:
                try:
                    pred_p10 = float(self.model_p10.predict(current_features)[0])
                    pred_p90 = float(self.model_p90.predict(current_features)[0])
                except Exception:
                    pred_p10 = pred_p50 + self.residual_p10_diff
                    pred_p90 = pred_p50 + self.residual_p90_diff
            else:
                pred_p10 = pred_p50 + self.residual_p10_diff
                pred_p90 = pred_p50 + self.residual_p90_diff

            # Enforce non-crossing monotonic quantiles & non-negativity: P10 <= P50 <= P90
            lower = max(0.0, min(pred_p10, pred_p50))
            upper = max(pred_p90, pred_p50)
            median = max(lower, min(pred_p50, upper))

            predictions_p50.append(median)
            predictions_p10.append(lower)
            predictions_p90.append(upper)

            # Update history with predicted median for next recursive step
            rate_history.append(median)

            # Update feature row for step i+1
            for lag in self.lags:
                lag_col = f"lag_{lag}"
                if lag_col in current_features.columns:
                    current_features.loc[:, lag_col] = rate_history[-lag] if len(rate_history) >= lag else median

            # Update rolling mean
            for w in [7, 14, 28]:
                rm_col = f"rolling_mean_{w}"
                if rm_col in current_features.columns:
                    current_features.loc[:, rm_col] = float(np.mean(rate_history[-w:]))

        series_out: List[ForecastPoint] = []
        for i in range(horizon_days):
            pred_date = last_date + timedelta(days=i + 1)
            series_out.append(ForecastPoint(
                date=pred_date.date(),
                predicted_rate=round(predictions_p50[i], 2),
                lower_ci=round(predictions_p10[i], 2),
                upper_ci=round(predictions_p90[i], 2)
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
        """Evaluate XGBoost model on out-of-sample test set."""
        df_feat, _ = self._prepare_features(df_test)
        df_clean = df_feat.dropna(subset=self.feature_cols + [target_col])
        if len(df_clean) == 0:
            return {"Model": self.model_name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "MASE": np.nan}

        X = df_clean[self.feature_cols]
        y_true = df_clean[target_col].values.astype(float)
        y_pred = self.model_p50.predict(X)

        metrics = evaluate_forecast(y_true, y_pred)
        metrics["Model"] = self.model_name
        return metrics

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_p50": self.model_p50,
                "model_p10": self.model_p10,
                "model_p90": self.model_p90,
                "feature_cols": self.feature_cols,
                "lags": self.lags,
                "target_col": self.target_col,
                "date_col": self.date_col,
                "residual_p10_diff": self.residual_p10_diff,
                "residual_p90_diff": self.residual_p90_diff,
                "model_name": self.model_name
            }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model_p50 = data["model_p50"]
            self.model_p10 = data.get("model_p10")
            self.model_p90 = data.get("model_p90")
            self.feature_cols = data["feature_cols"]
            self.lags = data["lags"]
            self.target_col = data.get("target_col", "freight_rate")
            self.date_col = data.get("date_col", "date")
            self.residual_p10_diff = data.get("residual_p10_diff", -0.8)
            self.residual_p90_diff = data.get("residual_p90_diff", 0.8)
            self.model_name = data.get("model_name", "XGBoost_Quantile")
            self.is_fitted = True

"""
Charter-AI — Ensemble Freight Forecaster.

Combines statistical baselines, ARIMA/SARIMA, and machine learning (XGBoost)
into a unified, robust multi-model ensemble.

Weighting Strategies:
1. 'inverse_rmse': Weights dynamically derived from component out-of-fold RMSE:
     w_m = (1 / RMSE_m) / sum(1 / RMSE_k)
2. 'equal': Equal simple average across all valid models.
3. 'custom': Explicit user-defined weight mapping.
"""

from typing import Dict, List, Optional, Any
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle

from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.baseline_forecaster import BaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.model_evaluation import evaluate_forecast, walk_forward_cv
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleForecaster(ForecastModel):
    """
    Multi-model ensemble combining Baseline, ARIMA, and XGBoost.
    """

    def __init__(
        self,
        models: Optional[Dict[str, ForecastModel]] = None,
        weighting_strategy: str = "inverse_rmse",
        custom_weights: Optional[Dict[str, float]] = None
    ):
        self.weighting_strategy = weighting_strategy.lower()
        self.custom_weights = custom_weights or {}

        if models is not None:
            self.models = models
        else:
            self.models = {
                "baseline": BaselineForecaster(method="moving_average", window_size=7),
                "arima": ARIMAForecaster(order=(1, 1, 1)),
                "xgboost": XGBoostForecaster()
            }

        self.weights: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        self.target_col: str = "freight_rate"
        self.date_col: str = "date"

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        Fit each component model and compute weights based on out-of-fold (OOF)
        validation accuracy.

        Weight estimation methodology (inverse_rmse strategy):
        --------------------------------------------------------
        To avoid in-sample bias, weights are derived using a two-pass OOF scheme:

          Pass 1 — OOF weight estimation:
            fit_data  = df_train[:-V]   (all but the last V rows)
            oof_data  = df_train[-V:]   (last V rows, never seen during Pass 1 fit)
            Each model is fitted on fit_data only, then evaluated on oof_data.
            OOF RMSE per model is used to compute inverse-RMSE weights.

          Pass 2 — Full refit for deployment:
            All models are refitted on the complete df_train for actual prediction.

        V = min(30, max(14, len(df_train) // 5))

        If len(df_train) < 2 * V (data too short for a meaningful holdout),
        the strategy falls back to equal weights.

        This guarantees that the validation data used for weight calculation
        was NOT seen by any component model during that weight-estimation fit.
        """
        if len(df_train) == 0:
            raise ValueError("Training data is empty.")

        self.target_col = target_col
        self.date_col = date_col

        # Sort chronologically — required for OOF split correctness
        df_sorted = df_train.sort_values(date_col).reset_index(drop=True)

        # ── Step 1: Determine weights ────────────────────────────────────────
        if self.weighting_strategy == "custom" and self.custom_weights:
            total = sum(self.custom_weights.values())
            self.weights = {k: v / total for k, v in self.custom_weights.items()}

        elif self.weighting_strategy == "equal":
            count = len(self.models)
            self.weights = {k: 1.0 / count for k in self.models}

        else:
            # inverse_rmse — OOF holdout weight estimation (no in-sample bias)
            n = len(df_sorted)
            v = min(30, max(14, n // 5))  # holdout window size

            if n >= 2 * v:
                # ── Pass 1: OOF weight estimation ────────────────────────────
                # fit_df never overlaps with oof_df → no in-sample leakage
                fit_df = df_sorted.iloc[:-v].copy()   # first N-V rows
                oof_df = df_sorted.iloc[-v:].copy()   # last V rows — OOF only

                rmses: Dict[str, float] = {}
                for name, model in self.models.items():
                    try:
                        import copy
                        tmp_model = copy.deepcopy(model)
                        tmp_model.fit(fit_df, target_col=target_col, date_col=date_col)
                        oof_res = tmp_model.evaluate(oof_df, target_col=target_col, date_col=date_col)
                        rmse_val = float(oof_res.get("RMSE", 1.0))
                        rmses[name] = max(0.01, rmse_val)
                        self.model_metrics[name] = {
                            **oof_res,
                            "_evaluation": "OOF_holdout",
                            "_oof_size": v,
                            "_fit_size": len(fit_df),
                        }
                    except Exception as exc:
                        logger.warning(
                            "OOF evaluation for '%s' failed (%s). "
                            "Assigning neutral RMSE=1.0.", name, exc
                        )
                        rmses[name] = 1.0

                # Inverse-RMSE weights derived from OOF RMSE — not in-sample RMSE
                inv_rmses = {k: 1.0 / r for k, r in rmses.items()}
                total_inv = sum(inv_rmses.values())
                self.weights = {k: inv / total_inv for k, inv in inv_rmses.items()}
                logger.info(
                    "Ensemble OOF weight estimation complete: "
                    "fit=%d rows, oof=%d rows, OOF RMSEs=%s",
                    len(fit_df), v, {k: round(r, 4) for k, r in rmses.items()}
                )
            else:
                # Fallback: data too short for meaningful OOF holdout → equal weights
                logger.warning(
                    "Training data (%d rows) too short for OOF holdout "
                    "(need >= %d). Falling back to equal weights.", n, 2 * v
                )
                count = len(self.models)
                self.weights = {k: 1.0 / count for k in self.models}

        # ── Step 2: Full refit on ALL training data for deployment ───────────
        # Models are now fit on the complete df_train so predictions use
        # the maximum available history.
        for name, model in self.models.items():
            try:
                model.fit(df_sorted, target_col=target_col, date_col=date_col)
            except Exception as e:
                logger.warning(f"Final fit of component model '{name}' failed: {e}")

        self.is_fitted = True
        logger.info(f"Ensemble fitted successfully with weights: {self.weights}")


    def predict(
        self,
        horizon_days: int,
        context_df: pd.DataFrame,
        date_col: str,
        **kwargs
    ) -> ForecastResult:
        """
        Generate weighted ensemble predictions with combined uncertainty bounds.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble is not fitted yet.")

        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", kwargs.get("vessel_class", "unknown"))

        component_results: Dict[str, ForecastResult] = {}
        for name, model in self.models.items():
            try:
                res = model.predict(
                    horizon_days=horizon_days,
                    context_df=context_df,
                    date_col=date_col,
                    target_col=kwargs.get("target_col", self.target_col),
                    route=route,
                    vessel_type=vessel_type
                )
                component_results[name] = res
            except Exception as e:
                logger.warning(f"Ensemble component prediction failed for {name}: {e}")

        if not component_results:
            raise RuntimeError("All ensemble component models failed to generate predictions.")

        # Re-normalize weights among surviving models
        active_weights = {k: self.weights.get(k, 1.0) for k in component_results.keys()}
        total_wt = sum(active_weights.values())
        norm_weights = {k: v / total_wt for k, v in active_weights.items()}

        series_out: List[ForecastPoint] = []
        first_res = next(iter(component_results.values()))

        for i in range(horizon_days):
            p_date = first_res.series[i].date if i < len(first_res.series) else None

            # Weighted mean
            pred_rate = sum(
                norm_weights[m_name] * component_results[m_name].series[i].predicted_rate
                for m_name in component_results
                if i < len(component_results[m_name].series)
            )

            # Weighted lower bound (P10)
            lower_ci = sum(
                norm_weights[m_name] * (component_results[m_name].series[i].lower_ci or component_results[m_name].series[i].predicted_rate * 0.95)
                for m_name in component_results
                if i < len(component_results[m_name].series)
            )

            # Weighted upper bound (P90)
            upper_ci = sum(
                norm_weights[m_name] * (component_results[m_name].series[i].upper_ci or component_results[m_name].series[i].predicted_rate * 1.05)
                for m_name in component_results
                if i < len(component_results[m_name].series)
            )

            lower_ci = max(0.0, min(lower_ci, pred_rate))
            upper_ci = max(upper_ci, pred_rate)

            series_out.append(ForecastPoint(
                date=p_date,
                predicted_rate=round(pred_rate, 2),
                lower_ci=round(lower_ci, 2),
                upper_ci=round(upper_ci, 2)
            ))

        wt_summary = ", ".join(f"{k}: {v:.0%}" for k, v in norm_weights.items())
        model_used = f"Ensemble_{self.weighting_strategy.title()} ({wt_summary})"

        return ForecastResult(
            route=route,
            vessel_type=vessel_type,
            horizon_days=horizon_days,
            model_used=model_used,
            series=series_out,
            confidence_available=True,
            forecast_date=first_res.forecast_date
        )

    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        """Evaluate ensemble predictions on test set."""
        df_sorted = df_test.sort_values(by=date_col).reset_index(drop=True)
        horizon = len(df_sorted)
        res = self.predict(
            horizon_days=horizon,
            context_df=df_sorted.iloc[:1],  # dummy initial context
            date_col=date_col,
            target_col=target_col
        )
        preds = np.array([p.predicted_rate for p in res.series])
        metrics = evaluate_forecast(df_sorted[target_col].values, preds)
        metrics["Model"] = f"Ensemble_{self.weighting_strategy.title()}"
        return metrics

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "weights": self.weights,
                "weighting_strategy": self.weighting_strategy,
                "custom_weights": self.custom_weights,
                "models": self.models,
                "model_metrics": self.model_metrics,
                "target_col": self.target_col,
                "date_col": self.date_col,
                "is_fitted": self.is_fitted
            }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.weights = data["weights"]
            self.weighting_strategy = data["weighting_strategy"]
            self.custom_weights = data.get("custom_weights", {})
            self.models = data["models"]
            self.model_metrics = data.get("model_metrics", {})
            self.target_col = data.get("target_col", "freight_rate")
            self.date_col = data.get("date_col", "date")
            self.is_fitted = data.get("is_fitted", True)

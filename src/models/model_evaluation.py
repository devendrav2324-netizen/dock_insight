"""
DockInsights — Model Evaluation & Time-Series Walk-Forward Cross-Validation.

Provides:
1. Standard Error Metrics: MAE, RMSE, MAPE, sMAPE, MASE
2. Strict Chronological Walk-Forward Cross-Validation (expanding & rolling windows)
3. Model Comparison and Ranking utilities

Evaluation Methodology
-----------------------
All evaluation in this module is strictly OUT-OF-SAMPLE and forward-looking.

walk_forward_cv() guarantee:
  - The dataset is sorted chronologically before splitting.
  - Each fold trains on rows[0:cutoff] and evaluates on rows[cutoff:cutoff+horizon].
  - No future observations are ever accessible to the model during training.
  - MASE scale is computed from the training window only (y_train argument),
    never from the test window.
  - At no point is the same data used for both fitting and evaluation.

Caller contract for model.evaluate():
  - The caller is responsible for ensuring that df_test was NOT used during
    model.fit(). Passing training data to model.evaluate() produces in-sample
    (optimistic) metrics and is incorrect.
  - EnsembleForecaster.fit() uses an internal OOF holdout for weight estimation
    so that weights are never derived from training data predictions.
  - CongestionPredictor.fit() uses an internal 80/20 chronological split
    for its stored self.metrics.

Rules:
- NEVER use random train/test splitting for time series.
- All reported metrics must be based on OOS predictions.
"""

from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (%).
    Safely handles zeros in y_true.
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE, %).
    Bounded between 0% and 200%. Handles zeros safely.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    Scales error by in-sample 1-step naive forecast error.
    MASE < 1 means model is better than naive baseline.
    """
    mae_test = calculate_mae(y_true, y_pred)

    if y_train is not None and len(y_train) > 1:
        scale = np.mean(np.abs(np.diff(y_train)))
    elif len(y_true) > 1:
        scale = np.mean(np.abs(np.diff(y_true)))
    else:
        scale = 1.0

    if scale < 1e-8:
        return float(mae_test)
    return float(mae_test / scale)


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
    y_train: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive forecasting metrics given true and predicted series.
    Returns MAE, RMSE, MAPE, sMAPE, and MASE.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) != len(y_p):
        raise ValueError(f"y_true ({len(y_t)}) and y_pred ({len(y_p)}) must have the same length.")

    if len(y_t) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "sMAPE": 0.0, "MASE": 0.0}

    y_tr = np.asarray(y_train, dtype=float) if y_train is not None else None

    metrics = {
        "MAE": calculate_mae(y_t, y_p),
        "RMSE": calculate_rmse(y_t, y_p),
        "MAPE": calculate_mape(y_t, y_p),
        "sMAPE": calculate_smape(y_t, y_p),
        "MASE": calculate_mase(y_t, y_p, y_tr)
    }
    return metrics


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a comparison report across multiple models sorted by MAE.
    """
    df = pd.DataFrame(results)
    if not df.empty and "MAE" in df.columns:
        df = df.sort_values("MAE").reset_index(drop=True)
    return df


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, float]
    actuals: List[float]
    predictions: List[float]


@dataclass
class WalkForwardCVResult:
    model_name: str
    horizon_days: int
    folds: List[WalkForwardFold] = field(default_factory=list)
    overall_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def mae(self) -> float:
        return self.overall_metrics.get("MAE", 0.0)

    @property
    def rmse(self) -> float:
        return self.overall_metrics.get("RMSE", 0.0)


def walk_forward_cv(
    model_factory: Union[Callable[[], Any], Any],
    df: pd.DataFrame,
    target_col: str = "freight_rate",
    date_col: str = "date",
    horizon_days: int = 7,
    initial_train_size: int = 180,
    step_size: int = 14,
    window_type: str = "expanding",
    **predict_kwargs
) -> WalkForwardCVResult:
    """
    Perform rigorous chronological walk-forward cross-validation.

    Args:
        model_factory: Callable creating a fresh ForecastModel instance or a class with fit/predict.
        df: DataFrame sorted chronologically containing date_col and target_col.
        target_col: Column name of target rate.
        date_col: Column name of dates.
        horizon_days: Number of steps to forecast in each fold.
        initial_train_size: Minimum number of historical observations for initial training window.
        step_size: Number of periods to roll forward between folds.
        window_type: 'expanding' (uses all past history) or 'rolling' (fixed initial_train_size).
        **predict_kwargs: Extra parameters passed to model.predict (e.g. route, vessel_type).

    Returns:
        WalkForwardCVResult with fold details and aggregated cross-validation metrics.
    """
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

    n_samples = len(df_sorted)
    if n_samples < initial_train_size + horizon_days:
        raise ValueError(
            f"Dataset length ({n_samples}) is too short for initial_train_size ({initial_train_size}) + horizon ({horizon_days})"
        )

    folds: List[WalkForwardFold] = []
    all_actuals: List[float] = []
    all_preds: List[float] = []
    all_train_history: List[float] = []

    current_cutoff = initial_train_size
    fold_idx = 0

    while current_cutoff + horizon_days <= n_samples:
        # Define train and test slices
        train_start_idx = 0 if window_type == "expanding" else (current_cutoff - initial_train_size)
        train_df = df_sorted.iloc[train_start_idx:current_cutoff].copy()
        test_df = df_sorted.iloc[current_cutoff:current_cutoff + horizon_days].copy()

        # Instantiate fresh model
        if callable(model_factory):
            model = model_factory()
        else:
            import copy
            model = copy.deepcopy(model_factory)

        # Fit model on strictly past data
        model.fit(train_df, target_col=target_col, date_col=date_col)

        # Predict multi-step horizon using train_df as context
        forecast_res = model.predict(
            horizon_days=horizon_days,
            context_df=train_df,
            date_col=date_col,
            target_col=target_col,
            **predict_kwargs
        )

        y_test = test_df[target_col].values.astype(float)
        y_pred = np.array([p.predicted_rate for p in forecast_res.series], dtype=float)

        # If model returned fewer points than requested horizon, handle gracefully
        if len(y_pred) != len(y_test):
            min_len = min(len(y_pred), len(y_test))
            y_test = y_test[:min_len]
            y_pred = y_pred[:min_len]

        fold_metrics = evaluate_forecast(y_test, y_pred, y_train=train_df[target_col].values)

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_start=str(train_df[date_col].iloc[0].date()),
            train_end=str(train_df[date_col].iloc[-1].date()),
            test_start=str(test_df[date_col].iloc[0].date()),
            test_end=str(test_df[date_col].iloc[-1].date()),
            metrics=fold_metrics,
            actuals=y_test.tolist(),
            predictions=y_pred.tolist()
        ))

        all_actuals.extend(y_test.tolist())
        all_preds.extend(y_pred.tolist())
        all_train_history.extend(train_df[target_col].values.tolist())

        current_cutoff += step_size
        fold_idx += 1

    overall_metrics = evaluate_forecast(
        np.array(all_actuals),
        np.array(all_preds),
        y_train=np.array(all_train_history)
    )

    model_name = getattr(model, "model_used", getattr(model, "__class__", type("Model", (), {}))). __name__
    if hasattr(forecast_res, "model_used") and forecast_res.model_used:
        model_name = forecast_res.model_used

    return WalkForwardCVResult(
        model_name=model_name,
        horizon_days=horizon_days,
        folds=folds,
        overall_metrics=overall_metrics
    )

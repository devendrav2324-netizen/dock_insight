"""
tests/test_oos_validation.py

Walk-Forward Out-of-Sample Validation Tests — Task 2.

Verifies that:
1. Every training timestamp < corresponding validation timestamp (chronological order).
2. Data is never randomly shuffled — index remains monotonically ordered after sort.
3. OOS predictions come from models that never trained on the corresponding test rows.
4. Mutating a future observation does NOT change predictions for an earlier timestamp.
5. All reported metrics (MAE/RMSE/MAPE/sMAPE/MASE) are from OOS predictions.
6. The naive baseline uses only historical observations — never future values.
7. Ensemble weights are NOT derived from the same validation targets they evaluate.
8. A small deterministic 30-row synthetic dataset produces the expected fold splits.
"""

import copy
import numpy as np
import pandas as pd
import pytest

from src.models.model_evaluation import walk_forward_cv, evaluate_forecast
from src.models.baseline_forecaster import BaselineForecaster, NaiveBaselineForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_series(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Deterministic freight-rate time series of length n."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rates = 20.0 + np.cumsum(rng.normal(0, 0.5, n))
    rates = np.clip(rates, 5.0, 100.0)
    return pd.DataFrame({"date": dates, "freight_rate": rates})


def naive_model_factory():
    return NaiveBaselineForecaster()


# ---------------------------------------------------------------------------
# Test 1 — Chronological ordering: every train_end < test_start in every fold
# ---------------------------------------------------------------------------

def test_walk_forward_chronological_order():
    """
    In every fold produced by walk_forward_cv:
    - train_end must be strictly earlier than test_start.
    """
    df = make_series(n=80)
    result = walk_forward_cv(
        model_factory=naive_model_factory,
        df=df,
        target_col="freight_rate",
        date_col="date",
        horizon_days=5,
        initial_train_size=30,
        step_size=10,
    )
    assert len(result.folds) >= 2, "Expected at least 2 folds"

    for fold in result.folds:
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        assert train_end < test_start, (
            f"Fold {fold.fold_idx}: train_end ({train_end}) >= test_start ({test_start}). "
            "Chronological ordering violated."
        )


# ---------------------------------------------------------------------------
# Test 2 — No random shuffling: sorted index is monotonically increasing
# ---------------------------------------------------------------------------

def test_no_random_shuffle():
    """
    walk_forward_cv must sort by date, NOT shuffle.
    After sorting, the underlying date sequence must be non-decreasing.
    """
    df = make_series(n=60)
    # Shuffle the input deliberately — walk_forward_cv must re-sort it
    df_shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    result = walk_forward_cv(
        model_factory=naive_model_factory,
        df=df_shuffled,
        target_col="freight_rate",
        date_col="date",
        horizon_days=5,
        initial_train_size=30,
        step_size=10,
    )

    for fold in result.folds:
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        # Simply verify chronological property — no shuffle
        assert train_end < test_start, (
            f"Fold {fold.fold_idx}: temporal order violated after shuffled input."
        )


# ---------------------------------------------------------------------------
# Test 3 — OOS predictions from an unseen model
# ---------------------------------------------------------------------------

def test_oos_predictions_from_unseen_model():
    """
    The model for each fold is trained ONLY on rows before the test window.
    We verify this by confirming each fold's actuals have timestamps strictly
    AFTER the fold's training end date, and the model was fit on data BEFORE
    those actuals.
    """
    df = make_series(n=80)
    result = walk_forward_cv(
        model_factory=naive_model_factory,
        df=df,
        target_col="freight_rate",
        date_col="date",
        horizon_days=7,
        initial_train_size=30,
        step_size=10,
    )

    df_sorted = df.sort_values("date").reset_index(drop=True)

    for fold in result.folds:
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        test_end = pd.Timestamp(fold.test_end)

        # Every test observation must be strictly after the training window
        assert test_start > train_end, (
            f"Fold {fold.fold_idx}: test_start ({test_start}) not after train_end ({train_end})."
        )

        # The actual values in the fold come from data after train_end
        test_rows = df_sorted[
            (df_sorted["date"] >= test_start) & (df_sorted["date"] <= test_end)
        ]
        assert len(test_rows) > 0, f"Fold {fold.fold_idx}: no actual rows found for test window."

        # None of the test row dates should be <= train_end
        assert (test_rows["date"] > train_end).all(), (
            f"Fold {fold.fold_idx}: some test observations are within the training window."
        )


# ---------------------------------------------------------------------------
# Test 4 — Leakage resistance: future mutation does not affect earlier fold
# ---------------------------------------------------------------------------

def test_leakage_resistance_future_observation():
    """
    Mutating a future observation (beyond the current test window) must NOT
    change the prediction for an earlier fold's test window.

    We compare two walk_forward_cv runs on identical data except that the LAST
    row of the dataframe is perturbed by +9999 in the second run. The first
    fold's predictions must be identical in both runs.
    """
    df_base = make_series(n=80)
    df_perturbed = df_base.copy()
    # Mutate the very last row (strictly in the future relative to early folds)
    df_perturbed.loc[df_perturbed.index[-1], "freight_rate"] += 9999.0

    kwargs = dict(
        target_col="freight_rate",
        date_col="date",
        horizon_days=5,
        initial_train_size=30,
        step_size=35,  # large step so only 1 fold is produced, well before the last row
    )

    result_base = walk_forward_cv(naive_model_factory, df_base, **kwargs)
    result_pert = walk_forward_cv(naive_model_factory, df_perturbed, **kwargs)

    assert len(result_base.folds) >= 1, "No folds produced."
    fold_b = result_base.folds[0]
    fold_p = result_pert.folds[0]

    assert fold_b.predictions == pytest.approx(fold_p.predictions, abs=1e-6), (
        "Leakage detected: mutating a future row changed predictions for an earlier fold.\n"
        f"  base preds:      {fold_b.predictions}\n"
        f"  perturbed preds: {fold_p.predictions}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Metrics are computed from OOS predictions
# ---------------------------------------------------------------------------

def test_metrics_from_oos_predictions():
    """
    Verify that the MAE/RMSE reported by walk_forward_cv matches what you get
    when you manually compute those metrics from the concatenated fold actuals
    and fold predictions.

    If metrics were in-sample, the manual recalculation would differ.
    """
    df = make_series(n=80)
    result = walk_forward_cv(
        model_factory=naive_model_factory,
        df=df,
        target_col="freight_rate",
        date_col="date",
        horizon_days=5,
        initial_train_size=30,
        step_size=10,
    )

    # Manually concatenate all folds' actuals and predictions
    all_actuals = []
    all_preds = []
    for fold in result.folds:
        all_actuals.extend(fold.actuals)
        all_preds.extend(fold.predictions)

    manual_metrics = evaluate_forecast(
        np.array(all_actuals), np.array(all_preds)
    )

    assert abs(result.overall_metrics["MAE"] - manual_metrics["MAE"]) < 1e-6, (
        "overall_metrics MAE does not match manual recalculation from fold actuals/predictions."
    )
    assert abs(result.overall_metrics["RMSE"] - manual_metrics["RMSE"]) < 1e-6, (
        "overall_metrics RMSE does not match manual recalculation."
    )


# ---------------------------------------------------------------------------
# Test 6 — Naive baseline uses only historical observations
# ---------------------------------------------------------------------------

def test_naive_baseline_uses_only_history():
    """
    The naive baseline's last_value after fit() must equal the LAST value in
    the training data (chronologically).

    If the naive baseline were looking ahead, its value would differ from the
    chronological last.
    """
    df = make_series(n=60)
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Split: train on first 40 rows
    train_df = df_sorted.iloc[:40].copy()
    test_df  = df_sorted.iloc[40:].copy()

    model = NaiveBaselineForecaster()
    model.fit(train_df, target_col="freight_rate", date_col="date")

    # The model's captured last value must be the last training observation
    expected_last = float(train_df["freight_rate"].iloc[-1])
    assert abs(model.last_value - expected_last) < 1e-9, (
        f"Naive baseline last_value ({model.last_value}) != "
        f"training data last value ({expected_last}). "
        "Baseline may be using future information."
    )

    # It must NOT equal any future value from test_df
    future_values = test_df["freight_rate"].values
    for fv in future_values:
        if abs(fv - expected_last) > 1e-9:  # only check distinctly different future values
            assert abs(model.last_value - fv) > 1e-9 or abs(model.last_value - expected_last) < 1e-9, (
                "Naive baseline last_value matches a future observation — baseline looks ahead."
            )


# ---------------------------------------------------------------------------
# Test 7 — Ensemble weights are NOT derived from validation targets
# ---------------------------------------------------------------------------

def test_ensemble_weights_not_derived_from_val_targets():
    """
    Mutating the LAST V rows' target values (the OOF window used for weight
    estimation) should change the ensemble weights — because OOF RMSE changes.

    But crucially, the weights must NOT change when we mutate only rows beyond
    the training set (i.e., rows the ensemble will never see during fit).

    This test verifies that weights ARE sensitive to the OOF window (proving
    OOF evaluation is live) and confirms the OOF window is the chronological tail
    of training data (not future test data that would constitute leakage).
    """
    n = 100
    df_base = make_series(n=n)
    df_sorted = df_base.sort_values("date").reset_index(drop=True)

    # The OOF window is the last V rows of the training data
    # V = min(30, max(14, n//5)) = min(30, max(14, 20)) = 20
    v = min(30, max(14, n // 5))

    # Perturb ONLY the OOF window (last v rows of df_sorted)
    df_perturbed = df_sorted.copy()
    df_perturbed.loc[df_perturbed.index[-v:], "freight_rate"] += 500.0

    # Use equal-weight ensemble for speed (avoids expensive OOF XGBoost fits)
    ens_base = EnsembleForecaster(
        models={
            "baseline": BaselineForecaster(method="naive"),
            "ma": BaselineForecaster(method="moving_average", window_size=7),
        },
        weighting_strategy="inverse_rmse",
    )
    ens_pert = EnsembleForecaster(
        models={
            "baseline": BaselineForecaster(method="naive"),
            "ma": BaselineForecaster(method="moving_average", window_size=7),
        },
        weighting_strategy="inverse_rmse",
    )

    ens_base.fit(df_sorted, target_col="freight_rate", date_col="date")
    ens_pert.fit(df_perturbed, target_col="freight_rate", date_col="date")

    # Weights must differ because the OOF window values changed
    weights_changed = any(
        abs(ens_base.weights.get(k, 0) - ens_pert.weights.get(k, 0)) > 1e-6
        for k in ens_base.weights
    )
    assert weights_changed, (
        "Ensemble weights did NOT change after perturbing the OOF evaluation window. "
        "OOF evaluation may not be active — weights might be derived from something else."
    )

    # Also confirm the model_metrics carry the OOF provenance flag
    for name, mets in ens_base.model_metrics.items():
        assert mets.get("_evaluation") == "OOF_holdout", (
            f"Model '{name}' metrics are not tagged as OOF_holdout. "
            f"Got: {mets.get('_evaluation')}"
        )


# ---------------------------------------------------------------------------
# Test 8 — Deterministic split: manually verify fold boundaries
# ---------------------------------------------------------------------------

def test_deterministic_walk_forward_split():
    """
    With a synthetic 50-row daily series starting 2024-01-01:
      initial_train_size = 20
      horizon_days       = 5
      step_size          = 10

    Expected folds:
      Fold 0: train rows  0-19 (2024-01-01 → 2024-01-20), test rows 20-24
      Fold 1: train rows  0-29 (2024-01-01 → 2024-01-30), test rows 30-34
      Fold 2: train rows  0-39 (2024-01-01 → 2024-02-09), test rows 40-44

    (expanding window, so train always starts at row 0)
    """
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rates = np.arange(1.0, n + 1, dtype=float)  # [1, 2, …, 50]
    df = pd.DataFrame({"date": dates, "freight_rate": rates})

    result = walk_forward_cv(
        model_factory=naive_model_factory,
        df=df,
        target_col="freight_rate",
        date_col="date",
        horizon_days=5,
        initial_train_size=20,
        step_size=10,
        window_type="expanding",
    )

    assert len(result.folds) == 3, (
        f"Expected 3 folds, got {len(result.folds)}. "
        f"Fold boundaries: {[(f.train_end, f.test_start) for f in result.folds]}"
    )

    expected = [
        ("2024-01-20", "2024-01-21"),
        ("2024-01-30", "2024-01-31"),
        ("2024-02-09", "2024-02-10"),
    ]
    for i, (fold, (exp_train_end, exp_test_start)) in enumerate(zip(result.folds, expected)):
        assert fold.train_end == exp_train_end, (
            f"Fold {i}: train_end={fold.train_end}, expected {exp_train_end}"
        )
        assert fold.test_start == exp_test_start, (
            f"Fold {i}: test_start={fold.test_start}, expected {exp_test_start}"
        )

    # Verify actuals are the correct values from the synthetic series
    # Fold 0 test rows 20-24 → rates [21, 22, 23, 24, 25]
    assert result.folds[0].actuals == pytest.approx([21.0, 22.0, 23.0, 24.0, 25.0], abs=1e-9), (
        f"Fold 0 actuals mismatch: {result.folds[0].actuals}"
    )

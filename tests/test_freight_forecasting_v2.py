"""
DockInsights — Comprehensive Test Suite for Phase 3 REAL Freight Forecasting Engine.

Tests:
1. Multi-domain feature engineering (lags, rolling stats, momentum, indices, commodities, bunker, macro, operational, calendar)
2. Evaluation metrics (MAE, RMSE, MAPE, sMAPE, MASE)
3. Time-series walk-forward cross-validation (strict chronological splits)
4. Baselines (Naive, Moving Average, Seasonal)
5. SARIMA/ARIMA forecaster
6. XGBoost with quantile regression (P10, P50, P90)
7. Ensemble forecaster (inverse-RMSE & equal weighting)
8. FreightForecaster orchestrator & canonical API contract
9. Model persistence and metadata schema
10. FastAPI HTTP endpoint (/api/v1/forecast)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

from src.models.base_forecaster import ForecastPoint, ForecastResult, ForecastModel
from src.models.baseline_forecaster import (
    BaselineForecaster,
    NaiveBaselineForecaster,
    MovingAverageForecaster,
    SeasonalBaselineForecaster
)
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.forecast_features import (
    add_freight_lags,
    add_rolling_metrics,
    add_momentum,
    add_calendar_features,
    FreightFeatureBuilder,
    build_forecast_features
)
from src.models.model_evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_smape,
    calculate_mase,
    evaluate_forecast,
    walk_forward_cv,
    compare_models
)
from src.models.freight_forecaster import FreightForecaster
from src.services.freight_forecast_service import FreightForecastService
from src.api.main import app


@pytest.fixture
def sample_daily_timeseries():
    """Generates 90 days of realistic daily freight series."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    base = 15.0
    trend = np.linspace(0, 2.0, 90)
    cycle = np.sin(np.linspace(0, 4 * np.pi, 90)) * 0.8
    noise = np.random.normal(0, 0.15, 90)
    rates = base + trend + cycle + noise
    return pd.DataFrame({
        "date": dates,
        "freight_rate": rates,
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Capesize",
        "cargo_type": "thermal_coal"
    })


# =============================================================================
# 1. Feature Engineering Tests
# =============================================================================

def test_feature_engineering_lags(sample_daily_timeseries):
    df = add_freight_lags(sample_daily_timeseries, target_col="freight_rate", lags=[1, 3, 7, 14, 21, 28])
    for lag in [1, 3, 7, 14, 21, 28]:
        assert f"lag_{lag}" in df.columns
        # Shift property check
        assert df[f"lag_{lag}"].iloc[lag] == df["freight_rate"].iloc[0]


def test_feature_engineering_rolling_and_momentum(sample_daily_timeseries):
    df = add_rolling_metrics(sample_daily_timeseries, target_col="freight_rate", windows=[7, 14, 28])
    assert "rolling_mean_7" in df.columns
    assert "rolling_mean_14" in df.columns
    assert "rolling_mean_28" in df.columns
    assert "rolling_std_7" in df.columns
    assert "rolling_std_28" in df.columns

    df = add_momentum(df, target_col="freight_rate", periods=[7, 30])
    assert "momentum_7" in df.columns
    assert "momentum_30" in df.columns

    # -----------------------------------------------------------------------
    # Correct formula check: momentum_7 at index i uses only t-1 and t-8,
    # i.e.  rate_(i-1) - rate_(i-8)   (NOT rate_i - rate_(i-7)).
    # -----------------------------------------------------------------------
    i = 9  # first row where both shift(1) and shift(8) are defined
    expected = df["freight_rate"].iloc[i - 1] - df["freight_rate"].iloc[i - 8]
    assert np.isclose(df["momentum_7"].iloc[i], expected, atol=1e-5), (
        f"momentum_7 leakage: expected {expected}, got {df['momentum_7'].iloc[i]}"
    )


# =============================================================================
# TARGET LEAKAGE TESTS FOR MOMENTUM FEATURES
# =============================================================================

def test_momentum_does_not_use_target_t():
    """
    Verify that momentum features are independent of the current target value.

    Construct two dataframes that differ ONLY in the last row's freight_rate
    (i.e. rate_t).  The momentum feature at that last row must be identical in
    both dataframes because momentum should only use rate_(t-1) and rate_(t-1-p).
    """
    np.random.seed(0)
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rates = np.arange(1.0, n + 1, dtype=float)  # deterministic, easy to trace

    df_base = pd.DataFrame({"date": dates, "freight_rate": rates.copy()})
    df_perturbed = df_base.copy()
    # Change only the last row's target value (rate_t)
    df_perturbed.loc[df_perturbed.index[-1], "freight_rate"] += 999.0

    feat_base = add_momentum(df_base, target_col="freight_rate", periods=[7, 30])
    feat_pert = add_momentum(df_perturbed, target_col="freight_rate", periods=[7, 30])

    last = n - 1
    for p in [7, 30]:
        col = f"momentum_{p}"
        assert np.isclose(feat_base[col].iloc[last], feat_pert[col].iloc[last], atol=1e-9), (
            f"TARGET LEAKAGE DETECTED: {col} at the last row changed when only "
            f"target_t was modified.\n"
            f"  base={feat_base[col].iloc[last]}, perturbed={feat_pert[col].iloc[last]}"
        )


def test_momentum_uses_correct_known_values():
    """
    Verify momentum_{p} = rate_(t-1) - rate_(t-1-p) using a deterministic series.

    With rates = [1, 2, 3, …, 50]:
        momentum_7  at index 8 = rate[7] - rate[0] = 8.0 - 1.0 = 7.0
        momentum_7  at index 9 = rate[8] - rate[1] = 9.0 - 2.0 = 7.0
        momentum_30 at index 31 = rate[30] - rate[0] = 31.0 - 1.0 = 30.0
    """
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rates = np.arange(1.0, n + 1, dtype=float)
    df = pd.DataFrame({"date": dates, "freight_rate": rates})

    feat = add_momentum(df, target_col="freight_rate", periods=[7, 30])

    # momentum_7 at index 8: anchor = rate[7]=8, lag = rate[0]=1  => 7.0
    assert np.isclose(feat["momentum_7"].iloc[8], 7.0, atol=1e-9), (
        f"momentum_7 at i=8 expected 7.0, got {feat['momentum_7'].iloc[8]}"
    )
    # momentum_7 at index 9: anchor = rate[8]=9, lag = rate[1]=2  => 7.0
    assert np.isclose(feat["momentum_7"].iloc[9], 7.0, atol=1e-9), (
        f"momentum_7 at i=9 expected 7.0, got {feat['momentum_7'].iloc[9]}"
    )
    # momentum_30 at index 31: anchor = rate[30]=31, lag = rate[0]=1 => 30.0
    assert np.isclose(feat["momentum_30"].iloc[31], 30.0, atol=1e-9), (
        f"momentum_30 at i=31 expected 30.0, got {feat['momentum_30'].iloc[31]}"
    )


def test_momentum_responds_to_historical_change():
    """
    Verify that changing a historical value that genuinely affects the feature
    DOES change the computed momentum (ensuring the formula is not trivially constant).

    Changing rate_(t-2) changes anchor rate_(t-1) for all rows after t-2,
    so momentum at the last row must change.
    """
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rates = np.arange(1.0, n + 1, dtype=float)

    df_base = pd.DataFrame({"date": dates, "freight_rate": rates.copy()})
    df_hist = df_base.copy()
    # Perturb rate at index (n-2), which is rate_(t-1) for the last row.
    df_hist.loc[df_hist.index[n - 2], "freight_rate"] += 5.0

    feat_base = add_momentum(df_base, target_col="freight_rate", periods=[7])
    feat_hist = add_momentum(df_hist, target_col="freight_rate", periods=[7])

    last = n - 1
    assert not np.isclose(feat_base["momentum_7"].iloc[last], feat_hist["momentum_7"].iloc[last]), (
        "momentum_7 did NOT change after modifying a historical value that should affect it."
    )


def test_xgboost_prepare_features_no_momentum_leakage(sample_daily_timeseries):
    """
    End-to-end check: after XGBoostForecaster._prepare_features(), the momentum
    columns must equal add_momentum output on the non-NaN rows, confirming there
    is no separate leaky path.

    Note: _prepare_features calls ffill/bfill on early NaN rows (warm-up period),
    so we only compare rows where add_momentum produces a valid (non-NaN) value.
    """
    model = XGBoostForecaster(n_estimators=5, max_depth=2, use_quantiles=False)
    df_sorted = sample_daily_timeseries.sort_values("date").reset_index(drop=True)
    df_feat, _ = model._prepare_features(df_sorted)

    # Independently compute correct momentum (canonical, no infilling)
    expected = add_momentum(df_sorted, target_col="freight_rate", periods=[7, 30])

    for p in [7, 30]:
        col = f"momentum_{p}"
        if col not in df_feat.columns:
            continue
        # Only compare rows where the canonical result is non-NaN
        valid_mask = expected[col].notna()
        assert np.allclose(
            df_feat[col][valid_mask].values,
            expected[col][valid_mask].values,
            atol=1e-9
        ), (
            f"XGBoostForecaster._prepare_features() {col} does not match "
            f"canonical add_momentum output on valid rows."
        )



def test_feature_engineering_calendar(sample_daily_timeseries):
    df = add_calendar_features(sample_daily_timeseries, date_col="date")
    for col in ["month", "week", "day_of_week", "month_sin", "month_cos", "is_monsoon", "is_cyclone_season"]:
        assert col in df.columns
    assert (df["month"] >= 1).all() and (df["month"] <= 12).all()
    assert (df["day_of_week"] >= 0).all() and (df["day_of_week"] <= 6).all()


def test_freight_feature_builder_multi_domain(sample_daily_timeseries):
    feat_df, cols = build_forecast_features(sample_daily_timeseries)
    assert len(feat_df) == len(sample_daily_timeseries)
    assert len(cols) >= 15
    assert "lag_1" in cols
    assert "rolling_mean_7" in cols
    assert "momentum_7" in cols
    # Zero NaN values after ffill/bfill
    assert feat_df[cols].isna().sum().sum() == 0


# =============================================================================
# 2. Evaluation Metrics Tests
# =============================================================================

def test_evaluation_metrics_values():
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([11.0, 12.0, 13.0, 18.0])

    mae = calculate_mae(y_true, y_pred)
    assert mae == (1.0 + 0.0 + 1.0 + 2.0) / 4.0

    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt((1 + 0 + 1 + 4) / 4.0))

    mape = calculate_mape(y_true, y_pred)
    assert mape > 0.0

    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 200.0

    y_train = np.array([8.0, 9.0, 10.0])
    mase = calculate_mase(y_true, y_pred, y_train=y_train)
    assert mase > 0.0

    metrics = evaluate_forecast(y_true, y_pred, y_train=y_train)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE", "MASE"}


# =============================================================================
# 3. Time-Series Walk-Forward Cross-Validation
# =============================================================================

def test_walk_forward_cross_validation(sample_daily_timeseries):
    cv_res = walk_forward_cv(
        model_factory=lambda: MovingAverageForecaster(window_size=7),
        df=sample_daily_timeseries,
        target_col="freight_rate",
        date_col="date",
        horizon_days=7,
        initial_train_size=40,
        step_size=15,
        window_type="expanding"
    )

    assert len(cv_res.folds) >= 2
    assert "MAE" in cv_res.overall_metrics
    assert "RMSE" in cv_res.overall_metrics
    assert "sMAPE" in cv_res.overall_metrics

    # Strict temporal causality check
    for fold in cv_res.folds:
        assert fold.train_end < fold.test_start, "Data leakage detected: train_end must precede test_start"
        assert len(fold.predictions) == 7
        assert len(fold.actuals) == 7


# =============================================================================
# 4. Baselines Tests
# =============================================================================

def test_naive_baseline(sample_daily_timeseries):
    model = NaiveBaselineForecaster()
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.last_value is not None

    res = model.predict(horizon_days=5, context_df=sample_daily_timeseries, date_col="date")
    assert res.horizon_days == 5
    assert len(res.series) == 5
    assert res.confidence_available is True

    # Check uncertainty bounds
    for pt in res.series:
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


def test_seasonal_baseline(sample_daily_timeseries):
    model = SeasonalBaselineForecaster(seasonal_period=7)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.seasonal_pattern is not None
    assert len(model.seasonal_pattern) == 7

    res = model.predict(horizon_days=14, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 14
    # Replicates pattern across cycles
    assert res.series[0].predicted_rate == res.series[7].predicted_rate


# =============================================================================
# 5. SARIMA/ARIMA Forecaster Tests
# =============================================================================

def test_arima_forecaster_fit_and_predict(sample_daily_timeseries):
    model = ARIMAForecaster(order=(1, 1, 0), alpha=0.10)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.model_fit is not None

    res = model.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert res.horizon_days == 7
    assert res.confidence_available is True
    for pt in res.series:
        assert pt.lower_ci is not None
        assert pt.upper_ci is not None
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 6. XGBoost Quantile Forecaster Tests
# =============================================================================

def test_xgboost_quantile_forecaster(sample_daily_timeseries):
    model = XGBoostForecaster(n_estimators=30, max_depth=3, use_quantiles=True)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert model.is_fitted is True

    res = model.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 7
    for pt in res.series:
        assert pt.lower_ci is not None
        assert pt.upper_ci is not None
        # Monotonic quantile constraint: P10 <= P50 <= P90
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 7. Ensemble Forecaster Tests
# =============================================================================

def test_ensemble_forecaster(sample_daily_timeseries):
    ensemble = EnsembleForecaster(weighting_strategy="inverse_rmse")
    ensemble.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")
    assert ensemble.is_fitted is True
    assert len(ensemble.weights) >= 2
    # Sum of weights equals 1.0
    assert np.isclose(sum(ensemble.weights.values()), 1.0, atol=1e-3)

    res = ensemble.predict(horizon_days=7, context_df=sample_daily_timeseries, date_col="date")
    assert len(res.series) == 7
    for pt in res.series:
        assert pt.lower_ci <= pt.predicted_rate <= pt.upper_ci


# =============================================================================
# 8. High-Level Orchestrator & Canonical API Output
# =============================================================================

def test_freight_forecaster_canonical_output():
    ff = FreightForecaster()
    pred = ff.predict_freight(
        origin="AUS_NEW",
        destination="IND_GVM",
        vessel_class="Capesize",
        cargo_type="thermal_coal",
        horizon_days=7
    )

    required_keys = {
        "current_rate",
        "forecast_rate",
        "lower_bound",
        "upper_bound",
        "trend",
        "confidence",
        "model_used",
        "metrics"
    }
    assert required_keys.issubset(pred.keys()), f"Missing keys in {pred.keys()}"
    assert pred["current_rate"] > 0
    assert pred["forecast_rate"] > 0
    assert pred["lower_bound"] <= pred["forecast_rate"] <= pred["upper_bound"]
    assert pred["trend"] in ["rising", "falling", "stable"]
    assert 0.50 <= pred["confidence"] <= 1.0
    assert isinstance(pred["model_used"], str)
    assert isinstance(pred["metrics"], dict)


# =============================================================================
# 9. Model Persistence & Metadata Schema
# =============================================================================

def test_model_persistence_and_metadata_roundtrip(sample_daily_timeseries):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "aus_new_ind_gvm_capesize" / "xgboost"
        model_dir.mkdir(parents=True)

        model = XGBoostForecaster(n_estimators=20, max_depth=3)
        model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

        artifact_file = model_dir / "model.pkl"
        model.save(str(artifact_file))
        assert artifact_file.exists()

        metadata = {
            "training_date": datetime.now().isoformat(),
            "dataset_version": "v2.0",
            "route": "AUS_NEW->IND_GVM",
            "vessel_class": "Capesize",
            "cargo_type": "thermal_coal",
            "model_name": "xgboost",
            "model_version": "1.0.0",
            "features": model.feature_cols,
            "metrics": {"MAE": 0.32, "RMSE": 0.41, "MAPE": 2.1, "sMAPE": 2.0, "MASE": 1.4},
            "horizons": [3, 7, 14, 30]
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Reload
        loaded = XGBoostForecaster()
        loaded.load(str(artifact_file))
        assert loaded.is_fitted is True
        res = loaded.predict(horizon_days=3, context_df=sample_daily_timeseries, date_col="date")
        assert len(res.series) == 3


# =============================================================================
# 10. HTTP REST API Endpoint Integration Test
# =============================================================================

def test_fastapi_forecast_endpoint():
    client = TestClient(app)
    response = client.get(
        "/api/v1/forecast",
        params={
            "origin": "AUS_NEW",
            "destination": "IND_GVM",
            "vessel_class": "Capesize",
            "horizon_days": 7,
            "cargo_type": "thermal_coal"
        }
    )
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Assert exact required API fields
    for field_name in ["current_rate", "forecast_rate", "lower_bound", "upper_bound", "trend", "confidence", "model_used", "metrics"]:
        assert field_name in data, f"Missing required API field: {field_name}"

    assert data["current_rate"] > 0
    assert data["forecast_rate"] > 0
    assert data["lower_bound"] <= data["forecast_rate"] <= data["upper_bound"]
    assert data["trend"] in ["rising", "falling", "stable"]
    assert 0.50 <= data["confidence"] <= 1.0

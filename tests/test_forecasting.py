"""
Tests for Freight Forecasting Engine
"""
import pandas as pd
import numpy as np
import pytest
from datetime import timedelta, date

from src.models.baseline_forecaster import BaselineForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.model_evaluation import evaluate_forecast, compare_models

@pytest.fixture
def synthetic_time_series():
    """Generates a simple synthetic time series of 30 days for testing."""
    dates = pd.date_range(start='2026-01-01', periods=30)
    # Simple linear trend + some noise
    values = np.linspace(10, 20, 30) + np.random.normal(0, 0.5, 30)
    return pd.DataFrame({
        'date': dates,
        'rate': values
    })

def test_evaluate_forecast():
    y_true = pd.Series([10.0, 12.0, 14.0])
    y_pred = pd.Series([11.0, 12.0, 13.0])
    
    metrics = evaluate_forecast(y_true, y_pred)
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'MAPE' in metrics
    assert metrics['MAE'] == (1.0 + 0.0 + 1.0) / 3

def test_compare_models():
    results = [
        {"Model": "A", "MAE": 2.0},
        {"Model": "B", "MAE": 1.0},
        {"Model": "C", "MAE": 3.0}
    ]
    df = compare_models(results)
    assert df.iloc[0]['Model'] == 'B'
    assert df.iloc[1]['Model'] == 'A'

def test_baseline_forecaster(synthetic_time_series):
    model = BaselineForecaster(window_size=3)
    # Fit
    model.fit(synthetic_time_series, target_col='rate', date_col='date')
    assert model.last_value is not None
    
    # Predict
    res = model.predict(horizon_days=5, context_df=synthetic_time_series, date_col='date')
    assert res.horizon_days == 5
    assert len(res.series) == 5
    assert res.model_used == "Baseline_MA3"
    
    # Evaluate
    metrics = model.evaluate(synthetic_time_series.tail(5), 'rate', 'date')
    assert "MAE" in metrics

def test_arima_forecaster(synthetic_time_series):
    # ARIMA might throw warnings for small series but should work
    model = ARIMAForecaster(order=(1, 0, 0))
    model.fit(synthetic_time_series, target_col='rate', date_col='date')
    assert model.model_fit is not None
    
    res = model.predict(horizon_days=3, context_df=synthetic_time_series, target_col='rate', date_col='date')
    assert res.horizon_days == 3
    assert len(res.series) == 3
    assert res.confidence_available is True
    assert res.series[0].lower_ci is not None

def test_xgboost_forecaster(synthetic_time_series):
    model = XGBoostForecaster(lags=[1, 2])
    model.fit(synthetic_time_series, target_col='rate', date_col='date')
    assert model.is_fitted is True
    
    res = model.predict(horizon_days=4, context_df=synthetic_time_series, target_col='rate', date_col='date')
    assert res.horizon_days == 4
    assert len(res.series) == 4
    assert res.confidence_available is True

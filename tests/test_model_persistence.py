"""
Charter-AI — Problem 7 Model Persistence, Versioning & Reproducibility Tests.

Verifies:
1. Model artifact persistence and reproducible inference without retraining.
2. Missing artifact error handling (raises ModelNotFoundError, no silent retraining).
3. Separation of training and inference paths.
4. Model registry version management and active version selection.
5. Wrong-model identity mismatch validation (raises ModelMismatchError).
6. Artifact integrity & corruption handling (raises ModelIntegrityError).
7. Test artifact isolation using pytest tmp_path fixtures.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from src.models.base_forecaster import ForecastResult, ForecastPoint
from src.models.xgboost_forecaster import XGBoostForecaster
from src.models.freight_forecaster import FreightForecaster
from src.models.registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelMismatchError,
    ModelIntegrityError,
)
from src.models.training import TrainingPipeline, TrainingConfig


@pytest.fixture
def sample_daily_timeseries():
    """Generates 90 days of daily freight series."""
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


def test_model_save_and_load_reproducibility(sample_daily_timeseries, tmp_path):
    """PART I: Test saving, reloading, and verifying reproducible inference without retraining."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    # 1. Train model
    model_orig = XGBoostForecaster(n_estimators=10, max_depth=3)
    model_orig.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

    # 2. Save artifact
    artifact_path = registry.save_model(
        model=model_orig,
        model_name="xgboost",
        model_version="v1.0.0",
        metrics={"MAE": 0.25, "RMSE": 0.35},
        route="AUS_NEW->IND_GVM",
        vessel_class="Capesize",
    )
    assert os.path.exists(artifact_path)

    # 3. Predict with original model
    res_orig = model_orig.predict(horizon_days=5, context_df=sample_daily_timeseries, date_col="date")

    # 4. Clear in-memory reference and reload from disk
    del model_orig
    loaded_model = registry.load_model("xgboost", model_version="v1.0.0")

    # 5. Predict with loaded model and verify exact reproduction
    res_loaded = loaded_model.predict(horizon_days=5, context_df=sample_daily_timeseries, date_col="date")

    assert len(res_loaded.series) == len(res_orig.series)
    for p_orig, p_loaded in zip(res_orig.series, res_loaded.series):
        assert p_orig.date == p_loaded.date
        assert np.isclose(p_orig.predicted_rate, p_loaded.predicted_rate, atol=1e-5)
        assert np.isclose(p_orig.lower_ci, p_loaded.lower_ci, atol=1e-5)
        assert np.isclose(p_orig.upper_ci, p_loaded.upper_ci, atol=1e-5)


def test_missing_model_artifact_raises_error(tmp_path):
    """PART J: Verify loading non-existent model raises typed error without silent retraining."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    with pytest.raises(ModelNotFoundError):
        registry.load_model("non_existent_model")

    ff = FreightForecaster(models_dir=str(tmp_path))
    with pytest.raises(ModelNotFoundError):
        ff.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            model_type="xgboost",
            allow_on_the_fly=False,
        )


def test_inference_does_not_trigger_retraining(sample_daily_timeseries, tmp_path):
    """PART D: Verify inference path uses persisted model artifact without calling fit()."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    # Save model artifact under route key
    model_orig = XGBoostForecaster(n_estimators=10, max_depth=3)
    model_orig.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

    route_dir = tmp_path / "aus_new_ind_gvm_capesize" / "xgboost"
    route_dir.mkdir(parents=True)
    model_orig.save(str(route_dir / "model.pkl"))
    with open(route_dir / "metadata.json", "w") as f:
        json.dump({
            "model_name": "xgboost",
            "route": "AUS_NEW->IND_GVM",
            "vessel_class": "Capesize",
            "metrics": {"MAE": 0.25}
        }, f)

    ff = FreightForecaster(models_dir=str(tmp_path))
    
    # Mock XGBoostForecaster.fit to fail if called
    original_fit = XGBoostForecaster.fit
    def throwing_fit(self, *args, **kwargs):
        raise RuntimeError("FIT WAS CALLED DURING INFERENCE!")

    XGBoostForecaster.fit = throwing_fit
    try:
        res = ff.predict_freight(
            origin="AUS_NEW",
            destination="IND_GVM",
            vessel_class="Capesize",
            model_type="xgboost",
            allow_on_the_fly=False,
        )
        assert res["current_rate"] > 0
        assert res["forecast_rate"] > 0
    finally:
        XGBoostForecaster.fit = original_fit


def test_model_registry_versioning(sample_daily_timeseries, tmp_path):
    """PART F: Test registry version registration, listing, and active version loading."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    model_1 = XGBoostForecaster(n_estimators=5, max_depth=2)
    model_1.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

    model_2 = XGBoostForecaster(n_estimators=10, max_depth=3)
    model_2.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

    # Save v1.0.0
    registry.save_model(model_1, "xgboost", "v1.0.0", metrics={"MAE": 0.50}, is_active=False)
    # Save v2.0.0
    registry.save_model(model_2, "xgboost", "v2.0.0", metrics={"MAE": 0.30}, is_active=True)

    versions = registry.list_versions("xgboost")
    assert versions == ["v1.0.0", "v2.0.0"]
    assert registry.get_active_version("xgboost") == "v2.0.0"

    # Loading without version defaults to active v2.0.0
    loaded_active = registry.load_model("xgboost")
    assert loaded_active.model_p50.max_depth == 3

    # Explicit version loading
    loaded_v1 = registry.load_model("xgboost", model_version="v1.0.0")
    assert loaded_v1.model_p50.max_depth == 2


def test_wrong_model_mismatch_validation(sample_daily_timeseries, tmp_path):
    """PART G: Test metadata validation preventing accidental use of wrong route/vessel_class model."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    model = XGBoostForecaster(n_estimators=5)
    model.fit(sample_daily_timeseries, target_col="freight_rate", date_col="date")

    registry.save_model(
        model=model,
        model_name="xgboost",
        model_version="v1.0.0",
        metrics={"MAE": 0.30},
        route="AUS_NEW->IND_GVM",
        vessel_class="Capesize",
    )

    # Correct parameters -> pass
    loaded = registry.load_model("xgboost", route="AUS_NEW->IND_GVM", vessel_class="Capesize")
    assert loaded is not None

    # Wrong vessel class -> raise ModelMismatchError
    with pytest.raises(ModelMismatchError):
        registry.load_model("xgboost", route="AUS_NEW->IND_GVM", vessel_class="Panamax")

    # Wrong route -> raise ModelMismatchError
    with pytest.raises(ModelMismatchError):
        registry.load_model("xgboost", route="IDN_TAB->IND_PAR", vessel_class="Capesize")


def test_corrupt_artifact_handling(tmp_path):
    """PART H: Test empty or corrupted artifact file handling."""
    registry = ModelRegistry(base_dir=str(tmp_path))

    corrupt_dir = tmp_path / "corrupt_model" / "v1.0.0"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "model.pkl").write_bytes(b"")  # 0-byte file

    with pytest.raises(ModelIntegrityError):
        registry.load_model("corrupt_model", model_version="v1.0.0")


def test_training_pipeline_execution(sample_daily_timeseries, tmp_path):
    """PART L: Test end-to-end TrainingPipeline execution and registry registration."""
    registry = ModelRegistry(base_dir=str(tmp_path))
    config = TrainingConfig(
        model_type="xgboost",
        model_version="v1.0.0",
        route="AUS_NEW->IND_GVM",
        vessel_class="Capesize",
        test_size_days=15,
        hyperparams={"n_estimators": 10, "max_depth": 3},
    )

    pipeline = TrainingPipeline(config=config, registry=registry)
    result = pipeline.run(sample_daily_timeseries)

    assert result.model_type == "xgboost"
    assert result.model_version == "v1.0.0"
    assert "MAE" in result.metrics
    assert os.path.exists(result.artifact_path)

    # Verify model is loadable via registry
    reloaded = registry.load_model("xgboost", model_version="v1.0.0", route="AUS_NEW->IND_GVM", vessel_class="Capesize")
    assert reloaded.is_fitted is True


def test_test_artifact_isolation(tmp_path):
    """PART O: Verify test model artifacts are written strictly to tmp_path."""
    registry = ModelRegistry(base_dir=str(tmp_path))
    assert str(registry.base_dir).startswith(str(tmp_path))

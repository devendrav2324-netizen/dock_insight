"""
DockInsights — Problem 8 Data Provenance & Quality Contract Tests.

Tests:
1. Synthetic datasets are labeled SYNTHETIC_DEMO, not VERIFIED_EXTERNAL.
2. Dataset provenance metadata survives loading.
3. Training metadata inherits data mode.
4. Persisted model metadata retains data provenance.
5. Reloaded model metadata retains provenance.
6. Forecast result retains provenance.
7. Date coverage is correctly reported.
8. Unknown/unverified source cannot silently become verified.
9. Full provenance end-to-end regression test.
10. Data quality contract validation tests.
"""

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from src.data.provenance import (
    DATASET_REGISTRY,
    DataMode,
    DataProvenance,
    QualityStatus,
    SourceType,
    get_dataset_provenance,
    list_registered_datasets,
)
from src.data.validate_data import (
    check_date_coverage,
    check_duplicates,
    check_missing_values,
    check_numeric_finiteness,
    check_required_columns,
    generate_quality_report,
    validate_dataset_provenance,
    validate_dates,
)
from src.data.validators.business_validator import BusinessRuleValidator
from src.models.freight_forecaster import FreightForecaster
from src.models.registry import ModelRegistry
from src.models.training import TrainingConfig, TrainingPipeline


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


def test_synthetic_dataset_labeled_synthetic_demo(repo_root):
    """Verify demo datasets are explicitly labeled SYNTHETIC_DEMO in provenance registry and CSV files."""
    demo_dir = repo_root / "data" / "demo"
    assert demo_dir.exists(), "data/demo directory must exist"

    for csv_path in demo_dir.glob("*.csv"):
        dataset_name = csv_path.stem
        prov = get_dataset_provenance(dataset_name)
        assert prov.data_mode == DataMode.SYNTHETIC_DEMO
        assert prov.is_verified_external is False

        df = pd.read_csv(csv_path, comment="#")
        source_col = next((c for c in df.columns if c in ("source", "data_source")), None)
        if source_col:
            sources = df[source_col].dropna().unique()
            for s in sources:
                assert s == "SYNTHETIC_DEMO", f"File {csv_path.name} contains non-synthetic source label: {s}"


def test_synthetic_dataset_not_labeled_verified_external():
    """Verify no synthetic demo dataset claims to be VERIFIED_EXTERNAL."""
    registered = list_registered_datasets()
    for reg in registered:
        assert reg["data_mode"] == "SYNTHETIC_DEMO"
        assert reg["is_verified_external"] is False


def test_dataset_provenance_metadata_survives_loading():
    """Verify dataset provenance metadata is retrievable and complete."""
    prov = get_dataset_provenance("freight_rates")
    assert prov.dataset_name == "freight_rates"
    assert prov.data_mode == DataMode.SYNTHETIC_DEMO
    assert prov.source_type == SourceType.GENERATED
    assert prov.date_start == "2019-01-01"
    assert prov.date_end == "2024-12-31"


def test_training_metadata_inherits_data_mode(tmp_path):
    """Verify model training inherits data_mode from configuration/dataset."""
    config = TrainingConfig(
        model_type="moving_average",
        route="AUS_NEW->IND_GVM",
        vessel_class="Capesize",
        data_mode="SYNTHETIC_DEMO",
        provenance_status="SYNTHETIC_DEMO",
    )
    registry = ModelRegistry(base_dir=tmp_path / "models")
    pipeline = TrainingPipeline(config=config, registry=registry)

    # Synthetic sample data
    dates = pd.date_range("2024-01-01", periods=60)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Capesize",
        "freight_rate": [15.0 + i * 0.1 for i in range(60)],
        "source": "SYNTHETIC_DEMO",
    })

    result = pipeline.run(df)
    assert result.data_mode == "SYNTHETIC_DEMO"
    assert result.provenance_status == "SYNTHETIC_DEMO"


def test_persisted_model_metadata_retains_provenance(tmp_path):
    """Verify persisted model metadata.json explicitly includes data_mode."""
    registry = ModelRegistry(base_dir=tmp_path / "models")
    fake_model = {"type": "mock"}

    artifact_path = registry.save_model(
        model=fake_model,
        model_name="xgboost",
        model_version="v1.0.0",
        metrics={"MAE": 0.5},
        data_mode="SYNTHETIC_DEMO",
        provenance_status="SYNTHETIC_DEMO",
    )

    metadata = registry.get_metadata("xgboost", "v1.0.0")
    assert metadata["data_mode"] == "SYNTHETIC_DEMO"
    assert metadata["provenance_status"] == "SYNTHETIC_DEMO"
    assert metadata["is_verified_external"] is False


def test_reloaded_model_metadata_retains_provenance(tmp_path):
    """Verify reloading model from disk preserves data_mode in metadata."""
    registry = ModelRegistry(base_dir=tmp_path / "models")
    fake_model = {"type": "mock"}

    registry.save_model(
        model=fake_model,
        model_name="arima",
        model_version="v1.0.0",
        metrics={"RMSE": 1.2},
        data_mode="SYNTHETIC_DEMO",
        provenance_status="SYNTHETIC_DEMO",
    )

    metadata = registry.get_metadata("arima", "v1.0.0")
    assert metadata["data_mode"] == "SYNTHETIC_DEMO"


def test_forecast_result_retains_provenance(repo_root):
    """Verify FreightForecaster.predict_freight output dict includes data_mode."""
    forecaster = FreightForecaster(
        data_dir=str(repo_root / "data" / "demo"),
        models_dir=str(repo_root / "models"),
    )
    result = forecaster.predict_freight(
        origin="AUS_NEW",
        destination="IND_GVM",
        vessel_class="Capesize",
        horizon_days=7,
        allow_on_the_fly=True,
    )
    assert "data_mode" in result
    assert result["data_mode"] == "SYNTHETIC_DEMO"
    assert result["provenance_status"] == "SYNTHETIC_DEMO"
    assert result["is_verified_external"] is False


def test_date_coverage_correctly_reported(repo_root):
    """Verify date coverage reported for time-series datasets."""
    csv_path = repo_root / "data" / "demo" / "freight_rates.csv"
    df = pd.read_csv(csv_path, comment="#")
    coverage = check_date_coverage(df)

    assert coverage["has_dates"] is True
    assert coverage["date_start"] == "2019-01-01"
    assert coverage["date_end"] == "2024-12-31"
    assert coverage["total_days_spanned"] == 2192


def test_unknown_unverified_source_cannot_silently_become_verified():
    """Verify unregistered datasets default to UNKNOWN mode with is_verified_external=False."""
    prov = get_dataset_provenance("unknown_dataset_xyz")
    assert prov.data_mode == DataMode.UNKNOWN
    assert prov.source_type == SourceType.UNVERIFIED
    assert prov.is_verified_external is False


def test_provenance_regression_pipeline_end_to_end(tmp_path, repo_root):
    """
    End-to-end regression test:
    dataset load -> feature preparation -> training -> persistence -> reload -> inference
    Verifies data_mode == SYNTHETIC_DEMO throughout the entire pipeline.
    """
    # 1. Dataset load
    csv_path = repo_root / "data" / "demo" / "freight_rates.csv"
    df = pd.read_csv(csv_path, comment="#")
    filtered = df[(df["origin"] == "AUS_NEW") & (df["destination"] == "IND_GVM") & (df["vessel_class"] == "Capesize")].copy()
    assert len(filtered) > 0

    # 2. Training
    config = TrainingConfig(
        model_type="moving_average",
        route="AUS_NEW->IND_GVM",
        vessel_class="Capesize",
        data_mode="SYNTHETIC_DEMO",
        provenance_status="SYNTHETIC_DEMO",
    )
    registry = ModelRegistry(base_dir=tmp_path / "models")
    pipeline = TrainingPipeline(config=config, registry=registry)
    train_res = pipeline.run(filtered)
    assert train_res.data_mode == "SYNTHETIC_DEMO"

    # 3. Model metadata check after persistence
    metadata = registry.get_metadata("moving_average", config.model_version)
    assert metadata["data_mode"] == "SYNTHETIC_DEMO"

    # 4. Inference via forecaster
    forecaster = FreightForecaster(
        data_dir=str(repo_root / "data" / "demo"),
        models_dir=str(tmp_path / "models"),
    )
    result = forecaster.predict_freight(
        origin="AUS_NEW",
        destination="IND_GVM",
        vessel_class="Capesize",
        model_type="moving_average",
        allow_on_the_fly=False,
    )
    assert result["data_mode"] == "SYNTHETIC_DEMO"
    assert result["provenance_status"] == "SYNTHETIC_DEMO"


def test_data_quality_contract_validations():
    """Test data quality contract rules on clean and malformed dataframes."""
    # Clean df
    df_clean = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "freight_rate": [15.5, 16.0],
        "source": ["SYNTHETIC_DEMO", "SYNTHETIC_DEMO"]
    })
    report = generate_quality_report(df_clean, "freight_rates")
    assert report["missing_values_count"] == 0
    assert report["duplicate_rows"] == 0
    assert report["quality_status"] in ("VALID", "WARNING")

    # Missing column check
    assert check_required_columns(df_clean, ["date", "freight_rate"]) is True
    assert check_required_columns(df_clean, ["nonexistent_col"]) is False

    # Invalid dates check
    df_bad_dates = pd.DataFrame({"date": ["invalid-date-string", "2024-01-02"]})
    assert validate_dates(df_bad_dates, ["date"]) is False

    # Non-finite numeric check
    df_inf = pd.DataFrame({"value": [10.0, float("inf"), float("-inf")]})
    finiteness_issues = check_numeric_finiteness(df_inf)
    assert len(finiteness_issues) == 1

    # Deceptive external claim check
    df_deceptive = pd.DataFrame({
        "date": ["2024-01-01"],
        "source": ["BALTIC_EXCHANGE_HISTORICAL_REAL"]
    })
    prov_res = validate_dataset_provenance(df_deceptive, "freight_rates")
    assert prov_res["misleading_external_claims"] is True
    assert prov_res["provenance_valid"] is False

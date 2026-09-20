"""
Tests for DockInsights V2 Data Layer (Phase 2).

Validates loaders, schema validators, business validators, preprocessors,
and normalized data models.
"""

from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from src.data.loaders.csv_loader import CSVLoader
from src.data.validators.schema_validator import SchemaValidator
from src.data.validators.business_validator import BusinessRuleValidator
from src.data.preprocessors.cleaner import DataCleaner, clean_strings, handle_missing_values
from src.data.preprocessors.normalizer import DataNormalizer
from src.data.preprocessors.deduplicator import Deduplicator
from src.data.models import Port, Vessel, BunkerPrice, DryBulkIndex, PortCongestion


@pytest.fixture
def sample_csv_with_comments(tmp_path):
    f = tmp_path / "test_ports.csv"
    with open(f, "w") as fp:
        fp.write("port_id, port_name , country , latitude , longitude , max_draft_m , max_loa_m , max_beam_m \n")
        fp.write("# SYNTHETIC_DEMO: comment to skip\n")
        fp.write("IND_VZG, Visakhapatnam , IND , 17.6868 , 83.2185 , 18.1 , 280.0 , 45.0 \n")
    return str(f)


def test_csv_loader(sample_csv_with_comments):
    loader = CSVLoader()
    df = loader.load(sample_csv_with_comments)
    assert len(df) == 1
    assert "port_id" in df.columns
    assert df["port_id"].iloc[0] == "IND_VZG"
    assert df["port_name"].iloc[0] == "Visakhapatnam"
    assert df["country"].iloc[0] == "IND"


def test_schema_validator_success():
    validator = SchemaValidator()
    df = pd.DataFrame([{
        "port_id": "IND_VZG",
        "port_name": "Visakhapatnam",
        "country": "IND",
        "latitude": 17.68,
        "longitude": 83.21,
        "max_draft_m": 18.1,
        "max_loa_m": 280.0,
        "max_beam_m": 45.0,
    }])
    result = validator.validate_dataframe(df, "ports")
    assert result.is_valid is True
    assert result.error_count == 0


def test_schema_validator_failure():
    validator = SchemaValidator()
    df = pd.DataFrame([{
        "port_id": "IND_VZG",
        "port_name": "Visakhapatnam",
        "country": "IND",
        "latitude": 120.0,  # Invalid latitude (>90)
        "longitude": 83.21,
        "max_draft_m": -5.0,  # Negative draft
        "max_loa_m": 280.0,
        "max_beam_m": 45.0,
    }])
    result = validator.validate_dataframe(df, "ports")
    assert result.is_valid is False
    assert result.error_count >= 1


def test_business_validator_missing_fields():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"port_id": "IND_VZG"}])  # Missing required fields
    issues = validator.check_missing_fields(df, "ports")
    assert len(issues) > 0
    assert any("port_name" in iss for iss in issues)


def test_business_validator_negative_values():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"freight_rate": -15.0, "distance_nm": 5000.0}])
    issues = validator.check_negative_values(df, "freight_rates")
    assert len(issues) == 1
    assert "freight_rate" in issues[0]


def test_business_validator_impossible_dimensions():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"draft": 45.0, "loa": 600.0, "beam": 120.0, "dwt": 1000}])
    issues = validator.check_impossible_vessel_dimensions(df)
    assert len(issues) >= 3


def test_business_validator_invalid_coordinates():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"latitude": 95.0, "longitude": -190.0}])
    issues = validator.check_invalid_coordinates(df)
    assert len(issues) == 2


def test_business_validator_duplicates():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([
        {"port_id": "IND_VZG", "port_name": "Vizag"},
        {"port_id": "IND_VZG", "port_name": "Vizag"},
    ])
    issues = validator.check_duplicate_records(df, "ports")
    assert len(issues) == 1
    assert "Duplicate" in issues[0]


def test_business_validator_future_dates():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"date": "2050-01-01"}])
    issues = validator.check_future_historical_dates(df)
    assert len(issues) == 1
    assert "Future dates" in issues[0]


def test_business_validator_inconsistent_vessel_classes():
    validator = BusinessRuleValidator()
    df = pd.DataFrame([{"vessel_class": "Capesize", "dwt": 25000}])  # Way too small for Capesize
    issues = validator.check_inconsistent_vessel_classes(df)
    assert len(issues) == 1
    assert "Inconsistent vessel class" in issues[0]


def test_data_normalizer():
    normalizer = DataNormalizer()
    df = pd.DataFrame({
        "origin": ["VIZAG", "PARADIP", "AUS_NEW"],
        "date": ["2026-01-01 12:00:00", "2026-01-02 00:00:00", "2026-01-03"]
    })
    df_norm = normalizer.normalize_port_codes(df)
    assert df_norm["origin"].iloc[0] == "IND_VZG"
    assert df_norm["origin"].iloc[1] == "IND_PAR"

    df_norm = normalizer.normalize_timestamps(df_norm)
    assert df_norm["date"].iloc[0] == date(2026, 1, 1)


def test_deduplicator():
    dedup = Deduplicator()
    df = pd.DataFrame({
        "port_id": ["IND_VZG", "IND_VZG", "IND_PAR"],
        "name": ["A", "B", "C"]
    })
    df_dedup = dedup.deduplicate(df, subset=["port_id"])
    assert len(df_dedup) == 2


def test_models_instantiation():
    port = Port(
        port_id="IND_VZG",
        port_name="Visakhapatnam",
        country="IND",
        latitude=17.68,
        longitude=83.21,
        max_draft_m=18.1,
        max_loa_m=280.0,
        max_beam_m=45.0,
        source="SYNTHETIC_DEMO",
        confidence_level="LOW",
    )
    assert port.port_id == "IND_VZG"
    assert port.source == "SYNTHETIC_DEMO"
    assert port.confidence_level == "LOW"

    vessel = Vessel(
        vessel_id="VSL_001",
        imo_number="9451001",
        vessel_name="Test Bulker",
        vessel_class="Panamax",
        dwt=75000,
        loa_m=225.0,
        beam_m=32.2,
        max_draft_m=14.2,
        service_speed_knots=13.5,
        ballast_speed_knots=14.0,
        laden_speed_knots=13.0,
        fuel_consumption_mt_day=32.0,
        age_years=5,
    )
    assert vessel.vessel_id == "VSL_001"
    assert vessel.dwt == 75000

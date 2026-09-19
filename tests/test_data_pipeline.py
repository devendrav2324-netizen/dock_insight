"""
Tests for the data pipeline modules (load, validate, preprocess).
"""
import os
import pandas as pd
import pytest

from src.data.load_data import load_csv
from src.data.validate_data import (
    check_required_columns,
    check_missing_values,
    check_duplicates,
    validate_dates,
    generate_quality_report
)
from src.data.preprocess import (
    convert_types,
    clean_strings,
    handle_missing_values
)

@pytest.fixture
def sample_csv_path(tmp_path):
    """Creates a temporary CSV file with synthetic data."""
    filepath = tmp_path / "test_freight.csv"
    with open(filepath, "w") as f:
        f.write("date,origin,destination,rate\n")
        f.write("# SYNTHETIC DATA\n")
        f.write("2026-01-01, AUS ,IND,15.5\n")
        f.write("2026-01-02,USA,IND,\n") # Missing rate
        f.write("2026-01-02,USA,IND,\n") # Duplicate row
        f.write("invalid-date,AUS,IND,20.0\n")
    return str(filepath)

def test_load_csv(sample_csv_path):
    df = load_csv(sample_csv_path)
    # The comment row should be skipped
    assert len(df) == 4
    assert list(df.columns) == ["date", "origin", "destination", "rate"]

def test_check_required_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert check_required_columns(df, ["A", "B"]) is True
    assert check_required_columns(df, ["A", "C"]) is False

def test_check_missing_values():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    warnings = check_missing_values(df, threshold_pct=0.1)
    assert "A" in warnings
    assert "B" not in warnings

def test_check_duplicates():
    df = pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]})
    assert check_duplicates(df) == 1

def test_validate_dates():
    df = pd.DataFrame({"valid": ["2026-01-01", "2026-02-01"], "invalid": ["2026-01-01", "not-a-date"]})
    assert validate_dates(df, ["valid"]) is True
    assert validate_dates(df, ["invalid"]) is False

def test_generate_quality_report():
    df = pd.DataFrame({"A": [1, 2, 2], "B": [None, 4, 4]})
    report = generate_quality_report(df, "test")
    assert report["dataset"] == "test"
    assert report["rows"] == 3
    assert report["duplicate_rows"] == 1
    assert report["missing_values_count"] == 1

def test_convert_types():
    df = pd.DataFrame({"A": ["1", "2"], "date": ["2026-01-01", "2026-01-02"]})
    df_converted = convert_types(df, {"A": "int64", "date": "datetime"})
    assert df_converted["A"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(df_converted["date"])

def test_clean_strings():
    df = pd.DataFrame({"A": ["  str1  ", "str2\t"]})
    df_clean = clean_strings(df, ["A"])
    assert df_clean["A"].iloc[0] == "str1"
    assert df_clean["A"].iloc[1] == "str2"

def test_handle_missing_values():
    df = pd.DataFrame({"A": [1, None, 3]})
    df_dropped = handle_missing_values(df, strategy="drop")
    assert len(df_dropped) == 2
    
    df_filled = handle_missing_values(df, strategy="fill", fill_values={"A": 0})
    assert len(df_filled) == 3
    assert df_filled["A"].iloc[1] == 0

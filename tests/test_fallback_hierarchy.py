"""
tests/test_fallback_hierarchy.py

Task 3 - Safe Historical Data Fallback Hierarchy Tests.
"""

import pandas as pd
import pytest
from pathlib import Path

from src.models.freight_forecaster import FreightForecaster


@pytest.fixture
def mock_freight_data(tmp_path):
    # Create a mock freight_rates.csv with specific combos
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    
    # EXACT: 200 obs
    df_exact = pd.DataFrame({
        "date": dates,
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Capesize",
        "cargo_type": "thermal_coal",
        "freight_rate": 15.0
    })
    
    # EXACT but insufficient: 50 obs
    df_exact_short = pd.DataFrame({
        "date": dates[:50],
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Panamax",
        "cargo_type": "thermal_coal",
        "freight_rate": 14.0
    })
    
    # ROUTE_VESSEL fallback for the short exact one: 150 obs (different cargo)
    df_rv = pd.DataFrame({
        "date": dates[50:200],
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "vessel_class": "Panamax",
        "cargo_type": "other_cargo",
        "freight_rate": 13.0
    })
    
    # ROUTE fallback: origin + dest, different vessel (Supramax)
    df_route = pd.DataFrame({
        "date": dates,
        "origin": "AUS_NEW",
        "destination": "IND_PAR", # different dest to test another fallback
        "vessel_class": "Supramax",
        "cargo_type": "other_cargo",
        "freight_rate": 12.0
    })
    
    # UNRELATED ROUTE
    df_unrelated = pd.DataFrame({
        "date": dates,
        "origin": "IDN_TAB",
        "destination": "IND_HLD",
        "vessel_class": "Handysize",
        "cargo_type": "thermal_coal",
        "freight_rate": 10.0
    })

    df = pd.concat([df_exact, df_exact_short, df_rv, df_route, df_unrelated])
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "freight_rates.csv", index=False)
    
    return data_dir


def test_fallback_level_1_exact(mock_freight_data):
    """Test 1: Exact route/vessel/cargo data is selected when sufficient."""
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    data_scope = ff.load_historical_data("AUS_NEW", "IND_GVM", "Capesize", "thermal_coal")
    
    assert data_scope.level == "EXACT"
    assert data_scope.n_obs == 200
    assert data_scope.data_quality == "HIGH"
    
    # Test 5 & 6 & 7 (partially): Unrelated routes, cargos, vessels are not silently included
    assert (data_scope.df["origin"] == "AUS_NEW").all()
    assert (data_scope.df["destination"] == "IND_GVM").all()
    assert (data_scope.df["vessel_class"] == "Capesize").all()
    assert (data_scope.df["cargo_type"] == "thermal_coal").all()


def test_fallback_level_2_route_vessel(mock_freight_data):
    """Test 2: If exact data is insufficient, route+vessel data is selected."""
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    # Panamax + thermal_coal only has 50 rows (< 180).
    # But Panamax overall has 50 (thermal) + 150 (other) = 200 rows.
    data_scope = ff.load_historical_data("AUS_NEW", "IND_GVM", "Panamax", "thermal_coal")
    
    assert data_scope.level == "ROUTE_VESSEL"
    assert data_scope.n_obs == 200
    assert data_scope.data_quality == "MEDIUM"
    
    # Test 6: Unrelated cargo types are included only when explicitly permitted by fallback
    cargo_types = data_scope.df["cargo_type"].unique()
    assert "other_cargo" in cargo_types


def test_fallback_level_3_route(mock_freight_data):
    """Test 3: If route+vessel is insufficient, route-level data is selected."""
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    # Request Capesize for AUS_NEW -> IND_PAR.
    # No Capesize exists, but Supramax exists (200 rows).
    data_scope = ff.load_historical_data("AUS_NEW", "IND_PAR", "Capesize", "thermal_coal")
    
    assert data_scope.level == "ROUTE"
    assert data_scope.n_obs == 200
    assert data_scope.data_quality == "LOW"
    
    # Test 7: Unrelated vessel classes included by fallback
    assert (data_scope.df["vessel_class"] == "Supramax").all()


def test_fallback_insufficient_data(mock_freight_data):
    """Test 4: If no acceptable historical data exists, returns explicit insufficient-data result."""
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    # Request completely unrelated route not in dataset (e.g. ZAF_RIC -> IND_VZG)
    data_scope = ff.load_historical_data("ZAF_RIC", "IND_VZG", "Capesize", "thermal_coal")
    
    assert data_scope.level == "INSUFFICIENT"
    assert data_scope.n_obs == 0
    assert data_scope.data_quality == "INSUFFICIENT"
    assert data_scope.df.empty


def test_unrelated_routes_never_silently_included(mock_freight_data):
    """Test 5: Verify that unrelated routes are NEVER silently included."""
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    # IDN_TAB -> IND_HLD exists in data
    data_scope = ff.load_historical_data("IDN_TAB", "IND_HLD", "Handysize", "thermal_coal")
    
    assert data_scope.level == "EXACT"
    # Ensure no AUS_NEW data snuck in
    assert not (data_scope.df["origin"] == "AUS_NEW").any()


def test_predict_freight_api_response_fields(mock_freight_data):
    """
    Test 8, 9, 10: Verify fallback level, observation count, and data quality
    are exposed correctly via the API output fields.
    """
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    res = ff.predict_freight(
        origin="AUS_NEW",
        destination="IND_GVM",
        vessel_class="Panamax",
        cargo_type="thermal_coal",
        model_type="naive"  # Use naive for speed
    )
    
    assert res["data_scope"] == "ROUTE_VESSEL"
    assert res["fallback_level"] == "ROUTE_VESSEL"
    assert res["training_observations"] == 200
    assert res["data_quality"] == "MEDIUM"


def test_regression_empty_route_does_not_train_on_all(mock_freight_data):
    """
    Test 11: Add a regression test specifically proving that an empty
    requested route does NOT cause the system to train on all historical routes.
    """
    ff = FreightForecaster(data_dir=str(mock_freight_data))
    
    with pytest.raises(ValueError, match="Insufficient historical data"):
        ff.predict_freight(
            origin="",
            destination="",
            vessel_class="",
            cargo_type=""
        )


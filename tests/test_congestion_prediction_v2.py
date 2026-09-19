"""
Charter-AI — Comprehensive Test Suite for Phase 4 Port Congestion & Idle-Time Prediction.

Tests:
1. CongestionFeatureBuilder (historical rolling stats, calendar, weather, port specs)
2. Queuing theory and moving average baselines
3. XGBoost quantile regression (P10 <= P50 <= P90, delay probability, congestion classification)
4. IdleTimePredictor (port-level and voyage-level idle days & demurrage risk)
5. CongestionService integration
6. VoyageEconomicsService dynamic impact (waiting time increases duration, demurrage, and $/tonne)
7. DecisionEngine dynamic congestion integration (eliminating hard-coded 2.0 days)
8. FastAPI HTTP REST API endpoint (/api/v1/congestion/predict)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from fastapi.testclient import TestClient

from src.models.congestion_predictor import (
    CongestionPredictor,
    CongestionPrediction,
    CongestionFeatureBuilder,
    QueuingTheoryBaseline,
    PortMovingAverageBaseline,
    classify_congestion_level
)
from src.models.idle_time_predictor import IdleTimePredictor, VoyageIdleTimeResult
from src.services.congestion_service import CongestionService
from src.services.voyage_economics_service import VoyageEconomicsService
from src.services.risk_service import RiskService
from src.services.freight_forecast_service import FreightForecastService
from src.services.vessel_optimization_service import VesselOptimizationService
from src.services.contract_optimization_service import ContractOptimizationService
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.data.mock_db import get_mock_port_info, get_mock_vessel_db
from src.api.main import app


@pytest.fixture
def synthetic_congestion_df():
    """Generates 60 days of synthetic congestion observations for testing."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rows = []
    for d in dates:
        for p in ["IND_PAR", "IND_GVM", "AUS_NEW"]:
            base = 4.0 if p == "IND_PAR" else (2.0 if p == "IND_GVM" else 3.5)
            w_days = float(np.clip(base + np.sin(d.day / 5.0) + np.random.normal(0, 0.2), 0.5, 8.0))
            v_wait = max(1, int(round(w_days * 2.0)))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "port": p,
                "vessels_waiting": v_wait,
                "average_waiting_days": w_days,
                "berth_occupancy_pct": 80.0,
                "congestion_index": round(w_days * 0.6, 2),
                "source": "SYNTHETIC_TEST"
            })
    return pd.DataFrame(rows)


# =============================================================================
# 1. Feature Engineering Tests
# =============================================================================

def test_congestion_feature_builder(synthetic_congestion_df):
    fb = CongestionFeatureBuilder()
    df_feat = fb.build_features(synthetic_congestion_df)

    assert len(df_feat) == len(synthetic_congestion_df)
    for expected_col in [
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "month", "weekday", "is_weekend", "is_monsoon", "is_cyclone_season",
        "wind_speed_kmh", "wave_height_m", "storm_indicator",
        "berth_count", "cargo_handling_rate_mt_day", "max_draft_m"
    ]:
        assert expected_col in df_feat.columns, f"Missing feature: {expected_col}"

    assert df_feat["rolling_mean_7"].isna().sum() == 0


# =============================================================================
# 2. Baseline Model Tests
# =============================================================================

def test_queuing_theory_baseline():
    queuing = QueuingTheoryBaseline()
    wait_low = queuing.predict(berth_count=10, berth_occupancy_pct=60.0, vessels_waiting=2)
    wait_high = queuing.predict(berth_count=4, berth_occupancy_pct=95.0, vessels_waiting=12)

    assert wait_low < wait_high
    assert 0.5 <= wait_low <= 12.0
    assert 0.5 <= wait_high <= 12.0


def test_port_moving_average_baseline(synthetic_congestion_df):
    ma = PortMovingAverageBaseline(window_size=7)
    ma.fit(synthetic_congestion_df)

    wait_par = ma.predict("IND_PAR")
    wait_gvm = ma.predict("IND_GVM")
    assert wait_par > wait_gvm  # Paradip historically more congested than Gangavaram
    assert 1.0 <= wait_par <= 6.0


def test_classify_congestion_level():
    assert classify_congestion_level(1.0) == "LOW"
    assert classify_congestion_level(2.2) == "MODERATE"
    assert classify_congestion_level(4.1) == "HIGH"
    assert classify_congestion_level(6.0) == "SEVERE"


# =============================================================================
# 3. XGBoost Quantile Regressor Tests
# =============================================================================

def test_xgboost_congestion_predictor(synthetic_congestion_df):
    pred = CongestionPredictor()
    pred.fit(synthetic_congestion_df)
    assert pred.is_fitted is True

    res = pred.predict_congestion(
        port_id="IND_PAR",
        target_date="2025-04-15",
        vessel_class="Capesize",
        cargo_quantity=150000
    )

    assert isinstance(res, CongestionPrediction)
    assert res.expected_wait_days > 0
    # Monotonic quantile non-crossing guarantee: P10 <= P50 <= P90
    assert res.p10_wait_days <= res.p50_wait_days <= res.p90_wait_days
    assert 0.0 <= res.delay_probability <= 1.0
    assert res.congestion_level in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert 0.50 <= res.confidence <= 1.0


# =============================================================================
# 4. Idle Time Predictor Tests
# =============================================================================

def test_idle_time_predictor_voyage(synthetic_congestion_df):
    idle = IdleTimePredictor()
    idle.fit(synthetic_congestion_df)

    voyage_res = idle.predict_voyage_idle(
        origin_port="AUS_NEW",
        destination_port="IND_PAR",
        vessel_class="Capesize",
        cargo_quantity=150000
    )

    assert isinstance(voyage_res, VoyageIdleTimeResult)
    assert voyage_res.origin_port == "AUS_NEW"
    assert voyage_res.destination_port == "IND_PAR"
    assert np.isclose(voyage_res.total_idle_days, voyage_res.origin_wait_days + voyage_res.destination_wait_days, atol=1e-2)
    assert voyage_res.demurrage_exposure_days >= 0.0
    assert voyage_res.demurrage_risk_level in ["low", "moderate", "high", "critical"]
    assert 0.0 <= voyage_res.delivery_delay_probability <= 1.0


# =============================================================================
# 5. Congestion Service Tests
# =============================================================================

def test_congestion_service():
    service = CongestionService()
    pred = service.predict_congestion(port_id="IND_PAR", vessel_class="Capesize")

    required_keys = {
        "expected_wait_days", "p10_wait_days", "p50_wait_days",
        "p90_wait_days", "delay_probability", "congestion_level", "confidence"
    }
    assert required_keys.issubset(pred.keys())
    assert pred["expected_wait_days"] > 0
    assert pred["p10_wait_days"] <= pred["expected_wait_days"] <= pred["p90_wait_days"]

    wait_days = service.get_expected_wait_days("IND_PAR")
    assert isinstance(wait_days, float)
    assert wait_days > 0


# =============================================================================
# 6. Voyage Economics Dynamic Impact Tests
# =============================================================================

def test_voyage_economics_dynamic_impact():
    econ_service = VoyageEconomicsService()

    # Calculate cost with low waiting time (1.0 day) vs severe waiting time (6.0 days)
    cost_low = econ_service.calculate_cost(
        vessel_class="Capesize",
        cargo_tonnage=150000,
        sailing_distance_nm=5000.0,
        freight_rate_usd_per_day=30000.0,
        predicted_idle_days=1.0
    )

    cost_high = econ_service.calculate_cost(
        vessel_class="Capesize",
        cargo_tonnage=150000,
        sailing_distance_nm=5000.0,
        freight_rate_usd_per_day=30000.0,
        predicted_idle_days=6.0
    )

    # Higher waiting time must increase total charter days and total voyage cost
    assert cost_high.total_voyage_days > cost_low.total_voyage_days
    assert cost_high.total_voyage_cost_usd > cost_low.total_voyage_cost_usd
    assert cost_high.cost_per_tonne_usd > cost_low.cost_per_tonne_usd
    # Demurrage exposure must be higher for severe wait
    assert cost_high.breakdown.expected_demurrage_usd > cost_low.breakdown.expected_demurrage_usd


# =============================================================================
# 7. Decision Engine Dynamic Integration Tests
# =============================================================================

def test_decision_engine_dynamic_congestion_integration():
    engine = DecisionEngine()

    origin_p = get_mock_port_info("IDN_TAB")
    dest_p = get_mock_port_info("IND_DHM")

    inputs = DecisionEngineInputs(
        cargo_type="coal",
        cargo_quantity_t=100000.0,
        origin_port_id=origin_p.port_id,
        destination_port_id=dest_p.port_id,
        expected_loading_date=datetime(2026, 10, 1),
        required_delivery_date=datetime(2026, 11, 1),
        number_of_voyages=1,
        origin_port_info=origin_p,
        destination_port_info=dest_p,
        vessel_specs_db=get_mock_vessel_db()
    )

    output = engine.evaluate(inputs)
    assert output["status"] == "SUCCESS"
    assert "port_analysis" in output
    assert "congestion" in output["port_analysis"]

    cong = output["port_analysis"]["congestion"]
    assert "expected_wait_days" in cong
    assert "congestion_level" in cong
    assert "delay_probability" in cong

    # Verify explainability report contains dynamic port waiting time
    explanation_text = " ".join(output["explanation"]["primary_reasons"])
    assert "waiting time" in explanation_text or "wait days" in explanation_text


# =============================================================================
# 8. REST API Endpoint Integration Test
# =============================================================================

def test_fastapi_congestion_endpoint():
    client = TestClient(app)
    response = client.get(
        "/api/v1/congestion/predict",
        params={
            "port_id": "IND_PAR",
            "target_date": "2026-03-20",
            "vessel_class": "Capesize",
            "cargo_quantity": 150000
        }
    )

    assert response.status_code == 200, f"Endpoint error: {response.text}"
    data = response.json()

    for expected_key in [
        "expected_wait_days", "p10_wait_days", "p50_wait_days",
        "p90_wait_days", "delay_probability", "congestion_level", "confidence"
    ]:
        assert expected_key in data, f"Missing key in response: {expected_key}"

    assert data["expected_wait_days"] > 0
    assert data["p10_wait_days"] <= data["expected_wait_days"] <= data["p90_wait_days"]
    assert data["congestion_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert 0.50 <= data["confidence"] <= 1.0

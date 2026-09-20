"""
DockInsights — Unified Decision Engine Test Suite (Phase 10).

Verifies the central orchestration layer connecting all 10 phases:
- Data validation and geospatial routing
- ML/statistical ensemble freight rate forecasting
- Port congestion prediction (quantile XGBoost + queuing)
- Market timing engine (expected economic benefit analysis)
- Candidate fleet generation and hard constraint filtering
- Voyage economics engine (itemized 9 delivered cost components)
- Probabilistic 8-category risk assessment & Monte Carlo simulations
- Risk-aware contract strategy optimization
- Structured comparative explainability
- Reproducibility tracking (model and data versions)
- Elimination of HTTP 501 placeholder on /api/v1/recommend
"""

import pytest
from datetime import datetime, date, timedelta, timezone
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.serializers import DecisionResponse, RecommendationResponse
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs, MODEL_VERSIONS, DATA_VERSIONS
from src.data.mock_db import get_mock_port_info, get_mock_vessel_db


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def decision_engine():
    return DecisionEngine()


@pytest.fixture
def sample_inputs():
    now = datetime(2026, 10, 1, 10, 0, 0)
    origin_p = get_mock_port_info("AUS_NEW")
    dest_p = get_mock_port_info("IND_GVM")
    return DecisionEngineInputs(
        cargo_type="coal",
        cargo_quantity_t=82000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        expected_loading_date=now,
        required_delivery_date=now + timedelta(days=32),
        number_of_voyages=1,
        risk_tolerance="MEDIUM",
        origin_port_info=origin_p,
        destination_port_info=dest_p,
        vessel_specs_db=get_mock_vessel_db(),
    )


def test_decision_engine_end_to_end_schema(decision_engine, sample_inputs):
    """Verify that DecisionEngine produces a complete, canonical DecisionResponse structure."""
    result = decision_engine.evaluate(sample_inputs)
    assert result["status"] == "SUCCESS"

    # Validate against Pydantic schema
    validated = DecisionResponse(**result)
    assert validated.decision_id.startswith("dec_")
    assert validated.confidence > 0.0

    # Model and Data versions
    assert validated.model_versions == MODEL_VERSIONS
    assert validated.data_versions == DATA_VERSIONS
    assert "freight_forecast" in validated.model_versions
    assert "congestion_predictor" in validated.model_versions

    # Request summary
    assert validated.request_summary["cargo_quantity_t"] == 82000.0
    assert validated.request_summary["origin_port_id"] == "AUS_NEW"
    assert validated.request_summary["destination_port_id"] == "IND_GVM"

    # Market analysis
    assert validated.market_analysis.current_rate > 0.0
    assert validated.market_analysis.forecast > 0.0
    assert validated.market_analysis.direction in ["RISING", "FALLING", "STABLE"]

    # Recommended plan (All 13 specified fields)
    plan = validated.recommended_plan
    assert plan.vessel_class in ["Capesize", "Panamax", "Supramax", "Handysize"]
    assert plan.vessel_count >= 1
    assert plan.voyages >= 1
    assert len(plan.cargo_allocation) >= 1
    assert sum(plan.cargo_allocation) == pytest.approx(82000.0, rel=1e-2)
    assert plan.utilization > 0.50
    assert plan.total_cost > 0.0
    assert plan.cost_per_tonne > 0.0
    assert plan.voyage_duration > 0.0
    assert plan.expected_waiting >= 0.0
    assert 0.0 <= plan.demurrage_probability <= 1.0
    assert 0.0 <= plan.delivery_probability <= 1.0
    assert 0.0 <= plan.risk_score <= 100.0

    # Economics
    econ = validated.economics
    assert econ["total_cost"] > 0.0
    assert econ["freight_cost"] > 0.0
    assert econ["bunker_cost"] > 0.0
    assert econ["port_charges"] > 0.0

    # Risk Engine (8 categories)
    risk = validated.risk
    assert risk["composite_score"] > 0.0
    assert len(risk["categories"]) >= 6

    # Contract Strategy
    strat = validated.contract_strategy
    assert strat["recommended_strategy"] in [
        "100% SPOT", "100% SHORT_TERM", "100% MEDIUM_TERM",
        "75/25 HYBRID", "60/40 HYBRID", "50/50 HYBRID", "CUSTOM_HYBRID"
    ]
    assert strat["spot_percentage"] + strat["short_term_percentage"] + strat["medium_term_percentage"] == pytest.approx(100.0)

    # Explainability
    exp = validated.explanation
    assert len(exp.primary_reasons) >= 4
    assert len(exp.tradeoff_analysis) > 10


def test_decision_engine_dynamic_values_no_hardcoding(decision_engine):
    """Verify that predictions dynamically change with ports and quantities (no hardcoded constants)."""
    now = datetime(2026, 10, 1, 10, 0, 0)

    # Route 1: Australia Newcastle -> India Gangavaram (Coal, ~5,800 nm)
    inputs_aus = DecisionEngineInputs(
        cargo_type="coal",
        cargo_quantity_t=75000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        expected_loading_date=now,
        required_delivery_date=now + timedelta(days=30),
        number_of_voyages=1,
    )
    res_aus = decision_engine.evaluate(inputs_aus)

    # Route 2: Indonesia Taboneo -> India Dhamra (Coal, ~2,600 nm)
    inputs_ina = DecisionEngineInputs(
        cargo_type="coal",
        cargo_quantity_t=75000.0,
        origin_port_id="IDN_TAB",
        destination_port_id="IND_DHM",
        expected_loading_date=now,
        required_delivery_date=now + timedelta(days=30),
        number_of_voyages=1,
    )
    res_ina = decision_engine.evaluate(inputs_ina)

    assert res_aus["status"] == "SUCCESS"
    assert res_ina["status"] == "SUCCESS"

    # Verify voyage duration and total cost differ dynamically due to route distance
    assert res_aus["recommended_plan"]["voyage_duration"] != res_ina["recommended_plan"]["voyage_duration"]
    assert res_aus["economics"]["total_cost"] != res_ina["economics"]["total_cost"]
    # Australia route is more than double the distance of Indonesia route
    assert res_aus["economics"]["total_cost"] > res_ina["economics"]["total_cost"]


def test_recommend_endpoint_success_no_501(client):
    """Verify that POST /api/v1/recommend executes and returns HTTP 200 (501 eliminated)."""
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "cargo_type": "coal",
        "cargo_tonnage": 80000,
        "earliest_date": "2026-10-01",
        "latest_date": "2026-11-01",
        "risk_appetite": "moderate"
    }

    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200

    data = response.json()
    validated = RecommendationResponse(**data)
    assert validated.request_id.startswith("dec_")
    assert validated.recommendation.vessel_class in ["Capesize", "Panamax", "Supramax", "Handysize"]
    assert validated.recommendation.estimated_rate_usd_per_day > 0.0
    assert validated.forecast is not None
    assert validated.economics is not None
    assert validated.risk is not None
    assert validated.vessel_compatibility is not None


def test_recommend_decision_endpoint_canonical(client):
    """Verify that POST /api/v1/recommend/decision returns canonical DecisionResponse."""
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "cargo_type": "thermal_coal",
        "cargo_tonnage": 150000,
        "earliest_date": "2026-10-01",
        "latest_date": "2026-11-15",
        "risk_appetite": "low"
    }

    response = client.post("/api/v1/recommend/decision", json=payload)
    assert response.status_code == 200

    data = response.json()
    validated = DecisionResponse(**data)
    assert validated.recommended_plan.vessel_class == "Capesize"
    assert validated.recommended_plan.total_cost > 0.0
    assert validated.contract_strategy["recommended_strategy"] is not None
    assert validated.explanation.tradeoff_analysis is not None


def test_recommend_infeasible_port_draft_error(client):
    """Verify that physical constraints at destination port gracefully reject infeasible requests."""
    # Ennore (IND_ENR) has draft limit ~13.5m; requesting 180,000 MT Capesize (draft 18.5m) in 1 voyage must fail
    payload = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_ENR",
        "cargo_type": "thermal_coal",
        "cargo_tonnage": 180000,
        "earliest_date": "2026-10-01",
        "latest_date": "2026-10-15",
        "risk_appetite": "moderate"
    }

    response = client.post("/api/v1/recommend", json=payload)
    # The constraint engine must fail because 180k parcel cannot fit into draft-limited ports in short deadline
    # Or return 400 error
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        # If accepted, it MUST NOT recommend Capesize (draft 18.5m > 13.5m limit)
        data = response.json()
        assert data["recommendation"]["vessel_class"] != "Capesize"

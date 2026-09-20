"""
DockInsights — Phase 9 Comprehensive Tests: Risk-Aware Contract Optimization.

Validates:
1. All 7 contract strategies evaluated (100% SPOT, 100% SHORT_TERM, 100% MEDIUM_TERM,
   75/25 HYBRID, 60/40 HYBRID, 50/50 HYBRID, and custom combinations).
2. Monte Carlo quantitative cost & risk quantification (P10, P50, P90, volatility, availability, delivery, flexibility, demurrage).
3. Configurable risk tolerance (LOW, MEDIUM, HIGH) shifting recommendations.
4. Non-simplistic decision making (independent of simple rising/falling heuristics).
5. Explainable natural-language reasoning.
6. FastAPI endpoint POST /api/v1/contracts/optimize.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.optimization.contract_optimizer import (
    RiskAwareContractOptimizer,
    RiskAwareContractInputs,
    RiskTolerance,
    VesselAvailability,
    StrategyAllocation,
)
from src.services.contract_optimization_service import ContractOptimizationService


@pytest.fixture
def optimizer():
    return RiskAwareContractOptimizer()


@pytest.fixture
def contract_service():
    return ContractOptimizationService()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_evaluates_all_standard_and_custom_strategies(optimizer):
    """Verify all 6 standard strategies + custom strategies are evaluated with full metric sets."""
    custom = [StrategyAllocation("40/30/30 TRI-HYBRID", spot_pct=40.0, short_term_pct=30.0, medium_term_pct=30.0)]
    inputs = RiskAwareContractInputs(
        cargo_quantity_t=75000.0,
        spot_freight_rate=22.0,
        freight_volatility_pct=18.0,
        custom_strategies=custom,
        n_simulations=2000,
        seed=42,
    )

    rec = optimizer.optimize_contract_strategy(inputs)

    # 6 standard + 1 custom = 7 evaluated strategies
    assert len(rec.evaluated_strategies) == 7

    strat_names = [s.strategy_name for s in rec.evaluated_strategies]
    assert "100% SPOT" in strat_names
    assert "100% SHORT_TERM" in strat_names
    assert "100% MEDIUM_TERM" in strat_names
    assert "75/25 HYBRID" in strat_names
    assert "60/40 HYBRID" in strat_names
    assert "50/50 HYBRID" in strat_names
    assert "40/30/30 TRI-HYBRID" in strat_names

    # Check metrics for each strategy
    for strat in rec.evaluated_strategies:
        assert strat.expected_total_cost > 0
        assert strat.p10_cost <= strat.p50_cost <= strat.p90_cost
        assert 0.0 <= strat.flexibility_score <= 1.0
        assert strat.price_volatility_exposure >= 0.0
        assert 0.0 <= strat.vessel_availability_exposure <= 100.0
        assert 0.0 <= strat.delivery_risk <= 1.0
        assert strat.expected_demurrage_exposure >= 0.0
        assert len(strat.reasons) > 0


def test_risk_tolerance_shifts_recommendation(optimizer):
    """
    Acceptance Criteria: Configurable risk tolerance shifts recommendation between term and spot.
    Low risk tolerance: favor term contracts (punishes P90 tail risk).
    High risk tolerance: allow more spot exposure (captures flexibility & upside).
    """
    # Baseline setup: Spot has lower expected cost ($20/t) but higher volatility and availability risk.
    # Term contracts are slightly more expensive ($22/t) but provide price lock and guaranteed tonnage.
    inputs_low = RiskAwareContractInputs(
        cargo_quantity_t=70000.0,
        spot_freight_rate=20.0,
        short_term_freight_rate=21.5,
        medium_term_freight_rate=22.0,
        freight_volatility_pct=22.0,
        risk_tolerance=RiskTolerance.LOW,
        vessel_availability=VesselAvailability.TIGHT,
        n_simulations=2000,
        seed=42,
    )

    inputs_high = RiskAwareContractInputs(
        cargo_quantity_t=70000.0,
        spot_freight_rate=20.0,
        short_term_freight_rate=21.5,
        medium_term_freight_rate=22.0,
        freight_volatility_pct=22.0,
        risk_tolerance=RiskTolerance.HIGH,
        vessel_availability=VesselAvailability.TIGHT,
        n_simulations=2000,
        seed=42,
    )

    rec_low = optimizer.optimize_contract_strategy(inputs_low)
    rec_high = optimizer.optimize_contract_strategy(inputs_high)

    # Low risk tolerance must allocate LESS to spot than High risk tolerance
    assert rec_low.spot_percentage <= rec_high.spot_percentage
    # High risk tolerance should prioritize flexibility
    assert rec_high.flexibility_score >= rec_low.flexibility_score
    # Low risk tolerance bounds P90 tail risk more tightly or locks term
    assert rec_low.medium_term_percentage + rec_low.short_term_percentage >= rec_high.medium_term_percentage + rec_high.short_term_percentage


def test_no_simplistic_market_direction_rule(optimizer):
    """
    Acceptance Criteria: No simple rising/falling rule.
    Even in a tight/uncertain market, recommendations depend on quantitative cost/risk tradeoffs.
    """
    # Scenario: Spot rate is elevated with high volatility, but term is cheap.
    inputs = RiskAwareContractInputs(
        cargo_quantity_t=80000.0,
        spot_freight_rate=25.0,
        short_term_freight_rate=22.0,  # Term discount
        medium_term_freight_rate=21.0,
        freight_volatility_pct=25.0,
        risk_tolerance=RiskTolerance.MEDIUM,
        vessel_availability=VesselAvailability.SHORTAGE,
        n_simulations=2000,
        seed=42,
    )

    rec = optimizer.optimize_contract_strategy(inputs)
    # With term discount and shortage, term contract or term-heavy hybrid should win decisively
    assert rec.recommended_strategy in ("100% MEDIUM_TERM", "100% SHORT_TERM", "75/25 HYBRID")
    assert rec.spot_percentage <= 25.0


def test_hybrid_strategies_supported_and_blended(optimizer):
    """
    Acceptance Criteria: Hybrid strategies are supported and blend constituent risk profiles.
    """
    inputs = RiskAwareContractInputs(
        cargo_quantity_t=70000.0,
        spot_freight_rate=21.0,
        short_term_freight_rate=21.5,
        medium_term_freight_rate=22.5,
        freight_volatility_pct=16.0,
        risk_tolerance=RiskTolerance.MEDIUM,
        n_simulations=2000,
        seed=42,
    )

    rec = optimizer.optimize_contract_strategy(inputs)
    strat_dict = {s.strategy_name: s for s in rec.evaluated_strategies}

    spot = strat_dict["100% SPOT"]
    short = strat_dict["100% SHORT_TERM"]
    hybrid_50 = strat_dict["50/50 HYBRID"]

    # 50/50 hybrid flexibility score must be exactly midpoint between spot and short
    assert pytest.approx(hybrid_50.flexibility_score, abs=0.01) == 0.5 * spot.flexibility_score + 0.5 * short.flexibility_score
    # 50/50 hybrid P90 cost must lie between pure short and pure spot
    min_p90 = min(spot.p90_cost, short.p90_cost)
    max_p90 = max(spot.p90_cost, short.p90_cost)
    assert min_p90 <= hybrid_50.p90_cost <= max_p90


def test_explainable_reasons_returned(optimizer):
    """
    Acceptance Criteria: The recommendation must be explainable.
    """
    inputs = RiskAwareContractInputs(
        cargo_quantity_t=65000.0,
        spot_freight_rate=23.0,
        freight_volatility_pct=15.0,
        risk_tolerance=RiskTolerance.MEDIUM,
        n_simulations=2000,
        seed=42,
    )

    rec = optimizer.optimize_contract_strategy(inputs)

    assert len(rec.reasons) >= 2
    reasons_text = " ".join(rec.reasons)
    # Check that reasons cite key quantitative dimensions
    assert "$" in reasons_text or "cost" in reasons_text.lower()
    assert "risk" in reasons_text.lower() or "tolerance" in reasons_text.lower()


def test_service_layer_optimize_risk_aware(contract_service):
    """Verify ContractOptimizationService wrapper accepts dict and returns canonical format."""
    payload = {
        "cargo_quantity_t": 75000.0,
        "spot_freight_rate": 22.0,
        "freight_volatility_pct": 15.0,
        "risk_tolerance": "LOW",
        "vessel_availability": "TIGHT",
        "n_simulations": 1500,
        "seed": 42,
    }

    result = contract_service.optimize_risk_aware(payload)

    assert hasattr(result, "recommended_strategy")
    assert hasattr(result, "spot_percentage")
    assert hasattr(result, "short_term_percentage")
    assert hasattr(result, "medium_term_percentage")
    assert hasattr(result, "expected_cost")
    assert hasattr(result, "p90_cost")
    assert hasattr(result, "risk_score")
    assert hasattr(result, "flexibility_score")
    assert hasattr(result, "reasons")


def test_api_contract_optimization_endpoint(client):
    """Verify POST /api/v1/contracts/optimize returns valid JSON and correct schema."""
    payload = {
        "cargo_quantity_t": 75000.0,
        "spot_freight_rate": 21.5,
        "freight_volatility_pct": 16.0,
        "risk_tolerance": "HIGH",
        "vessel_availability": "ABUNDANT",
        "n_simulations": 2000,
        "seed": 42,
    }

    response = client.post("/api/v1/contracts/optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # Verify all required top-level keys
    assert "recommended_strategy" in data
    assert "spot_percentage" in data
    assert "short_term_percentage" in data
    assert "medium_term_percentage" in data
    assert "expected_cost" in data
    assert "p90_cost" in data
    assert "risk_score" in data
    assert "flexibility_score" in data
    assert "reasons" in data
    assert "evaluated_strategies" in data

    assert len(data["reasons"]) >= 2
    assert len(data["evaluated_strategies"]) >= 6
    assert data["expected_cost"] > 0
    assert data["p90_cost"] >= data["expected_cost"]

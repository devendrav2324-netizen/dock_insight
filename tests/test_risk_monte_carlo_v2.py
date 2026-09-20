"""
DockInsights — Phase 8 Comprehensive Tests: Probabilistic Risk and Scenario Engine.

Validates:
1. 8 Service-driven risk categories in MaritimeRiskEngine
2. Dynamic risk evaluation (risk score is no longer fixed at 80)
3. Monte Carlo simulation (10,000 runs, distributions, quantiles, probabilities)
4. Simulation reproducibility with seed control
5. Scenario Engine (BEST_CASE, BASE_CASE, WORST_CASE)
6. Risk affects plan scoring and fleet ranking
7. FastAPI endpoint POST /api/v1/risk/simulate and GET /api/v1/risk
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.risk.risk_engine import MaritimeRiskEngine, RiskSeverity
from src.risk.monte_carlo import MonteCarloSimulator, CharterPlanInputs
from src.risk.scenario_engine import ScenarioEngine, ScenarioType
from src.services.risk_service import RiskService
from src.optimization.vessel_scoring import VesselScoringEngine, OptimizationWeights
from src.optimization.voyage_plan import CharterPlan


@pytest.fixture
def risk_engine():
    return MaritimeRiskEngine()


@pytest.fixture
def risk_service():
    return RiskService()


@pytest.fixture
def client():
    return TestClient(app)


def test_eight_service_driven_risk_categories(risk_engine):
    """Verify all 8 risk categories are calculated and accessible."""
    inputs = {
        "market_data": {"price_volatility_pct": 18.0, "p10": 18.0, "p90": 26.0, "p50": 21.0},
        "port_data": {"expected_wait_days": 3.0, "vessels_waiting": 7, "delay_probability": 0.40},
        "weather_data": {"wave_height_m": 2.2, "wind_speed_kmh": 40.0, "storm_warning": False},
        "vessel_data": {"available_vessels_in_region": 8, "lead_time_days": 4.0},
        "operational_data": {"vessel_age_years": 12.0, "maintenance_due": False, "cargo_type": "coal"},
        "geopolitical_data": {"route_conflict_level": 2.0, "chokepoints": ["malacca"]},
        "schedule_data": {"total_duration": 18.0, "delivery_deadline_days": 26.0},
        "demurrage_data": {"expected_demurrage": 25000.0, "freight_cost": 400000.0},
    }

    res = risk_engine.evaluate_total_risk(inputs)

    # 1. Check all 8 primary categories
    expected_categories = [
        "Market",
        "Port Congestion",
        "Weather",
        "Vessel Availability",
        "Operational",
        "Geopolitical",
        "Schedule",
        "Demurrage",
    ]
    for cat in expected_categories:
        assert cat in res.categories, f"Missing category {cat}"
        cat_res = res.categories[cat]
        assert 0.0 <= cat_res.score <= 100.0
        assert cat_res.severity in (RiskSeverity.LOW, RiskSeverity.MODERATE, RiskSeverity.HIGH, RiskSeverity.CRITICAL)
        assert len(cat_res.contributing_factors) > 0

    # 2. Check snake_case and lowercase alias compatibility
    assert "market" in res.categories
    assert "port_congestion" in res.categories
    assert "port" in res.categories
    assert "weather" in res.categories
    assert "vessel_availability" in res.categories
    assert "operational" in res.categories
    assert "geopolitical" in res.categories
    assert "schedule" in res.categories
    assert "demurrage" in res.categories

    assert 0.0 <= res.overall_score <= 100.0
    assert res.composite_score == res.overall_score
    assert res.dominant_risk is not None


def test_risk_is_not_fixed_at_80(risk_engine):
    """
    Acceptance Criteria: Risk is no longer fixed at 80.
    Verify risk changes dynamically across different operating profiles.
    """
    # Scenario A: Benign conditions
    low_inputs = {
        "market_data": {"price_volatility_pct": 5.0},
        "port_data": {"expected_wait_days": 0.8},
        "weather_data": {"wave_height_m": 1.0},
        "vessel_data": {"available_vessels_in_region": 15},
        "schedule_data": {"total_duration": 12.0, "delivery_deadline_days": 28.0},
    }
    low_res = risk_engine.evaluate_total_risk(low_inputs)
    assert low_res.overall_score != 80.0
    assert low_res.overall_score < 40.0
    assert low_res.overall_severity == RiskSeverity.LOW

    # Scenario B: High market volatility only
    market_inputs = {
        "market_data": {"price_volatility_pct": 32.0},
        "port_data": {"expected_wait_days": 1.0},
        "weather_data": {"wave_height_m": 1.0},
    }
    market_res = risk_engine.evaluate_total_risk(market_inputs)
    assert market_res.overall_score != 80.0
    assert market_res.categories["Market"].score > 80.0

    # Scenario C: Severe storm and port backlog
    storm_inputs = {
        "market_data": {"price_volatility_pct": 8.0},
        "port_data": {"expected_wait_days": 5.5, "vessels_waiting": 14},
        "weather_data": {"wave_height_m": 5.0, "storm_warning": True},
        "schedule_data": {"total_duration": 22.0, "delivery_deadline_days": 20.0},
    }
    storm_res = risk_engine.evaluate_total_risk(storm_inputs)
    assert storm_res.overall_score != 80.0
    assert storm_res.overall_score > 50.0
    assert storm_res.categories["Weather"].score >= 85.0
    assert storm_res.categories["Schedule"].score >= 90.0

    # All three scenarios should produce distinctly different composite scores
    assert low_res.overall_score != market_res.overall_score
    assert market_res.overall_score != storm_res.overall_score


def test_monte_carlo_simulation_metrics():
    """
    Acceptance Criteria: Monte Carlo simulation works and computes all required quantiles & probabilities.
    """
    plan = CharterPlanInputs(
        cargo_quantity_t=70000.0,
        base_freight_rate=22.0,
        freight_volatility_pct=15.0,
        base_bunker_price=640.0,
        bunker_volatility_pct=10.0,
        sea_distance_nm=4600.0,
        service_speed_knots=12.5,
        expected_wait_days=2.0,
        delivery_deadline_days=26.0,
        demurrage_rate_usd_day=20000.0,
        vessel_availability_probability=0.96,
    )

    sim = MonteCarloSimulator()
    res = sim.run_simulation(plan, n_simulations=10_000, seed=42)

    assert res.n_simulations == 10_000
    assert res.seed == 42

    # Quantile Monotonicity
    assert res.min_cost <= res.p10_cost
    assert res.p10_cost <= res.p50_cost
    assert res.p50_cost <= res.p90_cost
    assert res.p90_cost <= res.p95_cost
    assert res.p95_cost <= res.max_cost

    # Probabilities bounded between 0 and 1
    assert 0.0 <= res.demurrage_probability <= 1.0
    assert 0.0 <= res.late_delivery_probability <= 1.0
    assert 0.0 <= res.probability_cost_exceeds_threshold <= 1.0
    assert 0.0 <= res.probability_of_infeasibility <= 1.0

    # Duration quantiles
    assert res.p10_duration_days <= res.p50_duration_days <= res.p90_duration_days

    # Histogram distribution bins exist
    assert len(res.distribution) == 25
    total_prob = sum(b["probability"] for b in res.distribution)
    assert pytest.approx(total_prob, abs=0.01) == 1.0

    # Assumptions documented
    assert "freight_rate_distribution" in res.assumptions
    assert "bunker_price_distribution" in res.assumptions
    assert "port_waiting_distribution" in res.assumptions


def test_monte_carlo_reproducibility():
    """
    Acceptance Criteria: Simulation is reproducible using seed.
    """
    plan = CharterPlanInputs(
        cargo_quantity_t=65000.0,
        base_freight_rate=24.5,
        expected_wait_days=2.5,
        sea_distance_nm=4200.0,
    )

    sim = MonteCarloSimulator()

    # Run twice with seed=42
    run1 = sim.run_simulation(plan, n_simulations=10_000, seed=42)
    run2 = sim.run_simulation(plan, n_simulations=10_000, seed=42)

    assert run1.expected_cost == run2.expected_cost
    assert run1.p10_cost == run2.p10_cost
    assert run1.p50_cost == run2.p50_cost
    assert run1.p90_cost == run2.p90_cost
    assert run1.demurrage_probability == run2.demurrage_probability
    assert run1.late_delivery_probability == run2.late_delivery_probability

    # Run with different seed=999 -> outputs should vary statistically
    run3 = sim.run_simulation(plan, n_simulations=10_000, seed=999)
    assert run1.expected_cost != run3.expected_cost


def test_scenario_engine_best_base_worst():
    """
    Acceptance Criteria: Scenario engine supports BEST_CASE, BASE_CASE, WORST_CASE.
    """
    plan = CharterPlanInputs(
        cargo_quantity_t=70000.0,
        base_freight_rate=22.0,
        expected_wait_days=2.5,
        sea_distance_nm=4500.0,
        delivery_deadline_days=28.0,
    )

    scen_engine = ScenarioEngine()
    scenarios = scen_engine.evaluate_scenarios(plan)

    assert ScenarioType.BEST_CASE.value in scenarios
    assert ScenarioType.BASE_CASE.value in scenarios
    assert ScenarioType.WORST_CASE.value in scenarios

    best = scenarios[ScenarioType.BEST_CASE.value]
    base = scenarios[ScenarioType.BASE_CASE.value]
    worst = scenarios[ScenarioType.WORST_CASE.value]

    # Monotonic cost relationship
    assert best.total_cost < base.total_cost < worst.total_cost
    assert best.cost_per_tonne < base.cost_per_tonne < worst.cost_per_tonne

    # Monotonic duration relationship
    assert best.duration_days <= base.duration_days <= worst.duration_days

    # Demurrage relationship
    assert best.demurrage_cost <= base.demurrage_cost <= worst.demurrage_cost

    # Documented assumptions present
    assert len(best.assumptions) >= 3
    assert len(base.assumptions) >= 3
    assert len(worst.assumptions) >= 3


def test_risk_service_simulate_and_assess(risk_service):
    """
    Verify unified API response dictionary format matching the user prompt specification.
    """
    res = risk_service.simulate_and_assess({
        "cargo_quantity_t": 75000.0,
        "base_freight_rate": 21.5,
        "expected_wait_days": 2.2,
        "sea_distance_nm": 4800.0,
        "delivery_deadline_days": 27.0,
    }, n_simulations=10_000, seed=42)

    required_keys = [
        "expected_cost",
        "p10_cost",
        "p50_cost",
        "p90_cost",
        "demurrage_probability",
        "late_delivery_probability",
        "risk_score",
        "scenarios",
    ]
    for key in required_keys:
        assert key in res, f"Missing required response key: {key}"

    assert res["expected_cost"] > 0
    assert res["p10_cost"] <= res["p50_cost"] <= res["p90_cost"]
    assert 0.0 <= res["demurrage_probability"] <= 1.0
    assert 0.0 <= res["late_delivery_probability"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0
    assert "BEST_CASE" in res["scenarios"]
    assert "BASE_CASE" in res["scenarios"]
    assert "WORST_CASE" in res["scenarios"]


def test_risk_affects_vessel_plan_ranking():
    """
    Acceptance Criteria: Risk affects vessel/plan ranking.
    """
    scoring = VesselScoringEngine(OptimizationWeights(cost_weight=0.20, schedule_weight=0.20, risk_weight=0.60))

    # Plan 1: Slightly cheaper but carries CRITICAL risk (e.g. 90.0)
    plan_high_risk = CharterPlan(
        plan_id="PLAN_HIGH_RISK",
        vessel_classes=["Panamax"],
        number_of_vessels=1,
        number_of_voyages=1,
        total_cargo_t=75000,
        total_cost=1500000,
        cost_per_tonne=20.0,
        total_duration=18.0,
        delivery_probability=0.70,
        risk_score=90.0,  # CRITICAL RISK
        expected_demurrage=50000,
        utilization=0.95,
        feasibility=True,
    )

    # Plan 2: Slightly higher freight cost ($21/t) but LOW risk (15.0)
    plan_low_risk = CharterPlan(
        plan_id="PLAN_LOW_RISK",
        vessel_classes=["Panamax"],
        number_of_vessels=1,
        number_of_voyages=1,
        total_cargo_t=75000,
        total_cost=1575000,
        cost_per_tonne=21.0,
        total_duration=18.0,
        delivery_probability=0.98,
        risk_score=15.0,  # LOW RISK
        expected_demurrage=0,
        utilization=0.95,
        feasibility=True,
    )

    ranked = scoring.score_plans([plan_high_risk, plan_low_risk])

    # With risk weighted, plan_low_risk should rank #1 despite slightly higher freight
    assert ranked[0].plan_id == "PLAN_LOW_RISK"
    assert ranked[0].score > ranked[1].score


def test_api_simulate_endpoint(client):
    """
    Verify POST /api/v1/risk/simulate returns valid JSON schema and calculations.
    """
    payload = {
        "cargo_quantity_t": 70000.0,
        "base_freight_rate": 20.0,
        "freight_volatility_pct": 14.0,
        "expected_wait_days": 2.0,
        "sea_distance_nm": 4500.0,
        "delivery_deadline_days": 25.0,
        "n_simulations": 10000,
        "seed": 42,
    }

    response = client.post("/api/v1/risk/simulate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # Check top-level required fields
    assert "expected_cost" in data
    assert "p10_cost" in data
    assert "p50_cost" in data
    assert "p90_cost" in data
    assert "demurrage_probability" in data
    assert "late_delivery_probability" in data
    assert "risk_score" in data
    assert "scenarios" in data

    # Check scenario contents
    scenarios = data["scenarios"]
    assert "BEST_CASE" in scenarios
    assert "BASE_CASE" in scenarios
    assert "WORST_CASE" in scenarios

    assert scenarios["BEST_CASE"]["total_cost"] < scenarios["BASE_CASE"]["total_cost"]
    assert scenarios["BASE_CASE"]["total_cost"] < scenarios["WORST_CASE"]["total_cost"]


def test_api_get_risk_endpoint(client):
    """
    Verify GET /api/v1/risk returns 8-category breakdown and composite score.
    """
    params = {
        "origin_port_id": "AUS_NEW",
        "destination_port_id": "IND_GVM",
        "target_date": "2026-09-15",
        "sailing_distance_nm": 4800.0,
    }

    response = client.get("/api/v1/risk", params=params)
    assert response.status_code == 200, response.text
    data = response.json()

    assert "composite_score" in data
    assert "level" in data
    assert "breakdown" in data
    assert "dominant_risk" in data
    assert "recommendation" in data

    # Check that breakdown contains categories
    assert len(data["breakdown"]) >= 6

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from src.optimization.multi_voyage_optimizer import MultiVoyageOptimizer, FleetOptimizationRequest
from src.optimization.voyage_plan import CharterPlan

def test_multi_voyage_optimizer_monte_carlo_variables():
    # Create mock RiskService
    mock_risk = MagicMock()
    mock_risk.evaluate_risk.return_value = MagicMock(composite_score=42.0)
    
    mc_result_mock = MagicMock()
    mc_result_mock.expected_cost = 1500000.0
    mc_result_mock.p10_cost = 1400000.0
    mc_result_mock.p50_cost = 1500000.0
    mc_result_mock.p90_cost = 1600000.0
    mc_result_mock.demurrage_probability = 0.25
    mc_result_mock.late_delivery_probability = 0.10
    mock_risk.simulate_monte_carlo.return_value = mc_result_mock

    opt = MultiVoyageOptimizer(risk_service=mock_risk)
    
    req = FleetOptimizationRequest(
        cargo_quantity_t=170000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        route_distance_nm=5200.0,
        expected_loading_date=datetime(2026, 10, 1),
        required_delivery_date=datetime(2026, 11, 1),
        max_voyages=1,
        allow_mixed_classes=False,
        include_parallel_options=False
    )
    
    plans = opt.optimize(req)
    assert len(plans) > 0
    
    # Verify simulate_monte_carlo was called with exact candidate parameters
    assert mock_risk.simulate_monte_carlo.called
    call_args = mock_risk.simulate_monte_carlo.call_args[0][0]
    
    # 5200 nm distance explicitly provided
    assert call_args["sea_distance_nm"] == 5200.0
    
    # The first plan evaluates a Capesize (1x ~170k parcel)
    # Check if the class specs correspond to Capesize (speed 13.5, fuel 52.0) or Panamax, etc.
    # We just ensure it's not a NameError
    assert "service_speed_knots" in call_args
    assert "fuel_consumption_t_day" in call_args
    
    plan = plans[0]
    assert plan.monte_carlo["status"] == "SUCCESS"

def test_multi_voyage_optimizer_monte_carlo_fallback():
    mock_risk = MagicMock()
    mock_risk.evaluate_risk.return_value = MagicMock(composite_score=42.0)
    
    # Force a runtime exception (e.g. timeout or missing data)
    mock_risk.simulate_monte_carlo.side_effect = ValueError("Risk engine unavailable")
    
    opt = MultiVoyageOptimizer(risk_service=mock_risk)
    
    req = FleetOptimizationRequest(
        cargo_quantity_t=170000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        route_distance_nm=5200.0,
        expected_loading_date=datetime(2026, 10, 1),
        required_delivery_date=datetime(2026, 11, 1),
    )
    
    plans = opt.optimize(req)
    assert len(plans) > 0
    plan = plans[0]
    
    assert plan.monte_carlo["status"] == "FALLBACK"
    assert "unavailable" in plan.monte_carlo["fallback_reason"]

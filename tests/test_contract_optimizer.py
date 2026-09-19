import pytest
from src.optimization.contract_optimizer import (
    ContractOptimizer,
    ContractOptimizationInputs,
    ContractStrategy,
    ForecastDirection,
    ForecastUncertainty,
    VesselAvailability
)

@pytest.fixture
def optimizer():
    return ContractOptimizer()

def test_rising_rates_low_uncertainty(optimizer):
    inputs = ContractOptimizationInputs(
        current_freight_rate=20.0,
        expected_future_rate=25.0,
        forecast_direction=ForecastDirection.RISING,
        forecast_uncertainty=ForecastUncertainty.LOW,
        vessel_availability=VesselAvailability.ABUNDANT,
        congestion_level=2.0,
        overall_risk_score=20.0,
        voyage_cost=100000.0,
        number_of_required_voyages=5
    )
    
    result = optimizer.recommend(inputs)
    assert result.strategy == ContractStrategy.MEDIUM_TERM
    assert any("upward trend" in reason for reason in result.reasons)
    assert any("High confidence" in reason for reason in result.reasons)

def test_falling_rates_abundant_vessels(optimizer):
    inputs = ContractOptimizationInputs(
        current_freight_rate=20.0,
        expected_future_rate=15.0,
        forecast_direction=ForecastDirection.FALLING,
        forecast_uncertainty=ForecastUncertainty.LOW,
        vessel_availability=VesselAvailability.ABUNDANT,
        congestion_level=2.0,
        overall_risk_score=20.0,
        voyage_cost=100000.0,
        number_of_required_voyages=5
    )
    
    result = optimizer.recommend(inputs)
    assert result.strategy == ContractStrategy.SPOT
    assert any("abundant" in reason for reason in result.reasons)

def test_vessel_shortage_override(optimizer):
    inputs = ContractOptimizationInputs(
        current_freight_rate=20.0,
        expected_future_rate=15.0,
        forecast_direction=ForecastDirection.FALLING,
        forecast_uncertainty=ForecastUncertainty.LOW,
        vessel_availability=VesselAvailability.SHORTAGE,
        congestion_level=2.0,
        overall_risk_score=20.0,
        voyage_cost=100000.0,
        number_of_required_voyages=5
    )
    
    result = optimizer.recommend(inputs)
    # Even if rates are falling, a shortage forces MEDIUM_TERM locking
    assert result.strategy == ContractStrategy.MEDIUM_TERM
    assert any("shortage overrides" in reason for reason in result.reasons)

def test_uncertain_tight_scenario(optimizer):
    inputs = ContractOptimizationInputs(
        current_freight_rate=20.0,
        expected_future_rate=20.0,
        forecast_direction=ForecastDirection.UNCERTAIN,
        forecast_uncertainty=ForecastUncertainty.HIGH,
        vessel_availability=VesselAvailability.TIGHT,
        congestion_level=5.0,
        overall_risk_score=80.0,
        voyage_cost=100000.0,
        number_of_required_voyages=5
    )
    
    result = optimizer.recommend(inputs)
    assert result.strategy == ContractStrategy.HYBRID
    assert "50/50" in result.recommendation_text or "50%" in result.recommendation_text
    assert any("risk diversification" in reason for reason in result.reasons)

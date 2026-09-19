import pytest
from datetime import datetime, timedelta
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.optimization.port_compatibility import PortInfo
from src.optimization.vessel_selector import VesselSpecs

from src.services.freight_forecast_service import FreightForecastService
from src.services.risk_service import RiskService
from src.services.voyage_economics_service import VoyageEconomicsService
from src.services.vessel_optimization_service import VesselOptimizationService
from src.services.contract_optimization_service import ContractOptimizationService

@pytest.fixture
def decision_engine():
    return DecisionEngine(
        forecast_service=FreightForecastService(),
        risk_service=RiskService(),
        economics_service=VoyageEconomicsService(),
        vessel_service=VesselOptimizationService(),
        contract_service=ContractOptimizationService()
    )

@pytest.fixture
def vessel_db():
    return [
        VesselSpecs(class_name="Handysize", dwt_max=40000, dwt_min=15000, typical_dwt=30000, draft_max_m=10.0, loa_max_m=180.0, beam_max_m=28.0),
        VesselSpecs(class_name="Supramax", dwt_max=60000, dwt_min=40000, typical_dwt=55000, draft_max_m=12.0, loa_max_m=200.0, beam_max_m=32.0),
        VesselSpecs(class_name="Panamax", dwt_max=85000, dwt_min=60000, typical_dwt=75000, draft_max_m=14.5, loa_max_m=230.0, beam_max_m=32.2),
        VesselSpecs(class_name="Capesize", dwt_max=400000, dwt_min=100000, typical_dwt=180000, draft_max_m=18.5, loa_max_m=290.0, beam_max_m=45.0)
    ]

@pytest.fixture
def valid_inputs(vessel_db):
    now = datetime.now()
    return DecisionEngineInputs(
        cargo_type="Coal",
        cargo_quantity_t=82000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        expected_loading_date=now,
        required_delivery_date=now + timedelta(days=30),
        number_of_voyages=5,
        vessel_specs_db=vessel_db,
        origin_port_info=PortInfo("AUS_NEW", "Newcastle", 20.0, 300.0, 50.0, 60000.0, 20),
        destination_port_info=PortInfo("IND_GVM", "Gangavaram", 21.0, 300.0, 50.0, 50000.0, 6)
    )

def test_successful_integration(decision_engine, valid_inputs):
    result = decision_engine.evaluate(valid_inputs)
    assert result["status"] == "SUCCESS"
    assert "market_forecast" in result
    assert "recommended_vessel" in result
    assert "port_analysis" in result
    assert "voyage_economics" in result
    assert "risk_analysis" in result
    assert "contract_strategy" in result
    assert "final_recommendation" in result
    assert "explanation" in result

    assert result["recommended_vessel"]["class"] == "Panamax"
    assert result["explanation"] is not None
    assert "recommendation_summary" in result["explanation"]
    assert len(result["explanation"]["primary_reasons"]) >= 5
    assert isinstance(result["explanation"]["alternatives_rejected"], list)

def test_graceful_failure(decision_engine, valid_inputs):
    # Overload cargo quantity so Capesize is required, but restrict port draft severely to force failure
    valid_inputs.cargo_quantity_t = 150000.0 # Requires Capesize
    valid_inputs.destination_port_info.max_draft_m = 10.0 # Capesize draft is 18.5, will fail port check if vessel selector allows it, or vessel selector will fail first

    result = decision_engine.evaluate(valid_inputs)
    assert result["status"] == "ERROR"
    assert "No feasible vessel" in result["error_message"] or "failed" in result["error_message"]
    assert result["explanation"] is None

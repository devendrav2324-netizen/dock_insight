"""
Charter-AI — Risk Assessment and Simulation Endpoints (Phase 8).
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query

from src.api.serializers import (
    RiskAssessmentResponse,
    RiskDimensionResponse,
    RiskSimulationRequest,
    RiskSimulationResponse,
)
from src.services.risk_service import RiskService
from src.risk.aggregator import RiskAggregator
from src.risk.operational_risk import OperationalRiskAssessor
from src.risk.weather_risk import WeatherRiskAssessor

router = APIRouter(prefix="/risk", tags=["Risk"])

_risk_service = RiskService()
_weather_assessor = WeatherRiskAssessor()
_operational_assessor = OperationalRiskAssessor()
_aggregator = RiskAggregator()


@router.post("/simulate", response_model=RiskSimulationResponse)
async def simulate_risk(request: RiskSimulationRequest) -> RiskSimulationResponse:
    """
    Run high-performance Monte Carlo simulation (10,000 runs) and scenario stress tests.

    Simulates uncertainty across:
    - Freight rate volatility (Log-normal distribution)
    - Bunker price volatility (Log-normal distribution)
    - Port waiting time & queue congestion (Gamma distribution)
    - Sea voyage duration & weather speed loss (Beta distribution)
    - Laytime and demurrage liability
    - Vessel availability & fixture cancellation risk

    Returns expected, P10, P50, P90 delivered costs, demurrage probability,
    late delivery probability, composite risk score, and scenario stress tests.
    """
    result = _risk_service.simulate_and_assess(
        plan_inputs=request.model_dump(),
        n_simulations=request.n_simulations,
        seed=request.seed,
    )

    return RiskSimulationResponse(
        expected_cost=result["expected_cost"],
        p10_cost=result["p10_cost"],
        p50_cost=result["p50_cost"],
        p90_cost=result["p90_cost"],
        demurrage_probability=result["demurrage_probability"],
        late_delivery_probability=result["late_delivery_probability"],
        risk_score=result["risk_score"],
        probability_of_infeasibility=result.get("probability_of_infeasibility", 0.0),
        scenarios=result["scenarios"],
        cost_distribution=result.get("cost_distribution", []),
        risk_assessment=result.get("risk_assessment"),
        seed=result["seed"],
        n_simulations=result["n_simulations"],
    )


@router.get("", response_model=RiskAssessmentResponse)
async def assess_risk(
    origin_port_id: str,
    destination_port_id: str,
    target_date: date,
    sailing_distance_nm: float = 5000.0,
    routing_note: str = "",
    vessel_class: Optional[str] = "Panamax",
):
    """
    Get comprehensive composite risk assessment for a route and date.
    Consumes live weather, operational, port congestion, and market risk dimensions.
    """
    inputs = {
        "vessel_class": vessel_class,
        "port_data": {
            "port_id": destination_port_id,
        },
        "operational_data": {
            "origin_port_id": origin_port_id,
            "sailing_distance_nm": sailing_distance_nm,
            "routing_note": routing_note,
        },
        "geopolitical_data": {
            "routing_note": routing_note,
        },
        "weather_data": {
            "port_id": destination_port_id,
            "target_date": target_date,
        },
    }

    assessment = _risk_service.evaluate_risk(inputs)

    breakdown = {}
    for dim_name, dim_result in assessment.categories.items():
        if dim_name.islower():
            continue
        breakdown[dim_name] = RiskDimensionResponse(
            score=dim_result.score,
            level=dim_result.severity.value,
            detail="; ".join(dim_result.contributing_factors) if dim_result.contributing_factors else dim_result.category,
        )

    return RiskAssessmentResponse(
        composite_score=assessment.composite_score,
        level=assessment.level.value,
        breakdown=breakdown,
        dominant_risk=assessment.dominant_risk,
        recommendation=assessment.recommendation,
    )

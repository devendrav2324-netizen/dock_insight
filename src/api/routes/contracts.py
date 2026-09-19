"""
Charter-AI — Contract Optimization Endpoint (Phase 9).
"""

from fastapi import APIRouter

from src.api.serializers import (
    ContractOptimizationApiRequest,
    ContractOptimizationApiResponse,
)
from src.services.contract_optimization_service import ContractOptimizationService

router = APIRouter(prefix="/contracts", tags=["Contracts"])

_contract_service = ContractOptimizationService()


@router.post("/optimize", response_model=ContractOptimizationApiResponse)
async def optimize_contract_strategy(request: ContractOptimizationApiRequest) -> ContractOptimizationApiResponse:
    """
    Quantitatively optimize chartering contract strategy across Spot, Term, and Hybrid allocations.

    Uses Monte Carlo simulation and a multi-objective utility function:
        Objective = Expected Total Cost + Risk Penalty + Schedule Penalty - Flexibility Benefit
    Calibrated by user risk tolerance: LOW (favors term), MEDIUM (balanced), HIGH (favors spot flexibility).
    """
    result = _contract_service.optimize_risk_aware(request.model_dump())

    return ContractOptimizationApiResponse(
        recommended_strategy=result.recommended_strategy,
        spot_percentage=result.spot_percentage,
        short_term_percentage=result.short_term_percentage,
        medium_term_percentage=result.medium_term_percentage,
        expected_cost=result.expected_cost,
        p90_cost=result.p90_cost,
        risk_score=result.risk_score,
        flexibility_score=result.flexibility_score,
        reasons=result.reasons,
        evaluated_strategies=[s.to_dict() for s in result.evaluated_strategies],
    )

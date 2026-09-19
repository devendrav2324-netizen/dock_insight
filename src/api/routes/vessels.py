"""
Charter-AI — Vessels Endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import VesselClassResponse, VesselSelectionResponse, VesselCompatibilityResponse
from src.data.repository import PortRepository, VesselClassRepository
from src.optimization.vessel_selector import PortConstraints, VesselSelector, VesselSpecs

router = APIRouter(prefix="/vessels", tags=["Vessels"])

_selector = VesselSelector()


@router.get("", response_model=List[VesselClassResponse])
async def list_vessel_classes(
    db: AsyncSession = Depends(get_session),
):
    """List all vessel classes with their specifications."""
    repo = VesselClassRepository(db)
    classes = await repo.get_all()
    return [
        VesselClassResponse(
            class_name=vc.class_name,
            dwt_min=vc.dwt_min,
            dwt_max=vc.dwt_max,
            typical_dwt=vc.typical_dwt,
            draft_max_m=vc.draft_max_m,
            loa_max_m=vc.loa_max_m,
            beam_max_m=vc.beam_max_m,
        )
        for vc in classes
    ]


@router.get("/compatibility", response_model=VesselSelectionResponse)
async def check_vessel_compatibility(
    origin_port_id: str,
    destination_port_id: str,
    cargo_tonnage: Optional[int] = None,
    db: AsyncSession = Depends(get_session),
):
    """
    Check which vessel classes are compatible with an origin-destination pair.

    Applies hard physical constraints: draft, LOA, beam.
    """
    port_repo = PortRepository(db)
    vessel_repo = VesselClassRepository(db)

    origin = await port_repo.get_by_id(origin_port_id)
    destination = await port_repo.get_by_id(destination_port_id)
    all_vessels = await vessel_repo.get_all()

    if not origin or not destination:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Origin or destination port not found")

    origin_constraints = PortConstraints(
        port_id=origin.port_id,
        port_name=origin.port_name,
        max_draft_m=origin.max_draft_m or 0,
        max_loa_m=origin.max_loa_m or 0,
        max_beam_m=origin.max_beam_m or 0,
        max_dwt=origin.max_dwt or 0,
        is_anchorage_only=(origin.max_draft_m == -9999),
    )
    dest_constraints = PortConstraints(
        port_id=destination.port_id,
        port_name=destination.port_name,
        max_draft_m=destination.max_draft_m or 0,
        max_loa_m=destination.max_loa_m or 0,
        max_beam_m=destination.max_beam_m or 0,
        max_dwt=destination.max_dwt or 0,
        is_anchorage_only=(destination.max_draft_m == -9999),
    )

    vessel_specs = [
        VesselSpecs(
            class_name=vc.class_name,
            dwt_min=vc.dwt_min,
            dwt_max=vc.dwt_max,
            typical_dwt=vc.typical_dwt,
            draft_max_m=vc.draft_max_m,
            loa_max_m=vc.loa_max_m,
            beam_max_m=vc.beam_max_m,
        )
        for vc in all_vessels
    ]

    result = _selector.select_vessels(
        origin_port=origin_constraints,
        destination_port=dest_constraints,
        vessel_specs=vessel_specs,
        cargo_tonnage=cargo_tonnage,
    )

    return VesselSelectionResponse(
        origin_port_id=result.origin_port_id,
        destination_port_id=result.destination_port_id,
        feasible=[
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.feasible
        ],
        excluded=[
            VesselCompatibilityResponse(
                vessel_class=r.vessel_class,
                is_compatible=r.is_compatible,
                violations=r.violations,
            )
            for r in result.excluded
        ],
    )


class FleetPlanOptimizationRequest(BaseModel):
    """Request body for multi-voyage fleet and vessel plan optimization."""
    cargo_quantity_t: float = Field(..., gt=0, description="Total cargo volume in metric tonnes (MT)")
    origin_port_id: str = Field(default="AUS_NEW", description="Origin load port ID")
    destination_port_id: str = Field(default="IND_GVM", description="Destination discharge port ID")
    cargo_type: str = Field(default="Coal", description="Dry bulk commodity name")
    route_distance_nm: float = Field(default=4800.0, gt=0, description="Actual nautical distance")
    freight_rate_usd: float = Field(default=22.0, gt=0, description="Baseline freight rate USD/MT")
    delivery_deadline_days: Optional[float] = Field(default=None, gt=0, description="Target delivery deadline (days)")
    max_voyages: int = Field(default=5, ge=1, le=10, description="Maximum voyage count")
    allow_mixed_classes: bool = Field(default=True, description="Enable mixed vessel class combinations")
    cost_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    schedule_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    risk_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    utilization_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    demurrage_weight: float = Field(default=0.10, ge=0.0, le=1.0)


@router.post("/optimize-plans")
async def optimize_fleet_plans(request: FleetPlanOptimizationRequest):
    """
    Phase 6: Multi-Voyage Vessel and Fleet Plan Optimization.
    Generates, filters by hard physical constraints, evaluates voyage economics and risk,
    and returns soft multi-criteria ranked charter plans.
    """
    from src.optimization.multi_voyage_optimizer import MultiVoyageOptimizer, FleetOptimizationRequest
    from src.optimization.vessel_scoring import OptimizationWeights

    weights = OptimizationWeights(
        cost_weight=request.cost_weight,
        schedule_weight=request.schedule_weight,
        risk_weight=request.risk_weight,
        utilization_weight=request.utilization_weight,
        demurrage_weight=request.demurrage_weight,
    )

    optimizer = MultiVoyageOptimizer()
    opt_request = FleetOptimizationRequest(
        cargo_quantity_t=request.cargo_quantity_t,
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        cargo_type=request.cargo_type,
        route_distance_nm=request.route_distance_nm,
        freight_rate_usd=request.freight_rate_usd,
        delivery_deadline_days=request.delivery_deadline_days,
        max_voyages=request.max_voyages,
        allow_mixed_classes=request.allow_mixed_classes,
        weights=weights,
    )

    ranked_plans = optimizer.optimize(opt_request)
    return [plan.to_dict() for plan in ranked_plans]

"""
Charter-AI — Voyage Economics Endpoints (Phase 5).

Provides delivered cost calculations from the charterer/cargo-owner perspective,
dynamic congestion-driven demurrage exposure, itemized cost breakdowns,
and multi-factor sensitivity analysis.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.serializers import (
    CostBreakdownResponse,
    VoyageEconomicsResponse,
    DeliveredCostResponse,
)
from src.economics.voyage_calculator import VoyageCalculator
from src.economics.voyage_cost import (
    VoyageCostInputs,
    VoyageCostBreakdown,
    calculate_voyage_cost,
    perform_sensitivity_analysis,
    run_full_sensitivity_matrix,
)
from src.services.voyage_economics_service import VoyageEconomicsService
from src.data.vessel_repository import get_vessel_class_spec

router = APIRouter(prefix="/economics", tags=["Economics"])

_calculator = VoyageCalculator()
_service = VoyageEconomicsService()


class VoyageEconomicsRequest(BaseModel):
    """Legacy request body for voyage economics calculation."""
    origin_port_id: str
    destination_port_id: str
    vessel_class: str
    cargo_tonnage: int = Field(..., gt=0)
    sailing_distance_nm: float = Field(..., gt=0)
    freight_rate_usd_per_day: float = Field(..., gt=0)
    vlsfo_price_usd_per_tonne: float = Field(default=550.0, gt=0)
    predicted_idle_days: float = Field(default=2.0, ge=0)


class DeliveredCostRequest(BaseModel):
    """Phase 5 request body from the charterer/cargo-owner perspective."""
    cargo_quantity_t: float = Field(..., gt=0, description="Cargo quantity in metric tonnes (MT)")
    freight_rate_usd: float = Field(..., gt=0, description="Freight rate (USD/t or USD/day)")
    origin_port_id: str = Field(default="INA_TAB", description="Origin port ID")
    destination_port_id: str = Field(default="IND_PAR", description="Destination port ID")
    vessel_class: str = Field(default="Panamax", description="Vessel class")
    route_distance_nm: float = Field(default=3800.0, gt=0, description="Laden distance in nautical miles")
    vessel_speed_knots: Optional[float] = Field(default=None, gt=0, description="Laden vessel speed in knots")
    fuel_price_usd_per_t: Optional[float] = Field(default=None, gt=0, description="VLSFO fuel price per tonne")
    expected_waiting_days: Optional[float] = Field(default=None, ge=0, description="Anchorage waiting days from Congestion Predictor")
    positioning_distance_nm: float = Field(default=0.0, ge=0, description="Ballast positioning distance")
    canal_charges_usd: float = Field(default=0.0, ge=0, description="Canal / strait transit charges")
    agency_fees_usd: Optional[float] = Field(default=None, ge=0, description="Port agency fees")
    contingency_cost_usd: float = Field(default=0.0, ge=0, description="Risk and weather contingency buffer")
    delivery_deadline_days: Optional[float] = Field(default=None, gt=0, description="Delivery schedule deadline in days")
    charter_type: str = Field(default="voyage", description="'voyage' (USD/t) or 'time_charter' (USD/day)")


class SensitivityRequest(BaseModel):
    """Request body for sensitivity analysis."""
    base_inputs: DeliveredCostRequest
    parameter: Optional[str] = Field(default=None, description="bunker_price, freight_rate, congestion, or cargo_quantity")
    values: Optional[List[float]] = Field(default=None, description="Custom values for parameter sweep")


@router.post("", response_model=VoyageEconomicsResponse)
async def calculate_voyage_economics(request: VoyageEconomicsRequest):
    """Legacy endpoint: Calculate full voyage economics with VoyageCalculator."""
    result = _calculator.calculate(
        vessel_class=request.vessel_class,
        cargo_tonnage=request.cargo_tonnage,
        sailing_distance_nm=request.sailing_distance_nm,
        freight_rate_usd_per_day=request.freight_rate_usd_per_day,
        vlsfo_price_usd_per_tonne=request.vlsfo_price_usd_per_tonne,
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        predicted_idle_days=request.predicted_idle_days,
    )

    return VoyageEconomicsResponse(
        total_voyage_cost_usd=result.total_voyage_cost_usd,
        cost_per_tonne_usd=result.cost_per_tonne_usd,
        cargo_tonnage=result.cargo_tonnage,
        vessel_class=result.vessel_class,
        sailing_days=result.sailing_days,
        total_voyage_days=result.total_voyage_days,
        breakdown=CostBreakdownResponse(
            freight_cost_usd=result.breakdown.freight_cost_usd,
            bunker_cost_usd=result.breakdown.bunker_cost_usd,
            load_port_charges_usd=result.breakdown.load_port_charges_usd,
            discharge_port_charges_usd=result.breakdown.discharge_port_charges_usd,
            insurance_usd=result.breakdown.insurance_usd,
            expected_demurrage_usd=result.breakdown.expected_demurrage_usd,
            miscellaneous_usd=result.breakdown.miscellaneous_usd,
        ),
    )


@router.post("/calculate", response_model=DeliveredCostResponse)
async def calculate_delivered_cost(request: DeliveredCostRequest):
    """
    Phase 5: Calculate realistic delivered voyage economics from the charterer/cargo-owner perspective.
    Dynamically integrates port handling rates, predicted congestion wait time, laytime demurrage,
    positioning, canal/agency costs, and delivery probability.
    """
    breakdown = _service.calculate_delivered_cost(
        cargo_quantity_t=request.cargo_quantity_t,
        freight_rate_usd=request.freight_rate_usd,
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        vessel_class=request.vessel_class,
        route_distance_nm=request.route_distance_nm,
        vessel_speed_knots=request.vessel_speed_knots,
        fuel_price_usd_per_t=request.fuel_price_usd_per_t,
        expected_waiting_days=request.expected_waiting_days,
        positioning_distance_nm=request.positioning_distance_nm,
        canal_charges_usd=request.canal_charges_usd,
        agency_fees_usd=request.agency_fees_usd,
        contingency_cost_usd=request.contingency_cost_usd,
        delivery_deadline_days=request.delivery_deadline_days,
        charter_type=request.charter_type,
        as_dict=False,
    )

    details = {
        "sailing_days": breakdown.sailing_days,
        "loading_days": breakdown.loading_days,
        "discharge_days": breakdown.discharge_days,
        "waiting_days": breakdown.waiting_days,
        "total_port_days": breakdown.total_port_days,
        "laytime_allowed_days": breakdown.laytime_allowed_days,
        "excess_time_days": breakdown.excess_time_days,
        "despatch_savings": breakdown.despatch_savings,
        "canal_charges": breakdown.canal_charges,
        "agency_fees": breakdown.agency_fees,
        "contingency_cost": breakdown.contingency_cost,
    }

    return DeliveredCostResponse(
        freight_cost=breakdown.freight_cost,
        bunker_cost=breakdown.bunker_cost,
        port_cost=breakdown.port_cost,
        waiting_cost=breakdown.waiting_cost,
        demurrage_exposure=breakdown.demurrage_exposure,
        positioning_cost=breakdown.positioning_cost,
        miscellaneous_cost=breakdown.miscellaneous_cost,
        total_cost=breakdown.total_cost,
        cost_per_tonne=breakdown.cost_per_tonne,
        voyage_days=breakdown.voyage_days,
        delivery_probability=breakdown.delivery_probability,
        details=details,
    )


@router.post("/sensitivity")
async def calculate_sensitivity(request: SensitivityRequest) -> Dict[str, Any]:
    """
    Phase 5: Run sensitivity analysis across bunker price, freight rate, congestion, and cargo quantity.
    """
    # Build inputs using the service method
    res = _service.calculate_delivered_cost(
        cargo_quantity_t=request.base_inputs.cargo_quantity_t,
        freight_rate_usd=request.base_inputs.freight_rate_usd,
        origin_port_id=request.base_inputs.origin_port_id,
        destination_port_id=request.base_inputs.destination_port_id,
        vessel_class=request.base_inputs.vessel_class,
        route_distance_nm=request.base_inputs.route_distance_nm,
        vessel_speed_knots=request.base_inputs.vessel_speed_knots,
        fuel_price_usd_per_t=request.base_inputs.fuel_price_usd_per_t,
        expected_waiting_days=request.base_inputs.expected_waiting_days,
        positioning_distance_nm=request.base_inputs.positioning_distance_nm,
        canal_charges_usd=request.base_inputs.canal_charges_usd,
        agency_fees_usd=request.base_inputs.agency_fees_usd,
        contingency_cost_usd=request.base_inputs.contingency_cost_usd,
        delivery_deadline_days=request.base_inputs.delivery_deadline_days,
        charter_type=request.base_inputs.charter_type,
    )

    # Reconstruct VoyageCostInputs from breakdown or inputs using central vessel repository
    class_spec = get_vessel_class_spec(request.base_inputs.vessel_class)
    def_speed, def_fuel, def_hire = (
        class_spec.service_speed_knots,
        class_spec.fuel_consumption_tpd,
        class_spec.daily_hire_usd,
    )
    speed = request.base_inputs.vessel_speed_knots or def_speed
    fuel_price = request.base_inputs.fuel_price_usd_per_t or 550.0

    inputs = VoyageCostInputs(
        cargo_quantity_t=request.base_inputs.cargo_quantity_t,
        freight_rate_usd=request.base_inputs.freight_rate_usd,
        vessel_speed_knots=speed,
        vessel_daily_fuel_consumption_tpd=def_fuel,
        vessel_daily_hire_cost_usd=def_hire,
        route_distance_nm=request.base_inputs.route_distance_nm,
        positioning_distance_nm=request.base_inputs.positioning_distance_nm,
        fuel_price_usd_per_t=fuel_price,
        load_port_cost_usd=res.port_cost * 0.5,
        discharge_port_cost_usd=res.port_cost * 0.5,
        port_handling_rate_tpd=35000.0,
        discharge_port_handling_rate_tpd=30000.0,
        expected_waiting_days=res.waiting_days,
        canal_charges_usd=request.base_inputs.canal_charges_usd,
        agency_fees_usd=res.agency_fees,
        contingency_cost_usd=request.base_inputs.contingency_cost_usd,
        delivery_deadline_days=request.base_inputs.delivery_deadline_days,
        charter_type=request.base_inputs.charter_type,
    )

    if request.parameter:
        sweep = perform_sensitivity_analysis(inputs, request.parameter, request.values)
        return {"parameter": request.parameter, "scenarios": sweep}

    matrix = run_full_sensitivity_matrix(inputs)
    return {"matrix": matrix}

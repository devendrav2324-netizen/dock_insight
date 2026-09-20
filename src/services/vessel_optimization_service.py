"""
DockInsights — Vessel & Fleet Optimization Service (Phase 6).

Coordinates deterministic vessel selection, constraint validation,
and multi-voyage fleet charter plan optimization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.optimization.vessel_selector import (
    VesselOptimizer,
    VesselSpecs,
    PortConstraints,
    VesselSelectionResult,
    VesselSelectionOptimizer,
)
from src.optimization.multi_voyage_optimizer import (
    MultiVoyageOptimizer,
    FleetOptimizationRequest,
)
from src.optimization.voyage_plan import CharterPlan
from src.optimization.vessel_scoring import OptimizationWeights
from src.services.congestion_service import CongestionService
from src.services.risk_service import RiskService


class VesselOptimizationService:
    """
    Service for selecting, evaluating, and optimizing vessels and multi-voyage fleet plans.
    """

    def __init__(
        self,
        congestion_service: Optional[CongestionService] = None,
        risk_service: Optional[RiskService] = None,
    ):
        self.congestion_service = congestion_service or CongestionService()
        self.risk_service = risk_service or RiskService(congestion_service=self.congestion_service)
        self.optimizer = VesselOptimizer()
        self.multi_voyage_optimizer = MultiVoyageOptimizer(
            congestion_service=self.congestion_service,
            risk_service=self.risk_service,
        )

    def optimize_vessel(
        self,
        cargo_type: str,
        cargo_quantity_t: float,
        origin_constraints: PortConstraints,
        dest_constraints: PortConstraints,
        expected_loading_date: datetime,
        required_delivery_date: datetime,
        vessel_specs_db: List[VesselSpecs],
    ) -> VesselSelectionResult:
        """
        Evaluate candidate vessels for a single voyage (backward compatible).
        """
        if not vessel_specs_db:
            raise ValueError("Vessel specifications database is required for optimization.")

        return self.optimizer.optimize(
            cargo_type=cargo_type,
            cargo_quantity=int(cargo_quantity_t),
            origin_port=origin_constraints,
            destination_port=dest_constraints,
            expected_loading_date=expected_loading_date,
            required_delivery_date=required_delivery_date,
            vessel_specs=vessel_specs_db,
        )

    def optimize_fleet_plans(
        self,
        cargo_quantity_t: float,
        origin_port_id: str,
        destination_port_id: str,
        cargo_type: str = "Coal",
        route_distance_nm: float = 4200.0,
        freight_rate_usd: float = 22.0,
        expected_loading_date: Optional[datetime] = None,
        required_delivery_date: Optional[datetime] = None,
        delivery_deadline_days: Optional[float] = None,
        fuel_price_usd_per_t: float = 550.0,
        max_voyages: int = 5,
        allow_mixed_classes: bool = True,
        include_parallel_options: bool = True,
        weights: Optional[OptimizationWeights] = None,
    ) -> List[CharterPlan]:
        """
        Evaluates complete charter plans (e.g. 1xCapesize vs 2xPanamax vs 2xSupramax vs 3xHandysize),
        filtering hard physical constraints and ranking soft multi-objective scores.
        """
        request = FleetOptimizationRequest(
            cargo_quantity_t=cargo_quantity_t,
            origin_port_id=origin_port_id,
            destination_port_id=destination_port_id,
            cargo_type=cargo_type,
            route_distance_nm=route_distance_nm,
            freight_rate_usd=freight_rate_usd,
            expected_loading_date=expected_loading_date,
            required_delivery_date=required_delivery_date,
            delivery_deadline_days=delivery_deadline_days,
            fuel_price_usd_per_t=fuel_price_usd_per_t,
            max_voyages=max_voyages,
            allow_mixed_classes=allow_mixed_classes,
            include_parallel_options=include_parallel_options,
            weights=weights,
        )
        return self.multi_voyage_optimizer.optimize(request)

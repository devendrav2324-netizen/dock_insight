"""
Charter-AI — Multi-Voyage Vessel & Fleet Optimization Engine (Phase 6).

Compares complete charter plans (e.g. 1xCapesize vs 2xPanamax vs 2xSupramax vs 3xHandysize)
instead of simplistic cargo-to-vessel mappings.

Pipeline Architecture:
Cargo Request
    ↓
Candidate Generator
    ↓
Hard Constraint Filter (draft, LOA, beam, parcel limits, terminal specs, deadlines)
    ↓
Voyage Economics Engine (Phase 5 delivered costs, bunker, port charges, demurrage)
    ↓
Congestion & Risk Evaluation (Phase 4 predicted wait time, delay probability, risk score)
    ↓
Schedule Evaluation (sequential vs parallel execution duration)
    ↓
Multi-Criteria Soft Objective Scoring (cost, schedule, risk, utilization, demurrage)
    ↓
Ranked Charter Plans
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.optimization.candidate_generator import CandidateGenerator, CandidatePlanSpec, DEFAULT_VESSEL_CLASSES
from src.optimization.constraint_engine import ConstraintEngine
from src.optimization.voyage_plan import CharterPlan, VoyageLeg
from src.optimization.vessel_scoring import OptimizationWeights, VesselScoringEngine
from src.economics.voyage_cost import (
    VoyageCostInputs,
    calculate_voyage_cost,
    calculate_sailing_days,
    calculate_port_operational_time,
    calculate_delivery_probability,
)
from src.data.mock_db import get_mock_port_info, get_mock_vessel_db
from src.data.vessel_repository import get_vessel_class_spec
from src.economics.cost_models import get_port_costs
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FleetOptimizationRequest:
    """Request inputs for multi-voyage fleet optimization."""
    cargo_quantity_t: float
    origin_port_id: str
    destination_port_id: str
    cargo_type: str = "Coal"
    route_distance_nm: float = 4200.0  # Actual route distance
    freight_rate_usd: float = 22.0  # Freight benchmark $/t
    expected_loading_date: Optional[datetime] = None
    required_delivery_date: Optional[datetime] = None
    delivery_deadline_days: Optional[float] = None
    fuel_price_usd_per_t: float = 550.0
    max_voyages: int = 5
    allow_mixed_classes: bool = True
    include_parallel_options: bool = True
    weights: Optional[OptimizationWeights] = None
    origin_port_info: Optional[Any] = None
    destination_port_info: Optional[Any] = None


class MultiVoyageOptimizer:
    """
    Evaluates, optimizes, and ranks multi-voyage charter plans.
    """

    def __init__(
        self,
        congestion_service: Optional[Any] = None,
        risk_service: Optional[Any] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        constraint_engine: Optional[ConstraintEngine] = None,
        scoring_engine: Optional[VesselScoringEngine] = None,
    ):
        self.settings = get_settings()
        if congestion_service is None:
            from src.services.congestion_service import CongestionService
            self.congestion_service = CongestionService()
        else:
            self.congestion_service = congestion_service

        if risk_service is None:
            from src.services.risk_service import RiskService
            self.risk_service = RiskService(congestion_service=self.congestion_service)
        else:
            self.risk_service = risk_service

        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.constraint_engine = constraint_engine or ConstraintEngine()
        self.scoring_engine = scoring_engine or VesselScoringEngine()

    def optimize(self, request: FleetOptimizationRequest) -> List[CharterPlan]:
        """
        Executes full multi-voyage plan optimization.
        Returns a ranked list of CharterPlan objects.
        """
        logger.info(
            "Starting fleet optimization: %s MT %s from %s to %s (dist: %s nm)",
            f"{request.cargo_quantity_t:,.0f}",
            request.cargo_type,
            request.origin_port_id,
            request.destination_port_id,
            request.route_distance_nm,
        )

        # ---------------------------------------------------------
        # 1. Resolve Port & Infrastructure Context
        # ---------------------------------------------------------
        orig_info = request.origin_port_info or get_mock_port_info(request.origin_port_id)
        dest_info = request.destination_port_info or get_mock_port_info(request.destination_port_id)
        load_rate = getattr(orig_info, "cargo_handling_rate_mt_day", 35000.0) or 35000.0
        disch_rate = getattr(dest_info, "cargo_handling_rate_mt_day", 30000.0) or 30000.0

        orig_costs = get_port_costs(request.origin_port_id)
        dest_costs = get_port_costs(request.destination_port_id)
        load_port_fee = (orig_costs.pilotage_usd + orig_costs.towage_usd + 35000.0) if orig_costs else 50000.0
        disch_port_fee = (dest_costs.pilotage_usd + dest_costs.towage_usd + 35000.0) if dest_costs else 50000.0

        # Timeline
        if request.delivery_deadline_days is None:
            if request.required_delivery_date and request.expected_loading_date:
                allowed_days = (request.required_delivery_date - request.expected_loading_date).days
                request.delivery_deadline_days = max(5.0, float(allowed_days))
            else:
                request.delivery_deadline_days = 35.0  # Generous operational window

        # ---------------------------------------------------------
        # 2. Generate Candidate Plans
        # ---------------------------------------------------------
        candidate_specs = self.candidate_generator.generate_candidate_plans(
            total_cargo_t=request.cargo_quantity_t,
            max_voyages=request.max_voyages,
            allow_mixed_classes=request.allow_mixed_classes,
            include_parallel_options=request.include_parallel_options,
        )

        evaluated_plans: List[CharterPlan] = []

        # ---------------------------------------------------------
        # 3. Evaluate Each Candidate Plan
        # ---------------------------------------------------------
        for spec in candidate_specs:
            plan = self._evaluate_single_plan(
                spec=spec,
                request=request,
                orig_info=orig_info,
                dest_info=dest_info,
                load_rate=load_rate,
                disch_rate=disch_rate,
                load_port_fee=load_port_fee,
                disch_port_fee=disch_port_fee,
            )
            evaluated_plans.append(plan)

        # ---------------------------------------------------------
        # 4. Soft Multi-Criteria Scoring & Ranking
        # ---------------------------------------------------------
        if request.weights:
            self.scoring_engine.weights = request.weights

        ranked_plans = self.scoring_engine.score_plans(evaluated_plans)
        logger.info("Fleet optimization complete: %d candidate plans evaluated", len(ranked_plans))
        return ranked_plans

    def _evaluate_single_plan(
        self,
        spec: CandidatePlanSpec,
        request: FleetOptimizationRequest,
        orig_info: Any,
        dest_info: Any,
        load_rate: float,
        disch_rate: float,
        load_port_fee: float,
        disch_port_fee: float,
    ) -> CharterPlan:
        """
        Calculates complete voyage economics, operational durations, congestion waiting,
        demurrage exposure, and hard physical constraints for one plan specification.
        """
        plan = CharterPlan(
            plan_id=spec.plan_id,
            vessel_classes=spec.vessel_classes,
            number_of_vessels=spec.number_of_vessels,
            number_of_voyages=spec.number_of_voyages,
            execution_mode=spec.execution_mode,
            cargo_per_voyage=spec.cargo_per_voyage,
            total_cargo_t=sum(spec.cargo_per_voyage),
        )

        legs: List[VoyageLeg] = []
        tot_freight = 0.0
        tot_bunker = 0.0
        tot_port = 0.0
        tot_waiting = 0.0
        tot_demurrage = 0.0
        tot_positioning = 0.0
        tot_misc = 0.0
        tot_cost = 0.0
        util_weights = []

        # Predict congestion at destination for dominant vessel class
        primary_class = spec.vessel_classes[0]
        try:
            congestion_pred = self.congestion_service.predict_congestion(
                port_id=request.destination_port_id,
                vessel_class=primary_class,
                cargo_quantity=spec.cargo_per_voyage[0],
            )
            dest_wait_days = float(congestion_pred["expected_wait_days"])
            delay_prob = float(congestion_pred.get("delay_probability", 0.35))
        except Exception:
            dest_wait_days = 2.5
            delay_prob = 0.30

        plan.demurrage_probability = delay_prob

        # Calculate each leg
        leg_durations: List[float] = []
        for i, parcel_mt in enumerate(spec.cargo_per_voyage):
            v_class = spec.vessel_classes[i % len(spec.vessel_classes)]
            class_spec = get_vessel_class_spec(v_class)
            speed, fuel_tpd, hire_usd, dem_rate, typ_dwt = (
                class_spec.service_speed_knots,
                class_spec.fuel_consumption_tpd,
                class_spec.daily_hire_usd,
                class_spec.demurrage_rate_usd,
                float(class_spec.typical_dwt),
            )

            # Economics engine inputs
            econ_inputs = VoyageCostInputs(
                cargo_quantity_t=parcel_mt,
                freight_rate_usd=request.freight_rate_usd,
                vessel_speed_knots=speed,
                vessel_daily_fuel_consumption_tpd=fuel_tpd,
                vessel_daily_hire_cost_usd=hire_usd,
                route_distance_nm=request.route_distance_nm,
                fuel_price_usd_per_t=request.fuel_price_usd_per_t,
                load_port_cost_usd=load_port_fee,
                discharge_port_cost_usd=disch_port_fee,
                port_handling_rate_tpd=load_rate,
                discharge_port_handling_rate_tpd=disch_rate,
                expected_waiting_days=dest_wait_days,
                daily_demurrage_rate_usd=dem_rate,
                agency_fees_usd=8000.0,
                delivery_deadline_days=request.delivery_deadline_days,
            )
            breakdown = calculate_voyage_cost(econ_inputs)

            leg = VoyageLeg(
                leg_id=i + 1,
                vessel_class=v_class,
                cargo_quantity_t=parcel_mt,
                vessel_dwt=typ_dwt,
                utilization=round(parcel_mt / typ_dwt, 4),
                origin_port_id=request.origin_port_id,
                destination_port_id=request.destination_port_id,
                route_distance_nm=request.route_distance_nm,
                sailing_days=breakdown.sailing_days,
                loading_days=breakdown.loading_days,
                discharge_days=breakdown.discharge_days,
                waiting_days=breakdown.waiting_days,
                total_leg_days=breakdown.voyage_days,
                freight_cost=breakdown.freight_cost,
                bunker_cost=breakdown.bunker_cost,
                port_cost=breakdown.port_cost,
                waiting_cost=breakdown.waiting_cost,
                demurrage_exposure=breakdown.demurrage_exposure,
                positioning_cost=breakdown.positioning_cost,
                miscellaneous_cost=breakdown.miscellaneous_cost,
                total_cost=breakdown.total_cost,
                cost_per_tonne=breakdown.cost_per_tonne,
            )
            legs.append(leg)
            leg_durations.append(breakdown.voyage_days)

            tot_freight += breakdown.freight_cost
            tot_bunker += breakdown.bunker_cost
            tot_port += breakdown.port_cost
            tot_waiting += breakdown.waiting_cost
            tot_demurrage += breakdown.demurrage_exposure
            tot_positioning += breakdown.positioning_cost
            tot_misc += breakdown.miscellaneous_cost
            tot_cost += breakdown.total_cost
            util_weights.append(parcel_mt / typ_dwt)

        plan.legs = legs
        plan.total_freight_cost = tot_freight
        plan.bunker_cost = tot_bunker
        plan.port_cost = tot_port
        plan.waiting_cost = tot_waiting
        plan.expected_demurrage = tot_demurrage
        plan.positioning_cost = tot_positioning
        plan.miscellaneous_cost = tot_misc
        plan.total_cost = tot_cost
        plan.cost_per_tonne = tot_cost / plan.total_cargo_t if plan.total_cargo_t > 0 else 0.0
        plan.utilization = sum(util_weights) / len(util_weights) if util_weights else 0.0

        # ---------------------------------------------------------
        # 4. Total Schedule Duration
        # ---------------------------------------------------------
        if spec.execution_mode == "sequential":
            # 1 vessel doing N sequential voyages
            # Each subsequent voyage requires ballast return leg: sailing_days
            ballast_days = sum(leg.sailing_days for leg in legs[:-1])
            plan.total_duration = sum(leg_durations) + ballast_days
        else:
            # N vessels operating in parallel
            # Staggered by berth loading time
            stagger = legs[0].loading_days * (len(legs) - 1) * 0.5
            plan.total_duration = max(leg_durations) + stagger

        # ---------------------------------------------------------
        # 5. Dynamic Risk Score (8-Category Risk Engine) & Monte Carlo Simulation
        # ---------------------------------------------------------
        risk_res = self.risk_service.evaluate_risk({
            "vessel_class": primary_class,
            "port_data": {
                "port_id": request.destination_port_id,
                "expected_wait_days": dest_wait_days,
                "vessels_waiting": int(round(dest_wait_days * 2.5)),
            },
            "schedule_data": {
                "total_duration": plan.total_duration,
                "delivery_deadline_days": request.delivery_deadline_days,
            },
            "demurrage_data": {
                "expected_demurrage": plan.expected_demurrage,
                "freight_cost": plan.total_freight_cost,
            },
            "cargo_quantity": request.cargo_quantity_t,
        })
        plan.risk_score = risk_res.composite_score

        # Run vectorized Monte Carlo simulation to establish true probabilistic delivery & demurrage
        primary_spec = get_vessel_class_spec(primary_class)
        try:
            mc_res = self.risk_service.simulate_monte_carlo({
                "cargo_quantity_t": request.cargo_quantity_t,
                "base_freight_rate": plan.cost_per_tonne,
                "expected_wait_days": dest_wait_days,
                "sea_distance_nm": request.route_distance_nm,
                "service_speed_knots": primary_spec.service_speed_knots,
                "fuel_consumption_t_day": primary_spec.fuel_consumption_tpd,
                "port_charges_usd": plan.port_cost,
                "delivery_deadline_days": request.delivery_deadline_days,
            }, n_simulations=1000)
            
            plan.demurrage_probability = mc_res.demurrage_probability
            plan.delivery_probability = max(0.01, 1.0 - mc_res.late_delivery_probability)
            plan.monte_carlo = {
                "expected_cost": round(mc_res.expected_cost, 2),
                "p10_cost": round(mc_res.p10_cost, 2),
                "p50_cost": round(mc_res.p50_cost, 2),
                "p90_cost": round(mc_res.p90_cost, 2),
                "demurrage_probability": round(mc_res.demurrage_probability, 4),
                "late_delivery_probability": round(mc_res.late_delivery_probability, 4),
                "status": "SUCCESS"
            }
        except (NameError, AttributeError, TypeError, KeyError) as e:
            # Do not silently swallow programming errors
            logger.error(f"Programming error in Monte Carlo integration: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.warning(f"Monte Carlo simulation unavailable, falling back: {e}")
            plan.delivery_probability = calculate_delivery_probability(
                total_elapsed_days=plan.total_duration,
                delivery_deadline_days=request.delivery_deadline_days,
            )
            if not hasattr(plan, 'monte_carlo') or plan.monte_carlo is None:
                plan.monte_carlo = {}
            plan.monte_carlo["status"] = "FALLBACK"
            plan.monte_carlo["fallback_reason"] = str(e)

        # ---------------------------------------------------------
        # 6. Hard Constraint Validation
        # ---------------------------------------------------------
        constraint_res = self.constraint_engine.evaluate_plan(
            plan=spec,
            origin_port=orig_info,
            destination_port=dest_info,
            cargo_type=request.cargo_type,
            delivery_deadline_days=request.delivery_deadline_days,
            estimated_duration_days=plan.total_duration,
        )
        plan.feasibility = constraint_res.is_feasible
        plan.failed_constraints = constraint_res.failed_constraints
        plan.reasons = constraint_res.reasons.copy()

        return plan

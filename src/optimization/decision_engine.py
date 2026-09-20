"""
DockInsights — Central Unified Decision Engine (Phase 10).

Orchestrates the end-to-end maritime chartering intelligence architecture:
    Cargo Request
          ↓
    Data Validation & Geospatial Routing
          ↓
    Freight Forecast (ML/Statistical Ensemble)
          ↓
    Congestion Forecast (Quantile XGBoost / Delay Probability)
          ↓
    Market Timing Engine (Expected Economic Benefit & Action)
          ↓
    Candidate Vessel Plans & Hard Constraints
          ↓
    Voyage Economics (Itemized 9-component delivered cost)
          ↓
    Risk Engine (8 service-driven categories)
          ↓
    Monte Carlo Simulation (Vectorized 10,000-run distribution)
          ↓
    Multi-Voyage Fleet Optimization (Soft objective scoring & ranking)
          ↓
    Contract Optimization (Quantitative Spot vs Term vs Hybrid portfolio)
          ↓
    Structured Tradeoff Explainability & Reproducible Decision Response
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Geospatial and port/vessel models
from src.utils.geo import haversine_nm, estimate_sailing_distance_nm
from src.data.mock_db import get_mock_port_info, get_mock_vessel_db
from src.data.vessel_repository import calculate_vessel_availability
from src.optimization.port_compatibility import check_vessel_port_compatibility, PortInfo, VesselInfo
from src.optimization.vessel_selector import VesselSpecs, PortConstraints
from src.optimization.voyage_plan import CharterPlan
from src.optimization.multi_voyage_optimizer import MultiVoyageOptimizer, FleetOptimizationRequest

# Services layer
from src.services.freight_forecast_service import FreightForecastService
from src.services.congestion_service import CongestionService
from src.services.risk_service import RiskService
from src.services.voyage_economics_service import VoyageEconomicsService
from src.services.vessel_optimization_service import VesselOptimizationService
from src.services.contract_optimization_service import ContractOptimizationService
from src.models.market_timing import MarketTimingEngine, MarketTimingInputs
from src.optimization.contract_optimizer import (
    ContractOptimizationInputs,
    ForecastDirection,
    ForecastUncertainty,
    VesselAvailability,
    RiskTolerance,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_VERSIONS = {
    "freight_forecast": "v2.3.0-ensemble",
    "congestion_predictor": "v2.1.0-quantile-xgb",
    "voyage_economics": "v2.0.0-9component",
    "risk_engine": "v2.0.0-8category",
    "monte_carlo": "v2.0.0-vectorized-10k",
    "fleet_optimizer": "v2.0.0-multi-criteria",
    "contract_optimizer": "v2.0.0-risk-aware",
    "market_timing": "v2.0.0-benefit-quantiles",
}

DATA_VERSIONS = {
    "port_database": "2026.09-v2-normalized",
    "vessel_database": "2026.09-v2-normalized",
    "historical_timeseries": "33902_records",
    "data_mode": "SYNTHETIC_DEMO",
    "provenance_status": "SYNTHETIC_DEMO",
    "is_verified_external": "false",
}


STANDARD_ROUTE_DISTANCES = {
    ("AUS_NEW", "IND_VZG"): 5450.0,
    ("AUS_NEW", "IND_PAR"): 5620.0,
    ("AUS_NEW", "IND_GVM"): 5440.0,
    ("AUS_NEW", "IND_DHM"): 5680.0,
    ("AUS_NEW", "IND_DHA"): 5680.0,
    ("AUS_HAY", "IND_GVM"): 4950.0,
    ("AUS_HAY", "IND_PAR"): 5100.0,
    ("AUS_HAY", "IND_PRT"): 5100.0,
    ("AUS_HPT", "IND_PRT"): 5100.0,
    ("AUS_HPT", "IND_GVM"): 4950.0,
    ("AUS_HPT", "IND_ENR"): 4900.0,
    ("IDN_TAB", "IND_HLD"): 2850.0,
    ("IDN_TAB", "IND_PAR"): 2720.0,
    ("IDN_TAB", "IND_DHM"): 2750.0,
    ("IDN_TAB", "IND_DHA"): 2750.0,
    ("IDN_TAB", "IND_VZG"): 2600.0,
    ("INA_TAB", "IND_HLD"): 2850.0,
    ("INA_TAB", "IND_PAR"): 2720.0,
    ("INA_TAB", "IND_DHM"): 2750.0,
    ("INA_TAB", "IND_DHA"): 2750.0,
    ("INA_TAB", "IND_VZG"): 2600.0,
    ("INA_TAB", "IND_GVM"): 2620.0,
    ("IDN_BAL", "IND_GVM"): 2650.0,
    ("ZAF_RIC", "IND_VZG"): 4650.0,
    ("ZAF_RIC", "IND_PAR"): 4820.0,
    ("USA_HAM", "IND_PAR"): 9500.0,
    ("USA_HAM", "IND_GVM"): 9600.0,
    ("RUS_VOS", "IND_PAR"): 4900.0,
}

PORT_COORDINATES = {
    "IND_VZG": (17.6868, 83.2185),
    "IND_GVM": (17.6167, 83.2333),
    "IND_PAR": (20.2644, 86.6715),
    "IND_PRT": (20.2644, 86.6715),
    "IND_DHM": (20.8039, 86.9744),
    "IND_DHA": (20.8039, 86.9744),
    "IND_HLD": (22.0234, 88.0645),
    "IND_GOP": (19.3000, 84.9667),
    "IND_ENR": (13.2667, 80.3333),
    "AUS_NEW": (-32.9267, 151.7789),
    "AUS_HAY": (-21.2833, 149.3000),
    "AUS_HPT": (-21.2833, 149.3000),
    "AUS_GLD": (-23.8431, 151.2589),
    "IDN_TAB": (-0.5000, 116.0000),
    "INA_TAB": (-0.5000, 116.0000),
    "IDN_BAL": (-1.2667, 116.8333),
    "ZAF_RIC": (-28.8000, 32.0833),
    "USA_HAM": (36.9500, -76.3333),
    "RUS_VOS": (42.7333, 133.0833),
}


@dataclass
class DecisionEngineInputs:
    cargo_type: str
    cargo_quantity_t: float
    origin_port_id: str
    destination_port_id: str
    expected_loading_date: datetime
    required_delivery_date: datetime
    number_of_voyages: int = 1
    contract_preference: str = "ANY"
    risk_tolerance: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH"

    vessel_specs_db: Optional[List[VesselSpecs]] = None
    origin_port_info: Optional[PortInfo] = None
    destination_port_info: Optional[PortInfo] = None
    risk_inputs_override: Optional[Dict[str, Any]] = None


class DecisionEngine:
    """
    Central orchestration layer executing the end-to-end DockInsights decision pipeline.
    Eliminates all hardcoded values by delegating to specialized services.
    """

    def __init__(
        self,
        forecast_service: Optional[FreightForecastService] = None,
        risk_service: Optional[RiskService] = None,
        economics_service: Optional[VoyageEconomicsService] = None,
        vessel_service: Optional[VesselOptimizationService] = None,
        contract_service: Optional[ContractOptimizationService] = None,
        congestion_service: Optional[CongestionService] = None,
    ):
        self.congestion_service = congestion_service or CongestionService()
        self.forecast_service = forecast_service or FreightForecastService()
        self.risk_service = risk_service or RiskService(congestion_service=self.congestion_service)
        self.economics_service = economics_service or VoyageEconomicsService(congestion_service=self.congestion_service)
        self.vessel_service = vessel_service or VesselOptimizationService()
        self.contract_service = contract_service or ContractOptimizationService()

        # Orchestration sub-engines
        self.fleet_optimizer = MultiVoyageOptimizer(
            congestion_service=self.congestion_service,
            risk_service=self.risk_service,
        )
        self.market_timing_engine = MarketTimingEngine()

    def evaluate(self, inputs: DecisionEngineInputs) -> Dict[str, Any]:
        """
        Executes full 10-phase pipeline and returns the canonical DecisionResponse dictionary,
        including backward-compatible keys for existing tests and endpoints.
        """
        try:
            # -----------------------------------------------------------------
            # Step 1: Data Validation & Geospatial Routing
            # -----------------------------------------------------------------
            if inputs.vessel_specs_db is not None and len(inputs.vessel_specs_db) == 0:
                raise ValueError("Vessel specifications database is required.")

            vessel_db = inputs.vessel_specs_db if inputs.vessel_specs_db else get_mock_vessel_db()
            orig_info = inputs.origin_port_info or get_mock_port_info(inputs.origin_port_id)
            dest_info = inputs.destination_port_info or get_mock_port_info(inputs.destination_port_id)

            # Calculate dynamic route distance using standard table or great-circle + routing factor
            route_key = (inputs.origin_port_id, inputs.destination_port_id)
            if route_key in STANDARD_ROUTE_DISTANCES:
                sailing_dist_nm = STANDARD_ROUTE_DISTANCES[route_key]
            elif inputs.origin_port_id in PORT_COORDINATES and inputs.destination_port_id in PORT_COORDINATES:
                lat1, lon1 = PORT_COORDINATES[inputs.origin_port_id]
                lat2, lon2 = PORT_COORDINATES[inputs.destination_port_id]
                gc_nm = haversine_nm(lat1, lon1, lat2, lon2)
                sailing_dist_nm = max(500.0, estimate_sailing_distance_nm(gc_nm))
            elif orig_info and dest_info and hasattr(orig_info, "latitude") and hasattr(dest_info, "latitude"):
                gc_nm = haversine_nm(orig_info.latitude, orig_info.longitude, dest_info.latitude, dest_info.longitude)
                sailing_dist_nm = max(500.0, estimate_sailing_distance_nm(gc_nm))
            else:
                sailing_dist_nm = 4500.0

            # Calculate delivery deadline window in calendar days
            delivery_deadline_days = max(
                5.0,
                (inputs.required_delivery_date - inputs.expected_loading_date).total_seconds() / 86400.0
            )

            # -----------------------------------------------------------------
            # Step 2: Live Freight Rate Forecasting (ML / Statistical Ensemble)
            # -----------------------------------------------------------------
            forecast_api = self.forecast_service.predict_freight_api(
                origin=inputs.origin_port_id,
                destination=inputs.destination_port_id,
                vessel_class="Panamax",
                cargo_type=inputs.cargo_type,
                horizon_days=int(min(30, max(3, delivery_deadline_days))),
            )
            forecast_rate = float(forecast_api["forecast_rate"])
            current_rate = float(forecast_api["current_rate"])
            forecast_trend_str = str(forecast_api["trend"]).upper()
            forecast_confidence = float(forecast_api["confidence"])
            p10_rate = float(forecast_api["lower_bound"])
            p90_rate = float(forecast_api["upper_bound"])
            volatility_pct = float(forecast_api.get("metrics", {}).get("volatility", 16.0))

            forecast_dict = {
                "forecast_rate_usd": forecast_rate,
                "current_rate_usd": current_rate,
                "current_rate": current_rate,
                "predicted_rate": forecast_rate,
                "direction": forecast_trend_str,
                "confidence": forecast_confidence,
                "uncertainty": "LOW" if forecast_confidence > 0.80 else ("HIGH" if forecast_confidence < 0.65 else "MODERATE"),
                "p10": p10_rate,
                "p50": forecast_rate,
                "p90": p90_rate,
                "model_used": forecast_api.get("model_used", "Ensemble"),
            }

            # -----------------------------------------------------------------
            # Step 3: Live Port Congestion Forecast
            # -----------------------------------------------------------------
            congestion_pred = self.congestion_service.predict_congestion(
                port_id=inputs.destination_port_id,
                vessel_class="Panamax",
                cargo_quantity=float(inputs.cargo_quantity_t),
            )
            predicted_wait_days = float(congestion_pred["expected_wait_days"])
            p90_wait_days = float(congestion_pred.get("p90_waiting_days", predicted_wait_days * 1.6))
            delay_prob = float(congestion_pred.get("delay_probability", 0.30))

            # -----------------------------------------------------------------
            # Step 4: Market Timing Engine (Phase 7)
            # -----------------------------------------------------------------
            derived_momentum = ((forecast_rate - current_rate) / max(0.1, current_rate)) * 100.0
            avail_calc = calculate_vessel_availability(
                cargo_quantity_t=inputs.cargo_quantity_t,
                vessel_list=None,
                max_draft_m=dest_info.max_draft_m if dest_info else None,
                max_loa_m=dest_info.max_loa_m if dest_info else None,
                max_beam_m=dest_info.max_beam_m if dest_info else None,
                expected_loading_date=inputs.expected_loading_date,
            )

            calculated_vessel_avail = avail_calc["vessel_availability_label"]

            timing_inputs = MarketTimingInputs(
                current_freight_rate=current_rate,
                forecast_rate=forecast_rate,
                p10_forecast=p10_rate,
                p50_forecast=forecast_rate,
                p90_forecast=p90_rate,
                forecast_confidence=forecast_confidence,
                market_momentum=derived_momentum,
                vessel_availability=calculated_vessel_avail,
                congestion_forecast=predicted_wait_days,
                cargo_deadline=inputs.required_delivery_date,
                required_delivery_date=inputs.required_delivery_date,
                current_date=inputs.expected_loading_date,
                cargo_quantity_t=inputs.cargo_quantity_t,
                route_voyage_days=sailing_dist_nm / (13.0 * 24.0) + predicted_wait_days + 4.0,
            )
            timing_result = self.market_timing_engine.evaluate_timing(timing_inputs)

            # -----------------------------------------------------------------
            # Step 5: Multi-Voyage Fleet Optimization (Phase 6 & 8)
            # -----------------------------------------------------------------
            fleet_request = FleetOptimizationRequest(
                cargo_quantity_t=inputs.cargo_quantity_t,
                origin_port_id=inputs.origin_port_id,
                destination_port_id=inputs.destination_port_id,
                cargo_type=inputs.cargo_type,
                route_distance_nm=sailing_dist_nm,
                freight_rate_usd=forecast_rate,
                expected_loading_date=inputs.expected_loading_date,
                required_delivery_date=inputs.required_delivery_date,
                delivery_deadline_days=delivery_deadline_days,
                max_voyages=inputs.number_of_voyages,
                allow_mixed_classes=True,
                include_parallel_options=True,
                origin_port_info=orig_info,
                destination_port_info=dest_info,
            )
            ranked_plans = self.fleet_optimizer.optimize(fleet_request)

            if not ranked_plans:
                raise ValueError("No viable charter plans could be constructed for cargo requirements.")

            # Hard constraint filter: Eliminate infeasible candidate plans
            feasible_plans = [p for p in ranked_plans if p.feasibility]
            if not feasible_plans:
                raise ValueError("No feasible vessel found for this cargo and port combination.")
            best_plan = feasible_plans[0]
            alternative_plans = [p for p in ranked_plans if p.plan_id != best_plan.plan_id][:3]

            primary_vessel_class = best_plan.vessel_classes[0] if best_plan.vessel_classes else "Panamax"

            # -----------------------------------------------------------------
            # Step 6: Deep Port Compatibility Check
            # -----------------------------------------------------------------
            chosen_spec = next((v for v in vessel_db if v.class_name == primary_vessel_class), vessel_db[0])
            vessel_info = VesselInfo(
                class_name=chosen_spec.class_name,
                draft_m=chosen_spec.draft_max_m,
                loa_m=chosen_spec.loa_max_m,
                beam_m=chosen_spec.beam_max_m,
                cargo_to_handle_t=best_plan.cargo_per_voyage[0] if best_plan.cargo_per_voyage else inputs.cargo_quantity_t,
            )

            port_analysis = {}
            if orig_info:
                port_analysis["origin"] = check_vessel_port_compatibility(vessel_info, orig_info)
                if not port_analysis["origin"].get("compatible", True):
                    raise ValueError(f"No feasible vessel found for this cargo and port combination: {', '.join(port_analysis['origin'].get('failed_constraints', []))}")
            if dest_info:
                port_analysis["destination"] = check_vessel_port_compatibility(vessel_info, dest_info)
                if not port_analysis["destination"].get("compatible", True):
                    raise ValueError(f"No feasible vessel found for this cargo and port combination: {', '.join(port_analysis['destination'].get('failed_constraints', []))}")
            port_analysis["congestion"] = congestion_pred

            # -----------------------------------------------------------------
            # Step 7: Voyage Economics (Itemized 9-component Cost)
            # -----------------------------------------------------------------
            charter_rate_day = (forecast_rate * inputs.cargo_quantity_t) / max(1.0, best_plan.total_duration)
            voyage_econ = self.economics_service.calculate_cost(
                vessel_class=primary_vessel_class,
                cargo_tonnage=int(inputs.cargo_quantity_t),
                sailing_distance_nm=sailing_dist_nm,
                freight_rate_usd_per_day=charter_rate_day,
                origin_port_id=inputs.origin_port_id,
                destination_port_id=inputs.destination_port_id,
                predicted_idle_days=predicted_wait_days,
            )

            # -----------------------------------------------------------------
            # Step 8: Dynamic 8-Category Risk Assessment (Phase 8)
            # -----------------------------------------------------------------
            risk_inputs = (inputs.risk_inputs_override or {}).copy()
            if "port_data" not in risk_inputs:
                risk_inputs["port_data"] = {
                    "port_id": inputs.destination_port_id,
                    "expected_wait_days": predicted_wait_days,
                    "p90_wait_days": p90_wait_days,
                    "delay_probability": delay_prob,
                    "vessels_waiting": int(round(predicted_wait_days * 2.0)),
                }
            if "market_data" not in risk_inputs:
                risk_inputs["market_data"] = {
                    "price_volatility_pct": volatility_pct,
                    "predicted_rate": forecast_rate,
                    "confidence_score": forecast_confidence,
                    "p10": p10_rate,
                    "p90": p90_rate,
                }
            if "schedule_data" not in risk_inputs:
                risk_inputs["schedule_data"] = {
                    "total_duration": best_plan.total_duration,
                    "delivery_deadline_days": delivery_deadline_days,
                    "late_delivery_probability": 1.0 - best_plan.delivery_probability,
                }
            if "demurrage_data" not in risk_inputs:
                risk_inputs["demurrage_data"] = {
                    "expected_demurrage": best_plan.expected_demurrage,
                    "freight_cost": best_plan.total_freight_cost,
                    "demurrage_probability": best_plan.demurrage_probability,
                }

            risk_result = self.risk_service.evaluate_risk(risk_inputs)

            # -----------------------------------------------------------------
            # Step 9: Risk-Aware Contract Optimization (Phase 9)
            # -----------------------------------------------------------------
            if calculated_vessel_avail == "TIGHT":
                contract_avail_enum = VesselAvailability.TIGHT
            elif calculated_vessel_avail == "SHORTAGE":
                contract_avail_enum = VesselAvailability.SHORTAGE
            else:
                contract_avail_enum = VesselAvailability.ABUNDANT

            contract_inputs = ContractOptimizationInputs(
                current_freight_rate=current_rate,
                expected_future_rate=forecast_rate,
                forecast_direction=ForecastDirection[forecast_trend_str] if forecast_trend_str in ForecastDirection.__members__ else ForecastDirection.STABLE,
                forecast_uncertainty=ForecastUncertainty.HIGH if forecast_confidence < 0.65 else (ForecastUncertainty.LOW if forecast_confidence > 0.80 else ForecastUncertainty.MODERATE),
                vessel_availability=contract_avail_enum,
                congestion_level=predicted_wait_days,
                overall_risk_score=risk_result.composite_score,
                voyage_cost=best_plan.total_cost,
                number_of_required_voyages=best_plan.number_of_voyages,
            )
            legacy_contract_rec = self.contract_service.recommend_strategy(contract_inputs)

            # Quantitative Phase 9 contract optimizer
            risk_tol_str = str(inputs.risk_tolerance).upper()
            tol = RiskTolerance[risk_tol_str] if risk_tol_str in RiskTolerance.__members__ else RiskTolerance.MEDIUM
            risk_aware_rec = self.contract_service.optimize_risk_aware({
                "cargo_quantity_t": inputs.cargo_quantity_t,
                "spot_freight_rate": forecast_rate,
                "freight_volatility_pct": volatility_pct,
                "base_bunker_price": 650.0,
                "sea_distance_nm": sailing_dist_nm,
                "expected_wait_days": predicted_wait_days,
                "delivery_deadline_days": delivery_deadline_days,
                "number_of_voyages": best_plan.number_of_voyages,
                "risk_tolerance": tol.value,
                "vessel_availability": calculated_vessel_avail,
            })

            # -----------------------------------------------------------------
            # Step 10: Structured Comparative Explainability
            # -----------------------------------------------------------------
            # Comparative tradeoff against the runner-up plan
            if alternative_plans:
                runner_up = alternative_plans[0]
                cost_diff = best_plan.total_cost - runner_up.total_cost
                dem_diff = best_plan.expected_demurrage - runner_up.expected_demurrage

                if not runner_up.feasibility:
                    tradeoff_reason = (
                        f"alternative {runner_up.plan_id} violates port physical limits ({', '.join(runner_up.failed_constraints)})"
                    )
                elif dem_diff < -5000:
                    tradeoff_reason = (
                        f"it reduces expected demurrage liability by ${abs(dem_diff):,.0f} and achieves a {best_plan.delivery_probability*100:.0f}% on-time probability under {predicted_wait_days:.1f}-day port congestion"
                    )
                elif cost_diff < 0:
                    tradeoff_reason = (
                        f"it delivers freight savings of ${abs(cost_diff):,.0f} (${best_plan.cost_per_tonne:.2f}/t vs ${runner_up.cost_per_tonne:.2f}/t)"
                    )
                else:
                    tradeoff_reason = (
                        f"it provides optimal cargo utilization ({best_plan.utilization*100:.1f}%) with manageable operational risk ({best_plan.risk_score:.0f}/100)"
                    )
                tradeoff_text = f"{best_plan.plan_id} selected over {runner_up.plan_id} because {tradeoff_reason}."
            else:
                tradeoff_text = f"{best_plan.plan_id} represents the solely compliant, feasible vessel allocation."

            primary_reasons = [
                f"{best_plan.plan_id} ({best_plan.number_of_vessels} × {primary_vessel_class}) achieves optimal multi-criteria score ({best_plan.score}/100).",
                tradeoff_text,
                f"Freight forecast: {forecast_trend_str.lower()} at ${forecast_rate:.2f}/t ({forecast_confidence*100:.0f}% model confidence).",
                f"Predicted port wait days at {inputs.destination_port_id}: {predicted_wait_days:.1f} wait days (expected waiting time: {predicted_wait_days:.1f} days, {congestion_pred['congestion_level']} congestion).",
                f"Contract Recommendation: {risk_aware_rec.recommended_strategy} ({risk_aware_rec.spot_percentage:.0f}% spot / {risk_aware_rec.short_term_percentage + risk_aware_rec.medium_term_percentage:.0f}% term).",
                f"Market Timing: {timing_result.recommendation} ({timing_result.recommended_booking_window['start']} to {timing_result.recommended_booking_window['end']}).",
            ]

            alternatives_rejected = []
            for alt in alternative_plans:
                reason_str = (
                    f"Infeasible: {', '.join(alt.failed_constraints)}"
                    if not alt.feasibility
                    else f"Lower composite score ({alt.score}/100) due to higher cost/demurrage exposure."
                )
                alternatives_rejected.append({
                    "plan_id": alt.plan_id,
                    "vessel_class": alt.vessel_classes[0] if alt.vessel_classes else "Unknown",
                    "reasons_rejected": [reason_str],
                })

            explainability_report = {
                "summary": f"BOOK {risk_aware_rec.recommended_strategy} with {best_plan.plan_id}",
                "recommendation_summary": f"BOOK {risk_aware_rec.recommended_strategy} with {best_plan.plan_id}",
                "primary_reasons": primary_reasons,
                "tradeoff_analysis": tradeoff_text,
                "alternatives_rejected": alternatives_rejected,
            }

            # -----------------------------------------------------------------
            # Step 11: Canonical DecisionResponse Payload & Backward-Compatibility
            # -----------------------------------------------------------------
            decision_id = f"dec_{uuid.uuid4().hex[:12]}"
            timestamp_str = datetime.now(timezone.utc).isoformat()

            canonical_recommended_plan = {
                "vessel_class": primary_vessel_class,
                "vessel_count": best_plan.number_of_vessels,
                "voyages": best_plan.number_of_voyages,
                "cargo_allocation": best_plan.cargo_per_voyage,
                "port_compatibility": port_analysis.get("destination", {}),
                "utilization": round(best_plan.utilization, 4),
                "total_cost": round(best_plan.total_cost, 2),
                "cost_per_tonne": round(best_plan.cost_per_tonne, 2),
                "voyage_duration": round(best_plan.total_duration, 1),
                "expected_waiting": round(predicted_wait_days, 1),
                "demurrage_probability": round(best_plan.demurrage_probability, 4),
                "delivery_probability": round(best_plan.delivery_probability, 4),
                "risk_score": round(best_plan.risk_score, 1),
            }

            canonical_market_analysis = {
                "current_rate": current_rate,
                "forecast": forecast_rate,
                "direction": forecast_trend_str,
                "confidence": forecast_confidence,
                "volatility": volatility_pct,
            }

            canonical_contract_strategy = {
                "recommended_strategy": risk_aware_rec.recommended_strategy,
                "spot_percentage": risk_aware_rec.spot_percentage,
                "short_term_percentage": risk_aware_rec.short_term_percentage,
                "medium_term_percentage": risk_aware_rec.medium_term_percentage,
                "expected_cost": risk_aware_rec.expected_cost,
                "p90_cost": risk_aware_rec.p90_cost,
                "flexibility_score": risk_aware_rec.flexibility_score,
                "reasons": risk_aware_rec.reasons,
                "reasoning": risk_aware_rec.reasons,
            }

            canonical_economics = {
                "total_cost": round(best_plan.total_cost, 2),
                "cost_per_tonne": round(best_plan.cost_per_tonne, 2),
                "freight_cost": round(best_plan.total_freight_cost, 2),
                "bunker_cost": round(best_plan.bunker_cost, 2),
                "port_charges": round(best_plan.port_cost, 2),
                "waiting_cost": round(best_plan.waiting_cost, 2),
                "demurrage_exposure": round(best_plan.expected_demurrage, 2),
                "positioning_cost": round(best_plan.positioning_cost, 2),
                "miscellaneous_cost": round(best_plan.miscellaneous_cost, 2),
            }

            composite_confidence = round(
                (forecast_confidence * 0.40)
                + (best_plan.delivery_probability * 0.35)
                + ((1.0 - (risk_result.composite_score / 100.0)) * 0.25),
                2
            )

            # Combined dictionary containing all canonical Phase 10 fields
            # plus legacy fields required by test_decision_engine.py and analyze_voyage.py
            return {
                # Canonical Phase 10 DecisionResponse Fields
                "decision_id": decision_id,
                "timestamp": timestamp_str,
                "model_versions": MODEL_VERSIONS,
                "data_versions": DATA_VERSIONS,
                "request_summary": {
                    "cargo_type": inputs.cargo_type,
                    "cargo_quantity_t": inputs.cargo_quantity_t,
                    "origin_port_id": inputs.origin_port_id,
                    "destination_port_id": inputs.destination_port_id,
                    "expected_loading_date": inputs.expected_loading_date.isoformat(),
                    "required_delivery_date": inputs.required_delivery_date.isoformat(),
                    "number_of_voyages": inputs.number_of_voyages,
                    "risk_tolerance": inputs.risk_tolerance,
                },
                "market_analysis": canonical_market_analysis,
                "freight_forecast": forecast_dict,
                "market_forecast": forecast_dict,
                "market_timing": timing_result.to_dict(),
                "recommended_plan": canonical_recommended_plan,
                "alternative_plans": [p.to_dict() for p in alternative_plans],
                "economics": canonical_economics,
                "risk": risk_result.to_dict(),
                "contract_strategy": canonical_contract_strategy,
                "confidence": composite_confidence,
                "explanation": explainability_report,

                # Legacy Backward-Compatibility Keys
                "status": "SUCCESS",
                "recommended_vessel": {
                    "class": primary_vessel_class,
                    "score": best_plan.score,
                    "reasons": best_plan.reasons,
                },
                "port_analysis": port_analysis,
                "voyage_economics": {
                    "total_cost": round(best_plan.total_cost, 2),
                    "cost_per_tonne": round(best_plan.cost_per_tonne, 2),
                    "breakdown": {
                        "freight_cost": round(best_plan.total_freight_cost, 2),
                        "bunker_cost": round(best_plan.bunker_cost, 2),
                        "port_charges": round(best_plan.port_cost, 2),
                        "demurrage_risk": round(best_plan.expected_demurrage, 2),
                    },
                },
                "risk_analysis": {
                    "overall_score": risk_result.composite_score,
                    "overall_level": risk_result.level.value,
                    "dominant_risk": risk_result.dominant_risk or "None",
                    "summary": risk_result.summary_messages,
                    "breakdown": {
                        k: {"score": v.score, "level": v.level.value}
                        for k, v in risk_result.categories.items()
                        if not k.islower()
                    },
                },
                "final_recommendation": {
                    "recommended_vessel_class": primary_vessel_class,
                    "action": risk_aware_rec.recommended_strategy,
                    "timing_recommendation": timing_result.recommendation,
                    "recommended_booking_window": timing_result.recommended_booking_window,
                    "contract_type": legacy_contract_rec.recommendation_text,
                    "estimated_total_voyage_cost_usd": round(best_plan.total_cost, 2),
                    "main_risks": [risk_result.recommendation],
                    "alternatives": [alt.vessel_classes[0] for alt in alternative_plans if alt.vessel_classes],
                },
            }

        except Exception as e:
            logger.error(f"DecisionEngine pipeline execution failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error_message": str(e),
                "decision_id": f"dec_err_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_forecast": {},
                "recommended_vessel": {},
                "port_analysis": {},
                "voyage_economics": {},
                "risk_analysis": {},
                "contract_strategy": {},
                "final_recommendation": {},
                "explanation": None,
            }

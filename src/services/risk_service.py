"""
DockInsights — Risk Service (Phase 8 Upgrade).

Ties together:
1. MaritimeRiskEngine (8 service-driven risk categories)
2. MonteCarloSimulator (vectorized 10,000-run simulation)
3. ScenarioEngine (BEST_CASE, BASE_CASE, WORST_CASE stress testing)
4. Modular legacy assessors for backward compatibility
"""

from typing import Dict, Any, Optional, Union
from datetime import date
import pandas as pd
import numpy as np

from src.risk.risk_engine import MaritimeRiskEngine, ComprehensiveRiskAssessment, RiskCategoryResult
from src.risk.monte_carlo import (
    MonteCarloSimulator,
    CharterPlanInputs,
    MonteCarloResult,
    MonteCarloSimulationConfig,
)
from src.risk.scenario_engine import ScenarioEngine, ScenarioResult, ScenarioType
from src.risk.aggregator import RiskAggregator, CompositeRiskAssessment
from src.risk.market_risk import MarketRiskAssessor
from src.risk.port_risk import PortRiskAssessor
from src.risk.weather_risk import WeatherRiskAssessor
from src.risk.operational_risk import OperationalRiskAssessor
from src.services.congestion_service import CongestionService
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskService:
    """
    Service for generating comprehensive maritime risk assessments,
    Monte Carlo probabilistic simulations, and deterministic scenario evaluations.
    """

    def __init__(self, congestion_service: Optional[CongestionService] = None):
        self.settings = get_settings()
        self.engine = MaritimeRiskEngine()
        self.monte_carlo = MonteCarloSimulator()
        self.scenario_engine = ScenarioEngine()

        # Legacy assessors for backward compatibility
        self.aggregator = RiskAggregator()
        self.market_assessor = MarketRiskAssessor()
        self.port_assessor = PortRiskAssessor()
        self.weather_assessor = WeatherRiskAssessor()
        self.operational_assessor = OperationalRiskAssessor()
        self.congestion_service = congestion_service or CongestionService()

    def evaluate_risk(self, inputs: Optional[Dict[str, Any]] = None) -> ComprehensiveRiskAssessment:
        """
        Evaluate comprehensive voyage risk across all 8 dimensions based on provided inputs.
        If inputs are empty, generates dynamic defaults from configuration.
        """
        data = (inputs or {}).copy()
        if not data:
            data = self._get_default_risk_inputs()

        # 1. Congestion integration if port_id and vessel_class provided but wait days missing
        port_data = data.get("port_data", {})
        if "expected_wait_days" not in port_data and "port_id" in port_data:
            port_id = port_data.get("port_id", "IND_PAR")
            v_class = data.get("vessel_class", data.get("vessel_data", {}).get("vessel_class", "Panamax"))
            try:
                cong_pred = self.congestion_service.predict_congestion(
                    port_id=port_id,
                    vessel_class=v_class
                )
                port_data["expected_wait_days"] = cong_pred["expected_wait_days"]
                port_data["p90_wait_days"] = cong_pred.get("p90_waiting_days", cong_pred["expected_wait_days"] * 1.6)
                port_data["delay_probability"] = cong_pred.get("delay_probability", 0.25)
                port_data["vessels_waiting"] = int(round(cong_pred["expected_wait_days"] * 2.0))
            except Exception as e:
                logger.warning(f"Congestion service call failed during risk assessment: {e}")
                port_data["expected_wait_days"] = self.settings.default_port_wait_days
                port_data["vessels_waiting"] = 3
            data["port_data"] = port_data

        # 2. Evaluate total risk using 8-category MaritimeRiskEngine
        assessment = self.engine.evaluate_total_risk(data)
        return assessment

    def simulate_monte_carlo(
        self,
        plan_inputs: Union[CharterPlanInputs, Dict[str, Any]],
        n_simulations: int = 10_000,
        seed: int = 42,
        cost_threshold_usd: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Runs a vectorized Monte Carlo simulation with 10,000 iterations.
        """
        if isinstance(plan_inputs, dict):
            # Parse dict into CharterPlanInputs
            plan = CharterPlanInputs(
                cargo_quantity_t=float(plan_inputs.get("cargo_quantity_t", plan_inputs.get("cargo_tonnage", 70000.0))),
                base_freight_rate=float(plan_inputs.get("base_freight_rate", plan_inputs.get("freight_rate", 20.0))),
                freight_rate_is_per_tonne=bool(plan_inputs.get("freight_rate_is_per_tonne", True)),
                freight_volatility_pct=float(plan_inputs.get("freight_volatility_pct", 15.0)),
                freight_rate_p10=plan_inputs.get("freight_rate_p10"),
                freight_rate_p90=plan_inputs.get("freight_rate_p90"),
                base_bunker_price=float(plan_inputs.get("base_bunker_price", 650.0)),
                bunker_volatility_pct=float(plan_inputs.get("bunker_volatility_pct", 12.0)),
                sea_distance_nm=float(plan_inputs.get("sea_distance_nm", plan_inputs.get("sailing_distance_nm", 4500.0))),
                service_speed_knots=float(plan_inputs.get("service_speed_knots", plan_inputs.get("vessel_speed_knots", 12.5))),
                fuel_consumption_t_day=float(plan_inputs.get("fuel_consumption_t_day", 28.0)),
                expected_wait_days=float(plan_inputs.get("expected_wait_days", 2.0)),
                p90_wait_days=plan_inputs.get("p90_wait_days"),
                port_handling_rate_t_day=float(plan_inputs.get("port_handling_rate_t_day", 15000.0)),
                agreed_laytime_days=plan_inputs.get("agreed_laytime_days"),
                demurrage_rate_usd_day=float(plan_inputs.get("demurrage_rate_usd_day", 20000.0)),
                port_charges_usd=float(plan_inputs.get("port_charges_usd", 45000.0)),
                canal_charges_usd=float(plan_inputs.get("canal_charges_usd", 0.0)),
                misc_agency_usd=float(plan_inputs.get("misc_agency_usd", 8000.0)),
                delivery_deadline_days=plan_inputs.get("delivery_deadline_days"),
                vessel_availability_probability=float(plan_inputs.get("vessel_availability_probability", 0.95)),
            )
        else:
            plan = plan_inputs

        return self.monte_carlo.run_simulation(
            plan=plan,
            n_simulations=n_simulations,
            seed=seed,
            cost_threshold_usd=cost_threshold_usd,
        )

    def evaluate_scenarios(
        self,
        plan_inputs: Union[CharterPlanInputs, Dict[str, Any]],
        mc_result: Optional[MonteCarloResult] = None,
    ) -> Dict[str, ScenarioResult]:
        """
        Evaluates BEST_CASE, BASE_CASE, and WORST_CASE scenarios.
        """
        if isinstance(plan_inputs, dict):
            # Parse dict into CharterPlanInputs
            plan = CharterPlanInputs(
                cargo_quantity_t=float(plan_inputs.get("cargo_quantity_t", plan_inputs.get("cargo_tonnage", 70000.0))),
                base_freight_rate=float(plan_inputs.get("base_freight_rate", plan_inputs.get("freight_rate", 20.0))),
                freight_rate_is_per_tonne=bool(plan_inputs.get("freight_rate_is_per_tonne", True)),
                freight_volatility_pct=float(plan_inputs.get("freight_volatility_pct", 15.0)),
                freight_rate_p10=plan_inputs.get("freight_rate_p10"),
                freight_rate_p90=plan_inputs.get("freight_rate_p90"),
                base_bunker_price=float(plan_inputs.get("base_bunker_price", 650.0)),
                bunker_volatility_pct=float(plan_inputs.get("bunker_volatility_pct", 12.0)),
                sea_distance_nm=float(plan_inputs.get("sea_distance_nm", plan_inputs.get("sailing_distance_nm", 4500.0))),
                service_speed_knots=float(plan_inputs.get("service_speed_knots", plan_inputs.get("vessel_speed_knots", 12.5))),
                fuel_consumption_t_day=float(plan_inputs.get("fuel_consumption_t_day", 28.0)),
                expected_wait_days=float(plan_inputs.get("expected_wait_days", 2.0)),
                p90_wait_days=plan_inputs.get("p90_wait_days"),
                port_handling_rate_t_day=float(plan_inputs.get("port_handling_rate_t_day", 15000.0)),
                agreed_laytime_days=plan_inputs.get("agreed_laytime_days"),
                demurrage_rate_usd_day=float(plan_inputs.get("demurrage_rate_usd_day", 20000.0)),
                port_charges_usd=float(plan_inputs.get("port_charges_usd", 45000.0)),
                canal_charges_usd=float(plan_inputs.get("canal_charges_usd", 0.0)),
                misc_agency_usd=float(plan_inputs.get("misc_agency_usd", 8000.0)),
                delivery_deadline_days=plan_inputs.get("delivery_deadline_days"),
                vessel_availability_probability=float(plan_inputs.get("vessel_availability_probability", 0.95)),
            )
        else:
            plan = plan_inputs

        return self.scenario_engine.evaluate_scenarios(plan=plan, mc_result=mc_result)

    def simulate_and_assess(
        self,
        plan_inputs: Union[CharterPlanInputs, Dict[str, Any]],
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes unified probabilistic simulation, scenario generation, and risk scoring.
        Returns the exact canonical response schema:
        {
            "expected_cost": float,
            "p10_cost": float,
            "p50_cost": float,
            "p90_cost": float,
            "demurrage_probability": float,
            "late_delivery_probability": float,
            "risk_score": float,
            "scenarios": { ... }
        }
        """
        # 1. Run Monte Carlo simulation
        mc_res = self.simulate_monte_carlo(plan_inputs, n_simulations=n_simulations, seed=seed)

        # 2. Evaluate Scenarios
        scenarios = self.evaluate_scenarios(plan_inputs, mc_result=mc_res)

        # 3. Evaluate 8-category Risk Assessment
        if isinstance(plan_inputs, dict):
            risk_input_dict = plan_inputs.copy()
        else:
            risk_input_dict = {
                "market_data": {
                    "price_volatility_pct": plan_inputs.freight_volatility_pct,
                    "p10": plan_inputs.freight_rate_p10,
                    "p90": plan_inputs.freight_rate_p90,
                    "predicted_rate": plan_inputs.base_freight_rate,
                },
                "port_data": {
                    "expected_wait_days": plan_inputs.expected_wait_days,
                    "p90_wait_days": plan_inputs.p90_wait_days,
                    "delay_probability": mc_res.demurrage_probability,
                },
                "weather_data": {"wave_height_m": 1.8},
                "vessel_data": {
                    "available_vessels_in_region": int(plan_inputs.vessel_availability_probability * 15)
                },
                "schedule_data": {
                    "total_duration": mc_res.expected_duration_days,
                    "delivery_deadline_days": plan_inputs.delivery_deadline_days,
                    "late_delivery_probability": mc_res.late_delivery_probability,
                },
                "demurrage_data": {
                    "expected_demurrage": mc_res.expected_demurrage_cost,
                    "freight_cost": plan_inputs.base_freight_rate * plan_inputs.cargo_quantity_t,
                    "demurrage_probability": mc_res.demurrage_probability,
                },
            }

        risk_assessment = self.evaluate_risk(risk_input_dict)

        return {
            "expected_cost": round(mc_res.expected_cost, 2),
            "p10_cost": round(mc_res.p10_cost, 2),
            "p50_cost": round(mc_res.p50_cost, 2),
            "p90_cost": round(mc_res.p90_cost, 2),
            "demurrage_probability": round(mc_res.demurrage_probability, 4),
            "late_delivery_probability": round(mc_res.late_delivery_probability, 4),
            "risk_score": round(risk_assessment.composite_score, 1),
            "probability_of_infeasibility": round(mc_res.probability_of_infeasibility, 4),
            "risk_assessment": risk_assessment.to_dict(),
            "scenarios": {k: v.to_dict() for k, v in scenarios.items()},
            "cost_distribution": mc_res.distribution,
            "seed": mc_res.seed,
            "n_simulations": mc_res.n_simulations,
        }

    def _get_default_risk_inputs(self) -> Dict[str, Any]:
        """
        Generate default dynamic risk inputs from configuration.
        """
        return {
            "market_data": {"price_volatility_pct": self.settings.default_market_volatility_pct},
            "port_data": {"expected_wait_days": self.settings.default_port_wait_days},
            "weather_data": {
                "wave_height_m": self.settings.default_wave_height_m,
                "storm_warning": self.settings.default_storm_warning,
            },
            "vessel_data": {"available_vessels_in_region": self.settings.default_available_vessels_in_region},
            "geopolitical_data": {"route_conflict_level": self.settings.default_route_conflict_level},
            "operational_data": {"maintenance_due": self.settings.default_maintenance_due},
            "schedule_data": {"total_duration_days": 18.0, "delivery_deadline_days": 25.0},
            "demurrage_data": {"expected_demurrage": 15000.0, "freight_cost": 450000.0},
        }

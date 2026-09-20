"""
DockInsights — Voyage Economics Service (Phase 5).

Calculates full voyage economics from the charterer/cargo-owner perspective,
dynamic bunker fuel costs, port handling durations, canal/agency costs,
demurrage exposures driven by predicted port congestion, and multi-factor
sensitivity analysis.
"""

from typing import Any, Dict, List, Optional, Union
from src.economics.voyage_calculator import VoyageCalculator, VoyageEconomics
from src.economics.voyage_cost import (
    VoyageCostInputs,
    VoyageCostBreakdown,
    calculate_voyage_cost,
    perform_sensitivity_analysis,
    run_full_sensitivity_matrix,
)
from src.economics.cost_models import get_port_costs
from src.data.mock_db import get_mock_port_info
from src.data.vessel_repository import get_vessel_class_spec
from src.services.congestion_service import CongestionService
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VoyageEconomicsService:
    """
    Service for calculating voyage economics.
    Wraps the Phase 5 VoyageEconomicsEngine, integrates CongestionService,
    and provides sensitivity analysis for charterers and cargo owners.
    """

    def __init__(self, congestion_service: Optional[CongestionService] = None):
        self.settings = get_settings()
        self.calculator = VoyageCalculator()
        self.congestion_service = congestion_service or CongestionService()

    def calculate_delivered_cost(
        self,
        cargo_quantity_t: float,
        freight_rate_usd: float,
        origin_port_id: str = "INA_TAB",
        destination_port_id: str = "IND_PAR",
        vessel_class: str = "Panamax",
        route_distance_nm: float = 3800.0,
        vessel_speed_knots: Optional[float] = None,
        fuel_price_usd_per_t: Optional[float] = None,
        expected_waiting_days: Optional[float] = None,
        positioning_distance_nm: float = 0.0,
        canal_charges_usd: float = 0.0,
        agency_fees_usd: Optional[float] = None,
        contingency_cost_usd: float = 0.0,
        delivery_deadline_days: Optional[float] = None,
        charter_type: str = "voyage",
        as_dict: bool = False,
    ) -> Union[VoyageCostBreakdown, Dict[str, Any]]:
        """
        Calculates realistic delivered voyage economics from the charterer perspective.
        Automatically resolves port handling rates and predicted waiting days.
        """
        # 1. Resolve vessel speeds and daily fuel from central vessel repository
        class_spec = get_vessel_class_spec(vessel_class)
        def_speed, def_fuel, def_hire = (
            class_spec.service_speed_knots,
            class_spec.fuel_consumption_tpd,
            class_spec.daily_hire_usd,
        )
        speed = vessel_speed_knots or def_speed
        fuel_price = fuel_price_usd_per_t or self.settings.default_fuel_price_usd_per_t

        # 2. Resolve port handling rates
        origin_info = get_mock_port_info(origin_port_id)
        dest_info = get_mock_port_info(destination_port_id)

        load_handling_rate = getattr(origin_info, "cargo_handling_rate_mt_day", 35000.0) or 35000.0
        disch_handling_rate = getattr(dest_info, "cargo_handling_rate_mt_day", 30000.0) or 30000.0

        # 3. Resolve port costs
        origin_costs = get_port_costs(origin_port_id)
        dest_costs = get_port_costs(destination_port_id)
        load_port_cost = (origin_costs.pilotage_usd + origin_costs.towage_usd + 35000.0) if origin_costs else 50000.0
        disch_port_cost = (dest_costs.pilotage_usd + dest_costs.towage_usd + 35000.0) if dest_costs else 50000.0

        # 4. Resolve congestion waiting days
        if expected_waiting_days is None:
            try:
                expected_waiting_days = self.congestion_service.get_expected_wait_days(
                    port_id=destination_port_id,
                    vessel_class=vessel_class,
                )
            except Exception as e:
                logger.warning("Failed to get congestion wait days: %s, defaulting to 2.5", e)
                expected_waiting_days = 2.5

        # 5. Build inputs
        agency_cost = agency_fees_usd if agency_fees_usd is not None else 8000.0
        inputs = VoyageCostInputs(
            cargo_quantity_t=cargo_quantity_t,
            freight_rate_usd=freight_rate_usd,
            vessel_speed_knots=speed,
            vessel_daily_fuel_consumption_tpd=def_fuel,
            vessel_daily_hire_cost_usd=def_hire,
            route_distance_nm=route_distance_nm,
            positioning_distance_nm=positioning_distance_nm,
            fuel_price_usd_per_t=fuel_price,
            load_port_cost_usd=load_port_cost,
            discharge_port_cost_usd=disch_port_cost,
            port_handling_rate_tpd=load_handling_rate,
            discharge_port_handling_rate_tpd=disch_handling_rate,
            expected_waiting_days=float(expected_waiting_days),
            canal_charges_usd=canal_charges_usd,
            agency_fees_usd=agency_cost,
            contingency_cost_usd=contingency_cost_usd,
            delivery_deadline_days=delivery_deadline_days,
            charter_type=charter_type,
        )

        result = calculate_voyage_cost(inputs)
        return result.to_dict() if as_dict else result

    def calculate_cost(
        self,
        vessel_class: Optional[str] = None,
        cargo_tonnage: Optional[int] = None,
        sailing_distance_nm: Optional[float] = None,
        freight_rate_usd_per_day: Optional[float] = None,
        origin_port_id: Optional[str] = None,
        destination_port_id: Optional[str] = None,
        predicted_idle_days: Optional[float] = None,
        inputs: Optional[VoyageCostInputs] = None,
        **kwargs
    ) -> Union[VoyageEconomics, VoyageCostBreakdown]:
        """
        Calculate full voyage economics with dynamic congestion-driven idle time and demurrage.
        Preserved for full backward compatibility with VoyageCalculator callers.
        """
        if inputs is not None:
            return calculate_voyage_cost(inputs)

        dest_port = destination_port_id or "IND_PAR"
        v_class = vessel_class or "Panamax"

        if predicted_idle_days is None:
            try:
                predicted_idle_days = self.congestion_service.get_expected_wait_days(
                    port_id=dest_port,
                    vessel_class=v_class
                )
            except Exception:
                predicted_idle_days = 2.0

        return self.calculator.calculate(
            vessel_class=v_class,
            cargo_tonnage=cargo_tonnage or 75000,
            sailing_distance_nm=sailing_distance_nm or 4500.0,
            freight_rate_usd_per_day=freight_rate_usd_per_day or self.settings.default_current_freight_rate_usd,
            vlsfo_price_usd_per_tonne=kwargs.get("vlsfo_price_usd_per_tonne", self.settings.default_fuel_price_usd_per_t),
            origin_port_id=origin_port_id or "IND_VZG",
            destination_port_id=dest_port,
            predicted_idle_days=float(predicted_idle_days),
            **{k: v for k, v in kwargs.items() if k not in ["vlsfo_price_usd_per_tonne", "predicted_idle_days"]}
        )

    def calculate_simple_cost(self, inputs: VoyageCostInputs) -> VoyageCostBreakdown:
        """Direct calculation from VoyageCostInputs dataclass."""
        return calculate_voyage_cost(inputs)

    def perform_sensitivity(
        self,
        inputs: VoyageCostInputs,
        parameter: str,
        values: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run single-parameter sensitivity analysis."""
        return perform_sensitivity_analysis(inputs, parameter, values)

    def run_sensitivity_matrix(self, inputs: VoyageCostInputs) -> Dict[str, List[Dict[str, Any]]]:
        """Run comprehensive multi-parameter sensitivity analysis."""
        return run_full_sensitivity_matrix(inputs)

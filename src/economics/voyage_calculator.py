"""
DockInsights — Voyage Economics Calculator.

Full voyage P&L estimation combining freight cost, bunker fuel,
port charges, insurance, and demurrage exposure.

DETERMINISTIC: Industry-standard formulas with configurable parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.economics.bunker_estimator import BunkerEstimate, BunkerEstimator
from src.economics.cost_models import PortCosts, get_port_costs
from src.economics.demurrage_calculator import DemurrageCalculator, DemurrageEstimate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VoyageCostBreakdown:
    """Itemized voyage cost breakdown."""
    freight_cost_usd: float
    bunker_cost_usd: float
    load_port_charges_usd: float
    discharge_port_charges_usd: float
    insurance_usd: float
    expected_demurrage_usd: float
    miscellaneous_usd: float


@dataclass
class VoyageEconomics:
    """Complete voyage economics result."""
    total_voyage_cost_usd: float
    cost_per_tonne_usd: float
    cargo_tonnage: int
    vessel_class: str
    sailing_days: float
    total_voyage_days: float  # sailing + port + waiting
    breakdown: VoyageCostBreakdown
    bunker_detail: Optional[BunkerEstimate] = None
    demurrage_detail: Optional[DemurrageEstimate] = None


class VoyageCalculator:
    """
    Calculates full voyage economics for a given route, vessel, and cargo.

    Total Voyage Cost =
        Freight Cost (rate × days or rate × tonnes)
      + Bunker Cost (fuel consumption × days × VLSFO price)
      + Load Port Charges
      + Discharge Port Charges
      + Insurance Premium
      + Expected Demurrage Cost
      + Miscellaneous (agent fees, surveys, comms)
    """

    def __init__(self):
        self.bunker_estimator = BunkerEstimator()
        self.demurrage_calculator = DemurrageCalculator()

    def calculate(
        self,
        vessel_class: str,
        cargo_tonnage: int,
        sailing_distance_nm: float,
        freight_rate_usd_per_day: float,
        vlsfo_price_usd_per_tonne: float,
        origin_port_id: str,
        destination_port_id: str,
        predicted_idle_days: float = 2.0,
        loading_days: float = 3.0,
        discharge_days: float = 4.0,
        insurance_rate_pct: float = 0.05,
        misc_usd: float = 5_000,
    ) -> VoyageEconomics:
        """
        Calculate complete voyage economics.

        Args:
            vessel_class: Vessel class name.
            cargo_tonnage: Cargo quantity in tonnes.
            sailing_distance_nm: Sailing distance in nautical miles.
            freight_rate_usd_per_day: Charter rate (USD/day).
            vlsfo_price_usd_per_tonne: Current bunker fuel price.
            origin_port_id: Load port identifier.
            destination_port_id: Discharge port identifier.
            predicted_idle_days: ML-predicted waiting time at destination.
            loading_days: Expected loading time.
            discharge_days: Expected discharge time.
            insurance_rate_pct: Insurance as % of cargo value (approximate).
            misc_usd: Miscellaneous costs (agent fees, surveys, etc.).

        Returns:
            VoyageEconomics with total cost, cost per tonne, and full breakdown.
        """
        # 1. Bunker fuel cost
        port_days = loading_days + discharge_days + predicted_idle_days
        bunker = self.bunker_estimator.estimate(
            vessel_class=vessel_class,
            sailing_distance_nm=sailing_distance_nm,
            vlsfo_price_usd_per_tonne=vlsfo_price_usd_per_tonne,
            port_days=port_days,
        )

        # 2. Freight cost
        total_charter_days = bunker.sailing_days + port_days
        freight_cost = freight_rate_usd_per_day * total_charter_days

        # 3. Port charges
        load_port_costs = self._calculate_port_charges(
            origin_port_id, cargo_tonnage, loading_days
        )
        discharge_port_costs = self._calculate_port_charges(
            destination_port_id, cargo_tonnage, discharge_days
        )

        # 4. Insurance (simplified as % of freight cost)
        insurance = freight_cost * insurance_rate_pct

        # 5. Demurrage exposure
        demurrage = self.demurrage_calculator.calculate(
            vessel_class=vessel_class,
            predicted_idle_days=predicted_idle_days,
        )

        # Assemble breakdown
        breakdown = VoyageCostBreakdown(
            freight_cost_usd=round(freight_cost, 0),
            bunker_cost_usd=round(bunker.total_bunker_cost_usd, 0),
            load_port_charges_usd=round(load_port_costs, 0),
            discharge_port_charges_usd=round(discharge_port_costs, 0),
            insurance_usd=round(insurance, 0),
            expected_demurrage_usd=round(demurrage.net_demurrage_exposure_usd, 0),
            miscellaneous_usd=misc_usd,
        )

        total = (
            breakdown.freight_cost_usd
            + breakdown.bunker_cost_usd
            + breakdown.load_port_charges_usd
            + breakdown.discharge_port_charges_usd
            + breakdown.insurance_usd
            + max(0, breakdown.expected_demurrage_usd)
            + breakdown.miscellaneous_usd
        )

        cost_per_tonne = total / cargo_tonnage if cargo_tonnage > 0 else 0

        return VoyageEconomics(
            total_voyage_cost_usd=round(total, 0),
            cost_per_tonne_usd=round(cost_per_tonne, 2),
            cargo_tonnage=cargo_tonnage,
            vessel_class=vessel_class,
            sailing_days=bunker.sailing_days,
            total_voyage_days=round(total_charter_days, 1),
            breakdown=breakdown,
            bunker_detail=bunker,
            demurrage_detail=demurrage,
        )

    @staticmethod
    def _calculate_port_charges(
        port_id: str,
        cargo_tonnage: int,
        port_days: float,
    ) -> float:
        """Calculate total port charges for one port call."""
        costs = get_port_costs(port_id)
        if costs is None:
            # Fallback generic estimate
            return 25_000.0

        total = (
            costs.pilotage_usd
            + costs.towage_usd
            + costs.berth_hire_usd_per_day * port_days
            + costs.stevedoring_usd_per_tonne * cargo_tonnage
            + costs.wharfage_usd_per_tonne * cargo_tonnage
            + costs.agency_fees_usd
        )
        return total

"""
DockInsights — Bunker Fuel Cost Estimator.

Estimates fuel consumption and cost for a voyage based on vessel class,
distance, and current VLSFO price.

DETERMINISTIC: Standard consumption rates × sailing days × fuel price.
"""

from dataclasses import dataclass

from src.utils.constants import VESSEL_FUEL_CONSUMPTION_TPD, VESSEL_SPEEDS_KNOTS, VesselClass
from src.utils.geo import estimate_sailing_days
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BunkerEstimate:
    """Bunker fuel cost estimate."""
    sailing_days: float
    consumption_per_day_tonnes: float
    total_fuel_tonnes: float
    fuel_price_usd_per_tonne: float
    total_bunker_cost_usd: float
    port_fuel_tonnes: float  # Fuel consumed in port (at idle)
    total_fuel_with_port_tonnes: float


class BunkerEstimator:
    """
    Estimates bunker fuel cost for a voyage.

    Assumptions:
    - At-sea consumption: class-specific (from constants)
    - In-port consumption: ~10% of at-sea rate (hotel load)
    - Port time: loading days + discharge days + waiting
    """

    # In-port fuel consumption as fraction of at-sea rate
    PORT_CONSUMPTION_FACTOR = 0.10

    def estimate(
        self,
        vessel_class: str,
        sailing_distance_nm: float,
        vlsfo_price_usd_per_tonne: float,
        port_days: float = 5.0,  # Total port time (both ends)
        speed_knots: float | None = None,
    ) -> BunkerEstimate:
        """
        Estimate bunker fuel cost for a voyage.

        Args:
            vessel_class: Vessel class name.
            sailing_distance_nm: Total sailing distance in nautical miles.
            vlsfo_price_usd_per_tonne: Current VLSFO price.
            port_days: Total days in port (loading + discharge + waiting).
            speed_knots: Override default speed. If None, uses class default.

        Returns:
            BunkerEstimate with fuel quantities and total cost.
        """
        # Resolve vessel class enum
        try:
            vc = VesselClass(vessel_class)
        except ValueError:
            logger.warning(f"Unknown vessel class '{vessel_class}', using Panamax defaults")
            vc = VesselClass.PANAMAX

        speed = speed_knots or VESSEL_SPEEDS_KNOTS[vc]
        consumption_tpd = VESSEL_FUEL_CONSUMPTION_TPD[vc]

        sailing_days = estimate_sailing_days(sailing_distance_nm, speed)
        sea_fuel = consumption_tpd * sailing_days
        port_fuel = consumption_tpd * self.PORT_CONSUMPTION_FACTOR * port_days
        total_fuel = sea_fuel + port_fuel
        total_cost = total_fuel * vlsfo_price_usd_per_tonne

        return BunkerEstimate(
            sailing_days=round(sailing_days, 2),
            consumption_per_day_tonnes=consumption_tpd,
            total_fuel_tonnes=round(sea_fuel, 1),
            fuel_price_usd_per_tonne=vlsfo_price_usd_per_tonne,
            total_bunker_cost_usd=round(total_cost, 0),
            port_fuel_tonnes=round(port_fuel, 1),
            total_fuel_with_port_tonnes=round(total_fuel, 1),
        )

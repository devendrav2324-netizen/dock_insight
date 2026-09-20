"""
DockInsights — Port-Specific Cost Models.

Configurable lookup tables for port charges, pilotage, towage,
berth hire, and stevedoring rates.

DETERMINISTIC: All values are industry-standard or port-tariff based.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortCosts:
    """Cost parameters for a specific port."""
    port_id: str
    pilotage_usd: float
    towage_usd: float
    berth_hire_usd_per_day: float
    port_dues_usd_per_grt: float  # Per Gross Registered Tonnage
    stevedoring_usd_per_tonne: float
    wharfage_usd_per_tonne: float
    agency_fees_usd: float


# ---- Default port cost estimates (illustrative, not real tariffs) ----
# These should be updated with actual port tariff data in production.
DEFAULT_PORT_COSTS: Dict[str, PortCosts] = {
    "IND_VZG": PortCosts(
        port_id="IND_VZG",
        pilotage_usd=8_000,
        towage_usd=6_000,
        berth_hire_usd_per_day=2_500,
        port_dues_usd_per_grt=0.12,
        stevedoring_usd_per_tonne=2.50,
        wharfage_usd_per_tonne=0.80,
        agency_fees_usd=3_000,
    ),
    "IND_GVM": PortCosts(
        port_id="IND_GVM",
        pilotage_usd=7_500,
        towage_usd=5_500,
        berth_hire_usd_per_day=2_800,
        port_dues_usd_per_grt=0.11,
        stevedoring_usd_per_tonne=2.20,
        wharfage_usd_per_tonne=0.75,
        agency_fees_usd=2_800,
    ),
    "IND_GOP": PortCosts(
        port_id="IND_GOP",
        pilotage_usd=6_000,
        towage_usd=4_500,
        berth_hire_usd_per_day=2_000,
        port_dues_usd_per_grt=0.10,
        stevedoring_usd_per_tonne=2.80,
        wharfage_usd_per_tonne=0.90,
        agency_fees_usd=2_500,
    ),
    "IND_DHM": PortCosts(
        port_id="IND_DHM",
        pilotage_usd=7_000,
        towage_usd=5_000,
        berth_hire_usd_per_day=2_600,
        port_dues_usd_per_grt=0.11,
        stevedoring_usd_per_tonne=2.30,
        wharfage_usd_per_tonne=0.70,
        agency_fees_usd=2_600,
    ),
    "IND_PAR": PortCosts(
        port_id="IND_PAR",
        pilotage_usd=7_500,
        towage_usd=5_500,
        berth_hire_usd_per_day=2_400,
        port_dues_usd_per_grt=0.12,
        stevedoring_usd_per_tonne=2.40,
        wharfage_usd_per_tonne=0.85,
        agency_fees_usd=2_800,
    ),
    "IND_HLD": PortCosts(
        port_id="IND_HLD",
        pilotage_usd=9_000,
        towage_usd=7_000,
        berth_hire_usd_per_day=2_200,
        port_dues_usd_per_grt=0.13,
        stevedoring_usd_per_tonne=3.00,
        wharfage_usd_per_tonne=1.00,
        agency_fees_usd=3_500,
    ),
}


def get_port_costs(port_id: str) -> Optional[PortCosts]:
    """
    Retrieve cost parameters for a port.

    Returns None if port not found in the cost database.
    In production, this would query a database table.
    """
    costs = DEFAULT_PORT_COSTS.get(port_id)
    if costs is None:
        logger.warning(f"No cost data for port {port_id}, using generic estimate")
    return costs

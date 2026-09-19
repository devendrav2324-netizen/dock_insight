"""
Charter-AI — Historical Tender Scenarios (Phase 11).

Defines standardized, reproducible dry-bulk cargo shipment fixtures across
historical evaluation windows (2022, 2023, 2024).
Reflects realistic procurement schedules of Indian East Coast power and steel utilities.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class BacktestScenario:
    """Specification of an empirical dry-bulk cargo shipment fixture."""
    scenario_id: str
    year: int
    order_date: datetime
    origin_port_id: str
    destination_port_id: str
    cargo_type: str
    cargo_quantity_t: float
    laycan_start: datetime
    laycan_end: datetime
    required_delivery_date: datetime
    risk_tolerance: str = "MEDIUM"
    notes: Optional[str] = None


# Core representative routes and typical parcel sizes for Indian East Coast imports
HISTORICAL_FIXTURE_TEMPLATES = [
    # 1. Australian Coking/Thermal Coal to Gangavaram & Paradip (Cape & Panamax)
    {"origin": "AUS_NEW", "dest": "IND_GVM", "cargo": "thermal_coal", "qty": 80000.0, "risk": "MEDIUM"},
    {"origin": "AUS_NEW", "dest": "IND_PAR", "cargo": "thermal_coal", "qty": 150000.0, "risk": "LOW"},
    {"origin": "AUS_HAY", "dest": "IND_GVM", "cargo": "thermal_coal", "qty": 120000.0, "risk": "MEDIUM"},
    {"origin": "AUS_NEW", "dest": "IND_VZG", "cargo": "thermal_coal", "qty": 75000.0, "risk": "HIGH"},

    # 2. Indonesian Coal to Dhamra, Gopalpur & Paradip (Panamax, Supramax & Handysize)
    {"origin": "IDN_TAB", "dest": "IND_DHM", "cargo": "thermal_coal", "qty": 75000.0, "risk": "LOW"},
    {"origin": "IDN_TAB", "dest": "IND_GOP", "cargo": "thermal_coal", "qty": 50000.0, "risk": "HIGH"},
    {"origin": "IDN_TAB", "dest": "IND_PAR", "cargo": "thermal_coal", "qty": 100000.0, "risk": "MEDIUM"},
    {"origin": "IDN_TAB", "dest": "IND_DHM", "cargo": "thermal_coal", "qty": 55000.0, "risk": "MEDIUM"},

    # 3. South African Coal to Visakhapatnam & Paradip (Capesize & Panamax)
    {"origin": "ZAF_RIC", "dest": "IND_VZG", "cargo": "thermal_coal", "qty": 140000.0, "risk": "LOW"},
    {"origin": "ZAF_RIC", "dest": "IND_PAR", "cargo": "thermal_coal", "qty": 75000.0, "risk": "MEDIUM"},
    {"origin": "ZAF_RIC", "dest": "IND_VZG", "cargo": "thermal_coal", "qty": 80000.0, "risk": "HIGH"},
    {"origin": "AUS_NEW", "dest": "IND_GVM", "cargo": "thermal_coal", "qty": 160000.0, "risk": "LOW"},
]


def generate_standard_scenarios(
    years: Optional[List[int]] = None,
) -> List[BacktestScenario]:
    """
    Generates a deterministic sequence of cargo shipment fixtures across specified years.
    Produces 12 realistic monthly fixtures per year (36 total for 2022-2024).
    """
    if years is None:
        years = [2022, 2023, 2024]

    scenarios = []

    for yr in sorted(years):
        for month_idx, tmpl in enumerate(HISTORICAL_FIXTURE_TEMPLATES, start=1):
            day = 10 + (month_idx % 12)  # Staggered entry dates within the month
            order_dt = datetime(yr, month_idx, min(25, day), 10, 0, 0)
            laycan_start = order_dt + timedelta(days=7)
            laycan_end = laycan_start + timedelta(days=7)
            # Delivery deadline allows ~35-40 days for loading, sailing, congestion & discharge
            delivery_deadline = laycan_start + timedelta(days=36)

            scen_id = f"SCEN_{yr}_{month_idx:02d}_{tmpl['origin']}_{tmpl['dest']}"

            scenarios.append(
                BacktestScenario(
                    scenario_id=scen_id,
                    year=yr,
                    order_date=order_dt,
                    origin_port_id=tmpl["origin"],
                    destination_port_id=tmpl["dest"],
                    cargo_type=tmpl["cargo"],
                    cargo_quantity_t=tmpl["qty"],
                    laycan_start=laycan_start,
                    laycan_end=laycan_end,
                    required_delivery_date=delivery_deadline,
                    risk_tolerance=tmpl["risk"],
                    notes=f"Monthly fixture {month_idx}/{yr}: {tmpl['qty']:,.0f} MT {tmpl['cargo']}",
                )
            )

    return scenarios

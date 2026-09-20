"""
DockInsights — Central Vessel Specifications & Availability Repository.

Single source of truth for vessel class specifications, technical parameters,
demurrage benchmarks, and dynamic vessel availability calculations.
Eliminates duplicated vessel parameter dictionaries across optimization,
economics, and risk modules.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from src.utils.config import get_settings


@dataclass
class VesselClassSpec:
    """Canonical specification for a dry-bulk vessel class."""
    class_name: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    draft_max_m: float
    loa_max_m: float
    beam_max_m: float
    service_speed_knots: float
    fuel_consumption_tpd: float
    daily_hire_usd: float
    demurrage_rate_usd: float
    maneuverability_score: float = 78.0
    default_laytime_days: float = 6.0


_VESSEL_CLASS_REGISTRY: Dict[str, VesselClassSpec] = {
    "Handysize": VesselClassSpec(
        class_name="Handysize",
        dwt_min=15000,
        dwt_max=40000,
        typical_dwt=35000,
        draft_max_m=10.5,
        loa_max_m=180.0,
        beam_max_m=28.4,
        service_speed_knots=12.0,
        fuel_consumption_tpd=22.0,
        daily_hire_usd=14000.0,
        demurrage_rate_usd=12000.0,
        maneuverability_score=85.0,
        default_laytime_days=5.0,
    ),
    "Supramax": VesselClassSpec(
        class_name="Supramax",
        dwt_min=40000,
        dwt_max=60000,
        typical_dwt=55000,
        draft_max_m=12.5,
        loa_max_m=199.9,
        beam_max_m=32.2,
        service_speed_knots=12.5,
        fuel_consumption_tpd=27.0,
        daily_hire_usd=17000.0,
        demurrage_rate_usd=15000.0,
        maneuverability_score=82.0,
        default_laytime_days=5.0,
    ),
    "Panamax": VesselClassSpec(
        class_name="Panamax",
        dwt_min=60000,
        dwt_max=85000,
        typical_dwt=75000,
        draft_max_m=14.4,
        loa_max_m=225.0,
        beam_max_m=32.2,
        service_speed_knots=13.0,
        fuel_consumption_tpd=32.0,
        daily_hire_usd=20000.0,
        demurrage_rate_usd=20000.0,
        maneuverability_score=78.0,
        default_laytime_days=6.0,
    ),
    "Capesize": VesselClassSpec(
        class_name="Capesize",
        dwt_min=100000,
        dwt_max=200000,
        typical_dwt=175000,
        draft_max_m=18.5,
        loa_max_m=292.0,
        beam_max_m=45.0,
        service_speed_knots=13.5,
        fuel_consumption_tpd=52.0,
        daily_hire_usd=28000.0,
        demurrage_rate_usd=35000.0,
        maneuverability_score=72.0,
        default_laytime_days=7.0,
    ),
    "VLOC": VesselClassSpec(
        class_name="VLOC",
        dwt_min=200000,
        dwt_max=400000,
        typical_dwt=250000,
        draft_max_m=23.0,
        loa_max_m=360.0,
        beam_max_m=65.0,
        service_speed_knots=14.0,
        fuel_consumption_tpd=65.0,
        daily_hire_usd=35000.0,
        demurrage_rate_usd=45000.0,
        maneuverability_score=65.0,
        default_laytime_days=8.0,
    ),
}


_DEFAULT_PANAMAX_FALLBACK = _VESSEL_CLASS_REGISTRY["Panamax"]


def get_vessel_class_spec(class_name: str) -> VesselClassSpec:
    """
    Returns the authoritative specification for a given vessel class.
    Falls back to Panamax if class_name is not recognized.
    """
    return _VESSEL_CLASS_REGISTRY.get(class_name, _DEFAULT_PANAMAX_FALLBACK)


def get_all_vessel_class_specs() -> List[VesselClassSpec]:
    """Returns specifications for all registered vessel classes."""
    return list(_VESSEL_CLASS_REGISTRY.values())


def calculate_vessel_availability(
    cargo_quantity_t: float,
    vessel_list: Optional[List[Dict[str, Any]]] = None,
    target_vessel_class: Optional[str] = None,
    max_draft_m: Optional[float] = None,
    max_loa_m: Optional[float] = None,
    max_beam_m: Optional[float] = None,
    expected_loading_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculates dynamic tonnage availability based on actual candidate vessel records.
    Consumes available vessels from the database/list and compares eligible capacity
    against cargo requirements using configurable settings.

    Filter steps enforced:
    1. Availability status ("AVAILABLE")
    2. Vessel class / DWT match (if target_vessel_class supplied)
    3. Port physical draft constraint (vessel max_draft_m <= port max_draft_m)
    4. Port physical LOA constraint (vessel loa_m <= port max_loa_m)
    5. Port physical Beam constraint (vessel beam_m <= port max_beam_m)
    6. Availability window / laycan constraint (vessel available_from <= expected_loading_date)
    """
    settings = get_settings()
    if vessel_list is None:
        from src.data.mock_db import get_mock_vessels
        vessels = get_mock_vessels()
    else:
        vessels = vessel_list

    data_source = "SYNTHETIC_DEMO"
    if vessels and "data_source" in vessels[0]:
        data_source = vessels[0].get("data_source", "SYNTHETIC_DEMO")

    total_vessels = len(vessels)
    available_vessels_count = 0
    eligible_vessels = []

    for v in vessels:
        # 1. Status check
        if str(v.get("availability_status", "AVAILABLE")).upper() != "AVAILABLE":
            continue
        available_vessels_count += 1

        # 2. Vessel class check
        if target_vessel_class and v.get("vessel_class") != target_vessel_class:
            continue

        # 3. Port draft constraint
        if max_draft_m is not None and v.get("max_draft_m") is not None:
            if float(v["max_draft_m"]) > float(max_draft_m):
                continue

        # 4. Port LOA constraint
        if max_loa_m is not None and v.get("loa_m") is not None:
            if float(v["loa_m"]) > float(max_loa_m):
                continue

        # 5. Port Beam constraint
        if max_beam_m is not None and v.get("beam_m") is not None:
            if float(v["beam_m"]) > float(max_beam_m):
                continue

        # 6. Availability window constraint
        if expected_loading_date is not None and v.get("available_from") is not None:
            avail_date = v["available_from"]
            req_date = expected_loading_date
            if hasattr(avail_date, "date"):
                avail_date = avail_date.date()
            elif isinstance(avail_date, str):
                from datetime import date
                avail_date = date.fromisoformat(avail_date)
            if hasattr(req_date, "date"):
                req_date = req_date.date()
            elif isinstance(req_date, str):
                from datetime import date
                req_date = date.fromisoformat(req_date)
            if avail_date > req_date:
                continue

        eligible_vessels.append(v)

    total_eligible_capacity = sum(float(v.get("dwt", 0)) for v in eligible_vessels)
    eligible_count = len(eligible_vessels)

    cargo_req = max(1.0, float(cargo_quantity_t))
    capacity_ratio = total_eligible_capacity / cargo_req

    if capacity_ratio < settings.availability_tight_threshold:
        availability_label = "TIGHT"
    elif capacity_ratio < settings.availability_balanced_threshold:
        availability_label = "BALANCED"
    else:
        availability_label = "SURPLUS"

    return {
        "vessel_availability_label": availability_label,
        "total_vessels": total_vessels,
        "available_vessels": available_vessels_count,
        "eligible_vessels": eligible_count,
        "available_vessel_count": eligible_count,
        "eligible_capacity_t": total_eligible_capacity,
        "total_eligible_dwt": total_eligible_capacity,
        "capacity_ratio": round(capacity_ratio, 2),
        "data_source": data_source,
    }


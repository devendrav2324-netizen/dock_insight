"""
Charter-AI — Mock Database (V2).

Realistic synthetic demo data used for SIH Demo Mode.
All records are explicitly tagged source="SYNTHETIC_DEMO" with confidence_level="LOW".
No random values are used.
"""

from datetime import date
from typing import List, Dict, Any

from src.optimization.vessel_selector import VesselSpecs
from src.optimization.port_compatibility import PortInfo
from src.data.models import Port, Vessel
from src.utils.config import get_settings


def get_mock_vessel_db() -> List[VesselSpecs]:
    """Returns a deterministic list of vessel class specifications derived from vessel_repository."""
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    # Filter to standard 4 main classes or return all specs mapped to VesselSpecs
    return [
        VesselSpecs(
            class_name=s.class_name,
            dwt_max=s.dwt_max,
            dwt_min=s.dwt_min,
            typical_dwt=s.typical_dwt,
            draft_max_m=s.draft_max_m,
            loa_max_m=s.loa_max_m,
            beam_max_m=s.beam_max_m,
        )
        for s in specs if s.class_name in ("Handysize", "Supramax", "Panamax", "Capesize")
    ]



def get_mock_port_info(port_id: str) -> PortInfo:
    """Returns realistic port infrastructure constraints for known ports."""
    port_data = {
        "IND_VZG": {"name": "Visakhapatnam", "draft": 20.0, "loa": 320.0, "beam": 50.0, "rate": 45000, "berths": 12},
        "IND_GVM": {"name": "Gangavaram", "draft": 21.0, "loa": 320.0, "beam": 50.0, "rate": 55000, "berths": 6},
        "IND_PAR": {"name": "Paradip", "draft": 18.5, "loa": 300.0, "beam": 48.0, "rate": 40000, "berths": 10},
        "IND_DHM": {"name": "Dhamra", "draft": 20.0, "loa": 350.0, "beam": 55.0, "rate": 50000, "berths": 5},
        "IND_HLD": {"name": "Haldia", "draft": 8.5, "loa": 200.0, "beam": 31.0, "rate": 20000, "berths": 8},
        "IND_GOP": {"name": "Gopalpur", "draft": 14.5, "loa": 230.0, "beam": 33.0, "rate": 25000, "berths": 3},
        "IDN_TAB": {"name": "Taboneo", "draft": 20.0, "loa": 350.0, "beam": 55.0, "rate": 30000, "berths": 15},
    }
    spec = port_data.get(port_id, {"name": port_id.title(), "draft": 20.0, "loa": 350.0, "beam": 50.0, "rate": 20000, "berths": 5})
    return PortInfo(
        port_id=port_id,
        port_name=spec["name"],
        max_draft_m=spec["draft"],
        max_loa_m=spec["loa"],
        max_beam_m=spec["beam"],
        cargo_handling_rate_tpd=spec["rate"],
        berthing_capacity=spec["berths"],
        current_congestion_factor=1.0
    )


def get_mock_ports() -> List[Dict[str, Any]]:
    """Returns deterministic normalized mock port records."""
    return [
        {
            "port_id": "IND_VZG", "port_name": "Visakhapatnam", "country": "IND",
            "latitude": 17.6868, "longitude": 83.2185, "max_draft_m": 18.1,
            "max_loa_m": 280.0, "max_beam_m": 45.0, "cargo_handling_rate_mt_day": 45000.0,
            "berth_count": 12, "coal_terminal": True, "iron_ore_terminal": True,
            "grain_terminal": False, "tidal_restriction": False, "night_navigation_restriction": False,
            "source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1), "confidence_level": "LOW"
        },
        {
            "port_id": "IND_GVM", "port_name": "Gangavaram", "country": "IND",
            "latitude": 17.6167, "longitude": 83.2333, "max_draft_m": 21.0,
            "max_loa_m": 300.0, "max_beam_m": 50.0, "cargo_handling_rate_mt_day": 55000.0,
            "berth_count": 6, "coal_terminal": True, "iron_ore_terminal": True,
            "grain_terminal": False, "tidal_restriction": False, "night_navigation_restriction": False,
            "source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1), "confidence_level": "LOW"
        },
        {
            "port_id": "IND_PAR", "port_name": "Paradip", "country": "IND",
            "latitude": 20.2644, "longitude": 86.6715, "max_draft_m": 16.5,
            "max_loa_m": 260.0, "max_beam_m": 43.0, "cargo_handling_rate_mt_day": 40000.0,
            "berth_count": 10, "coal_terminal": True, "iron_ore_terminal": True,
            "grain_terminal": False, "tidal_restriction": False, "night_navigation_restriction": False,
            "source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1), "confidence_level": "LOW"
        },
        {
            "port_id": "IND_DHM", "port_name": "Dhamra", "country": "IND",
            "latitude": 20.8039, "longitude": 86.9744, "max_draft_m": 18.0,
            "max_loa_m": 290.0, "max_beam_m": 48.0, "cargo_handling_rate_mt_day": 50000.0,
            "berth_count": 5, "coal_terminal": True, "iron_ore_terminal": True,
            "grain_terminal": False, "tidal_restriction": True, "night_navigation_restriction": False,
            "source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1), "confidence_level": "LOW"
        },
        {
            "port_id": "IND_HLD", "port_name": "Haldia", "country": "IND",
            "latitude": 22.0234, "longitude": 88.0645, "max_draft_m": 8.5,
            "max_loa_m": 200.0, "max_beam_m": 31.0, "cargo_handling_rate_mt_day": 20000.0,
            "berth_count": 8, "coal_terminal": True, "iron_ore_terminal": False,
            "grain_terminal": False, "tidal_restriction": True, "night_navigation_restriction": True,
            "source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1), "confidence_level": "LOW"
        },
    ]


def get_mock_vessels() -> List[Dict[str, Any]]:
    """Returns deterministic normalized mock vessel records."""
    return [
        {
            "vessel_id": "VSL_001", "imo_number": "9451234", "vessel_name": "Bay Trader",
            "vessel_class": "Handysize", "dwt": 35000, "loa_m": 180.0, "beam_m": 28.4,
            "max_draft_m": 10.5, "service_speed_knots": 13.0, "ballast_speed_knots": 13.5,
            "laden_speed_knots": 12.5, "fuel_consumption_mt_day": 22.0, "age_years": 7,
            "current_latitude": 17.5, "current_longitude": 83.0, "availability_status": "AVAILABLE",
            "available_from": date(2026, 1, 1), "data_source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1)
        },
        {
            "vessel_id": "VSL_002", "imo_number": "9562345", "vessel_name": "Eastern Harmony",
            "vessel_class": "Supramax", "dwt": 56000, "loa_m": 199.9, "beam_m": 32.2,
            "max_draft_m": 12.5, "service_speed_knots": 13.5, "ballast_speed_knots": 14.0,
            "laden_speed_knots": 13.0, "fuel_consumption_mt_day": 28.0, "age_years": 5,
            "current_latitude": 18.0, "current_longitude": 84.0, "availability_status": "AVAILABLE",
            "available_from": date(2026, 1, 1), "data_source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1)
        },
        {
            "vessel_id": "VSL_003", "imo_number": "9673456", "vessel_name": "Ocean Fortune",
            "vessel_class": "Panamax", "dwt": 76000, "loa_m": 225.0, "beam_m": 32.2,
            "max_draft_m": 14.2, "service_speed_knots": 13.5, "ballast_speed_knots": 14.0,
            "laden_speed_knots": 13.0, "fuel_consumption_mt_day": 34.0, "age_years": 4,
            "current_latitude": 19.5, "current_longitude": 85.5, "availability_status": "AVAILABLE",
            "available_from": date(2026, 1, 1), "data_source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1)
        },
        {
            "vessel_id": "VSL_004", "imo_number": "9784567", "vessel_name": "Global Pioneer",
            "vessel_class": "Capesize", "dwt": 180000, "loa_m": 292.0, "beam_m": 45.0,
            "max_draft_m": 18.2, "service_speed_knots": 13.0, "ballast_speed_knots": 13.5,
            "laden_speed_knots": 12.5, "fuel_consumption_mt_day": 52.0, "age_years": 8,
            "current_latitude": 20.0, "current_longitude": 87.0, "availability_status": "AVAILABLE",
            "available_from": date(2026, 1, 1), "data_source": "SYNTHETIC_DEMO", "source_date": date(2026, 1, 1)
        },
    ]


def require_sih_demo_mode():
    """Raises an error if trying to access mock data when not in demo mode."""
    if not get_settings().sih_demo_mode:
        raise ValueError("System is not in SIH Demo Mode. Live database connection required.")

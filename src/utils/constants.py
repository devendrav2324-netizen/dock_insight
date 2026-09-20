"""
DockInsights — Business Constants & Enumerations.

Centralized enums and constants used across all modules.
"""

from enum import Enum


# =============================================================================
# Vessel Classes
# =============================================================================

class VesselClass(str, Enum):
    """Standard dry-bulk vessel size classifications."""
    HANDYSIZE = "Handysize"
    SUPRAMAX = "Supramax"
    PANAMAX = "Panamax"
    POST_PANAMAX = "Post-Panamax"
    CAPESIZE = "Capesize"
    VLOC = "VLOC"


def _get_vessel_speeds() -> dict[VesselClass, float]:
    from src.data.vessel_repository import get_vessel_class_spec
    res = {}
    for vc in VesselClass:
        spec = get_vessel_class_spec(vc.value)
        res[vc] = spec.service_speed_knots
    return res


def _get_vessel_fuel_consumptions() -> dict[VesselClass, float]:
    from src.data.vessel_repository import get_vessel_class_spec
    res = {}
    for vc in VesselClass:
        spec = get_vessel_class_spec(vc.value)
        res[vc] = spec.fuel_consumption_tpd
    return res


VESSEL_SPEEDS_KNOTS: dict[VesselClass, float] = _get_vessel_speeds()
VESSEL_FUEL_CONSUMPTION_TPD: dict[VesselClass, float] = _get_vessel_fuel_consumptions()



# =============================================================================
# Port IDs
# =============================================================================

class IndianPort(str, Enum):
    """East Coast Indian destination ports in scope."""
    VISAKHAPATNAM = "IND_VZG"
    GANGAVARAM = "IND_GVM"
    GOPALPUR = "IND_GOP"
    DHAMRA = "IND_DHM"
    PARADIP = "IND_PAR"
    HALDIA = "IND_HLD"
    SAGAR = "IND_SAG"  # Anchorage/lightering point, not a berthing port


class OriginPort(str, Enum):
    """Origin ports in scope."""
    NEWCASTLE = "AUS_NEW"
    HAY_POINT = "AUS_HAY"
    GLADSTONE = "AUS_GLD"
    TABANG = "IDN_TAB"
    BALIKPAPAN = "IDN_BAL"
    RICHARDS_BAY = "ZAF_RIC"
    HAMPTON_ROADS = "USA_HAM"
    NACALA = "MOZ_NAC"
    VOSTOCHNY = "RUS_VOS"


# =============================================================================
# Cargo Types
# =============================================================================

class CargoType(str, Enum):
    """Cargo types in scope."""
    THERMAL_COAL_HCV = "COAL_THM_HCV"
    THERMAL_COAL_MCV = "COAL_THM_MCV"
    HARD_COKING_COAL = "COAL_COK_HARD"
    SEMI_COKING_COAL = "COAL_COK_SEMI"
    PET_COKE = "COAL_PET"


# =============================================================================
# Contract Types
# =============================================================================

class ContractType(str, Enum):
    """Chartering contract types."""
    SPOT = "spot"
    SHORT_TERM = "short_term"        # 3-6 months
    MEDIUM_TERM = "medium_term"      # 6-18 months
    HYBRID = "hybrid"                # Spot base + CoA overlay


# =============================================================================
# Risk Levels
# =============================================================================

class RiskLevel(str, Enum):
    """Qualitative risk classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


def risk_level_from_score(score: float) -> RiskLevel:
    """Convert a 0-100 risk score to a qualitative level."""
    if score < 25:
        return RiskLevel.LOW
    elif score < 50:
        return RiskLevel.MODERATE
    elif score < 75:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


# =============================================================================
# Scoring Weights (configurable defaults)
# =============================================================================

# Default multi-criteria charter scoring weights
DEFAULT_CHARTER_SCORE_WEIGHTS = {
    "cost": 0.40,
    "risk": 0.25,
    "schedule_reliability": 0.20,
    "flexibility": 0.15,
}

# Default risk aggregation weights
DEFAULT_RISK_WEIGHTS = {
    "market": 0.30,
    "port": 0.25,
    "weather": 0.25,
    "operational": 0.20,
}


# =============================================================================
# Cyclone Season (Bay of Bengal)
# =============================================================================

# Primary cyclone months for the Bay of Bengal (East Coast India)
CYCLONE_SEASON_MONTHS = {4, 5, 6, 10, 11, 12}

# Monsoon season months (Southwest monsoon)
MONSOON_SEASON_MONTHS = {6, 7, 8, 9}

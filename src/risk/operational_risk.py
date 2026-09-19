"""
Charter-AI — Operational Risk Assessor.

Evaluates route-specific and vessel operational risks using
static scoring matrices.

RULE-BASED: Lookup tables and configurable risk factors.
"""

from dataclasses import dataclass

from src.utils.constants import RiskLevel, risk_level_from_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Route risk factors by origin region
# Higher score = more operational risk (piracy, chokepoint delays, political)
ROUTE_RISK_FACTORS = {
    "AUS": 5,   # Low risk — open ocean, stable corridor
    "IDN": 15,  # Moderate — strait transit, regulatory variability
    "USA": 10,  # Low-moderate — long voyage, but stable
    "MOZ": 25,  # Moderate-high — Mozambique Channel piracy risk, infrastructure
    "RUS": 30,  # High — sanctions risk, insurance surcharges, routing constraints
    "ZAF": 10,  # Low-moderate — well-established coal export infrastructure
}


@dataclass
class OperationalRiskAssessment:
    """Operational risk evaluation result."""
    score: float  # 0-100
    level: RiskLevel
    route_risk: float
    detail: str = ""


class OperationalRiskAssessor:
    """
    Evaluates operational risk based on route and vessel factors.

    Scoring components:
    1. Route risk (origin region) → 0-40 points
    2. Voyage length risk (longer = more exposure) → 0-30 points
    3. Strait/chokepoint transit risk → 0-30 points
    """

    def assess(
        self,
        origin_port_id: str,
        sailing_distance_nm: float,
        routing_note: str = "",
    ) -> OperationalRiskAssessment:
        """
        Compute operational risk score.

        Args:
            origin_port_id: Origin port identifier (prefix determines region).
            sailing_distance_nm: Estimated sailing distance in nautical miles.
            routing_note: Free-text routing description for chokepoint detection.

        Returns:
            OperationalRiskAssessment with score and breakdown.
        """
        score = 0.0
        details = []

        # 1. Route risk by origin region (0-40)
        region = origin_port_id.split("_")[0] if "_" in origin_port_id else origin_port_id[:3]
        route_base = ROUTE_RISK_FACTORS.get(region, 15)
        route_score = min(40.0, route_base * 1.33)  # Scale to 0-40
        score += route_score
        details.append(f"Origin region {region}: base risk {route_base} → {route_score:.0f}/40")

        # 2. Voyage length (0-30)
        # Longer voyages = more weather exposure, fuel cost variability, delay risk
        if sailing_distance_nm < 3000:
            length_score = 5.0
        elif sailing_distance_nm < 5000:
            length_score = 10.0
        elif sailing_distance_nm < 8000:
            length_score = 18.0
        elif sailing_distance_nm < 12000:
            length_score = 24.0
        else:
            length_score = 30.0
        score += length_score
        details.append(f"Voyage {sailing_distance_nm:.0f}nm → {length_score:.0f}/30")

        # 3. Chokepoint / strait risk (0-30)
        routing_lower = routing_note.lower()
        chokepoint_score = 0.0
        if "suez" in routing_lower:
            chokepoint_score += 15.0
        if "sunda" in routing_lower or "lombok" in routing_lower:
            chokepoint_score += 8.0
        if "malacca" in routing_lower:
            chokepoint_score += 12.0
        if "cape" in routing_lower and "good hope" in routing_lower:
            chokepoint_score += 5.0
        chokepoint_score = min(30.0, chokepoint_score)
        score += chokepoint_score
        if chokepoint_score > 0:
            details.append(f"Chokepoint/strait risk → {chokepoint_score:.0f}/30")

        score = min(100.0, score)
        level = risk_level_from_score(score)

        return OperationalRiskAssessment(
            score=round(score, 1),
            level=level,
            route_risk=route_base,
            detail="; ".join(details),
        )

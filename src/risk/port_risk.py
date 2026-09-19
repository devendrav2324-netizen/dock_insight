"""
Charter-AI — Port Risk Assessor.

Evaluates congestion severity, berth availability, and turnaround risk
at destination ports.

HYBRID: Rule-based thresholds + future ML congestion forecast.
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.constants import RiskLevel, risk_level_from_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortRiskAssessment:
    """Port risk evaluation result."""
    port_id: str
    score: float  # 0-100
    level: RiskLevel
    vessels_waiting: Optional[int] = None
    avg_waiting_days: Optional[float] = None
    berth_occupancy_pct: Optional[float] = None
    detail: str = ""


class PortRiskAssessor:
    """
    Evaluates port congestion and turnaround risk.

    Scoring components:
    1. Current berth occupancy % → 0-35 points
       - <60%: 0-5 (low), 60-80%: 5-20, 80-95%: 20-30, >95%: 30-35
    2. Average waiting time → 0-35 points
       - <1 day: 0-5, 1-3 days: 5-15, 3-7 days: 15-25, >7 days: 25-35
    3. Vessels waiting in queue → 0-30 points
       - <5: 0-5, 5-10: 5-15, 10-20: 15-25, >20: 25-30
    """

    def assess(
        self,
        port_id: str,
        vessels_waiting: Optional[int] = None,
        avg_waiting_days: Optional[float] = None,
        berth_occupancy_pct: Optional[float] = None,
    ) -> PortRiskAssessment:
        """
        Compute port risk score from congestion metrics.

        Args:
            port_id: Port identifier.
            vessels_waiting: Number of vessels at anchor.
            avg_waiting_days: Average waiting time in days.
            berth_occupancy_pct: Berth utilization percentage.

        Returns:
            PortRiskAssessment with score and breakdown.
        """
        score = 0.0
        details = []

        # 1. Berth occupancy (0-35)
        if berth_occupancy_pct is not None:
            if berth_occupancy_pct < 60:
                occ_score = berth_occupancy_pct / 12  # 0-5
            elif berth_occupancy_pct < 80:
                occ_score = 5 + (berth_occupancy_pct - 60) * 0.75  # 5-20
            elif berth_occupancy_pct < 95:
                occ_score = 20 + (berth_occupancy_pct - 80) * 0.67  # 20-30
            else:
                occ_score = 30 + min(5, (berth_occupancy_pct - 95))  # 30-35
            score += occ_score
            details.append(f"Berth occupancy {berth_occupancy_pct:.0f}% → {occ_score:.0f}/35")

        # 2. Waiting time (0-35)
        if avg_waiting_days is not None:
            if avg_waiting_days < 1:
                wait_score = avg_waiting_days * 5  # 0-5
            elif avg_waiting_days < 3:
                wait_score = 5 + (avg_waiting_days - 1) * 5  # 5-15
            elif avg_waiting_days < 7:
                wait_score = 15 + (avg_waiting_days - 3) * 2.5  # 15-25
            else:
                wait_score = 25 + min(10, (avg_waiting_days - 7) * 1.4)  # 25-35
            score += wait_score
            details.append(f"Avg wait {avg_waiting_days:.1f} days → {wait_score:.0f}/35")

        # 3. Queue depth (0-30)
        if vessels_waiting is not None:
            if vessels_waiting < 5:
                queue_score = vessels_waiting  # 0-5
            elif vessels_waiting < 10:
                queue_score = 5 + (vessels_waiting - 5) * 2  # 5-15
            elif vessels_waiting < 20:
                queue_score = 15 + (vessels_waiting - 10)  # 15-25
            else:
                queue_score = 25 + min(5, (vessels_waiting - 20) * 0.5)  # 25-30
            score += queue_score
            details.append(f"{vessels_waiting} vessels waiting → {queue_score:.0f}/30")

        score = min(100.0, score)
        level = risk_level_from_score(score)

        return PortRiskAssessment(
            port_id=port_id,
            score=round(score, 1),
            level=level,
            vessels_waiting=vessels_waiting,
            avg_waiting_days=avg_waiting_days,
            berth_occupancy_pct=berth_occupancy_pct,
            detail="; ".join(details),
        )

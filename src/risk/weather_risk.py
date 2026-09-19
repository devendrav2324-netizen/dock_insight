"""
Charter-AI — Weather Risk Assessor.

Evaluates cyclone exposure, monsoon disruption, and sea-state warnings
for East Coast Indian ports.

RULE-BASED: Seasonal cyclone calendar + weather thresholds + event data.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.utils.constants import (
    CYCLONE_SEASON_MONTHS,
    MONSOON_SEASON_MONTHS,
    RiskLevel,
    risk_level_from_score,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WeatherRiskAssessment:
    """Weather risk evaluation result."""
    port_id: str
    score: float  # 0-100
    level: RiskLevel
    is_cyclone_season: bool
    is_monsoon_season: bool
    wind_speed_kmh: Optional[float] = None
    wave_height_m: Optional[float] = None
    cyclone_alert: Optional[str] = None
    detail: str = ""


class WeatherRiskAssessor:
    """
    Evaluates weather risk for maritime operations.

    Scoring components:
    1. Seasonal risk (cyclone/monsoon calendar) → 0-25 points
    2. Current wind speed → 0-25 points
       - <20 kmh: 0-5, 20-40: 5-15, 40-60: 15-20, >60: 20-25
    3. Wave height → 0-25 points
       - <1.5m: 0-5, 1.5-3m: 5-15, 3-5m: 15-20, >5m: 20-25
    4. Cyclone alert level → 0-25 points
       - None: 0, Watch: 10, Warning: 20, Severe: 25
    """

    # Cyclone alert score mapping
    CYCLONE_ALERT_SCORES = {
        "None": 0,
        "none": 0,
        "Watch": 10,
        "watch": 10,
        "Warning": 20,
        "warning": 20,
        "Severe": 25,
        "severe": 25,
    }

    def assess(
        self,
        port_id: str,
        target_date: date,
        wind_speed_kmh: Optional[float] = None,
        wave_height_m: Optional[float] = None,
        cyclone_alert_level: Optional[str] = None,
    ) -> WeatherRiskAssessment:
        """
        Compute weather risk score.

        Args:
            port_id: Port identifier.
            target_date: Date of intended arrival/operation.
            wind_speed_kmh: Current or forecast wind speed.
            wave_height_m: Current or forecast wave height.
            cyclone_alert_level: Active cyclone alert level.

        Returns:
            WeatherRiskAssessment with score and breakdown.
        """
        score = 0.0
        details = []
        month = target_date.month

        is_cyclone = month in CYCLONE_SEASON_MONTHS
        is_monsoon = month in MONSOON_SEASON_MONTHS

        # 1. Seasonal component (0-25)
        seasonal_score = 0.0
        if is_cyclone and is_monsoon:
            seasonal_score = 25.0
            details.append("Peak cyclone + monsoon season → 25/25")
        elif is_cyclone:
            seasonal_score = 20.0
            details.append("Cyclone season (Bay of Bengal) → 20/25")
        elif is_monsoon:
            seasonal_score = 15.0
            details.append("Southwest monsoon season → 15/25")
        else:
            seasonal_score = 5.0
            details.append("Calm season → 5/25")
        score += seasonal_score

        # 2. Wind speed (0-25)
        if wind_speed_kmh is not None:
            if wind_speed_kmh < 20:
                wind_score = wind_speed_kmh / 4
            elif wind_speed_kmh < 40:
                wind_score = 5 + (wind_speed_kmh - 20) * 0.5
            elif wind_speed_kmh < 60:
                wind_score = 15 + (wind_speed_kmh - 40) * 0.25
            else:
                wind_score = 20 + min(5, (wind_speed_kmh - 60) * 0.1)
            score += wind_score
            details.append(f"Wind {wind_speed_kmh:.0f} km/h → {wind_score:.0f}/25")

        # 3. Wave height (0-25)
        if wave_height_m is not None:
            if wave_height_m < 1.5:
                wave_score = wave_height_m * 3.33
            elif wave_height_m < 3.0:
                wave_score = 5 + (wave_height_m - 1.5) * 6.67
            elif wave_height_m < 5.0:
                wave_score = 15 + (wave_height_m - 3.0) * 2.5
            else:
                wave_score = 20 + min(5, (wave_height_m - 5.0) * 1.0)
            score += wave_score
            details.append(f"Waves {wave_height_m:.1f}m → {wave_score:.0f}/25")

        # 4. Cyclone alert (0-25)
        if cyclone_alert_level:
            alert_score = self.CYCLONE_ALERT_SCORES.get(cyclone_alert_level, 0)
            score += alert_score
            details.append(f"Cyclone alert: {cyclone_alert_level} → {alert_score}/25")

        score = min(100.0, score)
        level = risk_level_from_score(score)

        return WeatherRiskAssessment(
            port_id=port_id,
            score=round(score, 1),
            level=level,
            is_cyclone_season=is_cyclone,
            is_monsoon_season=is_monsoon,
            wind_speed_kmh=wind_speed_kmh,
            wave_height_m=wave_height_m,
            cyclone_alert=cyclone_alert_level,
            detail="; ".join(details),
        )

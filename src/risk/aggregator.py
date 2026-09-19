"""
Charter-AI — Risk Aggregator.

Combines market, port, weather, and operational risk scores into
a composite risk assessment with dimension breakdown.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.risk.market_risk import MarketRiskAssessment
from src.risk.operational_risk import OperationalRiskAssessment
from src.risk.port_risk import PortRiskAssessment
from src.risk.weather_risk import WeatherRiskAssessment
from src.utils.constants import DEFAULT_RISK_WEIGHTS, RiskLevel, risk_level_from_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskDimensionSummary:
    """Summary of a single risk dimension."""
    score: float
    level: str
    detail: str


@dataclass
class CompositeRiskAssessment:
    """Aggregated risk assessment across all dimensions."""
    composite_score: float  # 0-100 weighted average
    level: RiskLevel
    breakdown: Dict[str, RiskDimensionSummary] = field(default_factory=dict)
    dominant_risk: Optional[str] = None  # Which dimension contributes most
    recommendation: str = ""


class RiskAggregator:
    """
    Weighted aggregation of all risk dimensions.

    Default weights:
        market:      30%
        port:        25%
        weather:     25%
        operational: 20%
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_RISK_WEIGHTS.copy()

    def aggregate(
        self,
        market: Optional[MarketRiskAssessment] = None,
        port: Optional[PortRiskAssessment] = None,
        weather: Optional[WeatherRiskAssessment] = None,
        operational: Optional[OperationalRiskAssessment] = None,
    ) -> CompositeRiskAssessment:
        """
        Combine individual risk assessments into a composite score.

        Any dimension can be None (omitted) — weights are renormalized
        across available dimensions.
        """
        scores = {}
        breakdown = {}

        if market:
            scores["market"] = market.score
            breakdown["market"] = RiskDimensionSummary(
                score=market.score, level=market.level.value, detail=market.detail
            )
        if port:
            scores["port"] = port.score
            breakdown["port"] = RiskDimensionSummary(
                score=port.score, level=port.level.value, detail=port.detail
            )
        if weather:
            scores["weather"] = weather.score
            breakdown["weather"] = RiskDimensionSummary(
                score=weather.score, level=weather.level.value, detail=weather.detail
            )
        if operational:
            scores["operational"] = operational.score
            breakdown["operational"] = RiskDimensionSummary(
                score=operational.score,
                level=operational.level.value,
                detail=operational.detail,
            )

        if not scores:
            return CompositeRiskAssessment(
                composite_score=0.0,
                level=RiskLevel.LOW,
                recommendation="No risk data available.",
            )

        # Renormalize weights for available dimensions
        available_weight = sum(self.weights[k] for k in scores)
        composite = sum(
            scores[k] * self.weights[k] / available_weight for k in scores
        )
        composite = min(100.0, composite)

        # Find dominant risk
        dominant = max(scores, key=scores.get) if scores else None

        # Generate recommendation
        level = risk_level_from_score(composite)
        recommendation = self._generate_recommendation(level, dominant, scores)

        return CompositeRiskAssessment(
            composite_score=round(composite, 1),
            level=level,
            breakdown=breakdown,
            dominant_risk=dominant,
            recommendation=recommendation,
        )

    @staticmethod
    def _generate_recommendation(
        level: RiskLevel, dominant: Optional[str], scores: Dict[str, float]
    ) -> str:
        """Generate a human-readable risk recommendation."""
        if level == RiskLevel.LOW:
            return "Low overall risk. Conditions are favorable for chartering."
        elif level == RiskLevel.MODERATE:
            return (
                f"Moderate risk, driven primarily by {dominant} risk "
                f"(score: {scores.get(dominant, 0):.0f}/100). "
                f"Proceed with standard due diligence."
            )
        elif level == RiskLevel.HIGH:
            return (
                f"High risk. {dominant.title()} risk is elevated at "
                f"{scores.get(dominant, 0):.0f}/100. Consider delaying, "
                f"alternative routing, or risk mitigation measures."
            )
        else:  # CRITICAL
            return (
                f"CRITICAL risk level. {dominant.title()} risk at "
                f"{scores.get(dominant, 0):.0f}/100. Strong recommendation "
                f"to postpone or seek alternative arrangements."
            )

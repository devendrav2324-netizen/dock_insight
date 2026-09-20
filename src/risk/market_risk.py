"""
DockInsights — Market Risk Assessor.

Evaluates freight rate volatility, BDI trend instability, and
coal price risk.

HYBRID: Statistical volatility (deterministic) + ML regime probability.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import RiskLevel, risk_level_from_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketRiskAssessment:
    """Market risk evaluation result."""
    score: float  # 0-100
    level: RiskLevel
    volatility_30d: Optional[float] = None  # Annualized volatility
    rate_trend: Optional[str] = None  # "rising", "falling", "stable"
    bdi_momentum: Optional[str] = None
    detail: str = ""


class MarketRiskAssessor:
    """
    Evaluates market risk from freight rate and economic indicator data.

    Scoring components:
    1. Historical volatility (σ of log-returns, 30-day window) → 0-40 points
    2. Rate trend deviation from seasonal norm → 0-30 points
    3. Coal price instability → 0-15 points
    4. BDI momentum divergence → 0-15 points
    """

    def assess(
        self,
        rate_series: pd.Series,
        bdi_series: Optional[pd.Series] = None,
        coal_price_series: Optional[pd.Series] = None,
    ) -> MarketRiskAssessment:
        """
        Compute market risk score.

        Args:
            rate_series: Historical freight rates (time-indexed, ≥30 points).
            bdi_series: BDI proxy index (optional).
            coal_price_series: Newcastle coal price (optional).

        Returns:
            MarketRiskAssessment with score and breakdown.
        """
        score = 0.0
        details = []

        # 1. Volatility component (0-40)
        if len(rate_series) >= 30:
            log_returns = np.log(rate_series / rate_series.shift(1)).dropna()
            vol_30d = float(log_returns.tail(30).std() * np.sqrt(252))  # Annualized
            # Map volatility to score: 0-20% vol → 0-10 score, 20-60%+ → 10-40
            vol_score = min(40.0, vol_30d * 100)
            score += vol_score
            details.append(f"30d annualized volatility: {vol_30d:.1%} → {vol_score:.0f}/40")
        else:
            details.append("Insufficient data for volatility calculation")

        # 2. Trend component (0-30) — placeholder
        # TODO: Compare current rate vs. seasonal baseline
        rate_change_30d = (
            (rate_series.iloc[-1] - rate_series.iloc[-30]) / rate_series.iloc[-30] * 100
            if len(rate_series) >= 30
            else 0
        )
        trend_score = min(30.0, abs(rate_change_30d))
        score += trend_score
        trend_dir = "rising" if rate_change_30d > 0 else "falling" if rate_change_30d < 0 else "stable"
        details.append(f"30d rate change: {rate_change_30d:+.1f}% ({trend_dir}) → {trend_score:.0f}/30")

        # 3-4. Coal/BDI placeholders (0-15 each)
        # TODO: Implement coal price and BDI momentum scoring

        score = min(100.0, score)
        level = risk_level_from_score(score)

        return MarketRiskAssessment(
            score=round(score, 1),
            level=level,
            volatility_30d=vol_30d if len(rate_series) >= 30 else None,
            rate_trend=trend_dir,
            detail="; ".join(details),
        )

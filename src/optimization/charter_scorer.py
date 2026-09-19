"""
Charter-AI — Multi-Criteria Charter Scorer.

Ranks chartering options (vessel class × timing × contract type) on
a weighted multi-criteria basis.

This is a DETERMINISTIC component with configurable weights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.constants import DEFAULT_CHARTER_SCORE_WEIGHTS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CharterOption:
    """A single chartering option to be scored."""
    vessel_class: str
    contract_type: str
    booking_date: str  # ISO date
    estimated_cost_per_tonne: float
    risk_score: float  # 0-100 composite
    schedule_reliability_score: float  # 0-100
    flexibility_score: float  # 0-100


@dataclass
class ScoredOption:
    """A chartering option with its computed score."""
    option: CharterOption
    total_score: float  # 0-100, higher = better
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank: int = 0


class CharterScorer:
    """
    Weighted multi-criteria scorer for chartering options.

    Default weights (configurable):
        - Cost:                 40%
        - Risk:                 25%
        - Schedule Reliability: 20%
        - Flexibility:          15%

    Scoring normalization:
        - Cost: Inverted (lower cost = higher score)
        - Risk: Inverted (lower risk = higher score)
        - Schedule Reliability: Direct (higher = better)
        - Flexibility: Direct (higher = better)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_CHARTER_SCORE_WEIGHTS.copy()

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Charter score weights sum to {total:.2f}, expected 1.0. "
                f"Normalizing."
            )
            for k in self.weights:
                self.weights[k] /= total

    def score(self, options: List[CharterOption]) -> List[ScoredOption]:
        """
        Score and rank a list of chartering options.

        Args:
            options: List of CharterOption objects to evaluate.

        Returns:
            List of ScoredOption objects, sorted by score descending.
        """
        if not options:
            return []

        # Extract raw values for normalization
        costs = [o.estimated_cost_per_tonne for o in options]
        risks = [o.risk_score for o in options]
        reliabilities = [o.schedule_reliability_score for o in options]
        flexibilities = [o.flexibility_score for o in options]

        scored = []
        for opt in options:
            breakdown = {}

            # Cost score: inverted and normalized (0-100)
            cost_score = self._normalize_inverted(
                opt.estimated_cost_per_tonne, costs
            )
            breakdown["cost"] = cost_score

            # Risk score: inverted (lower risk = higher score)
            risk_score = self._normalize_inverted(opt.risk_score, risks)
            breakdown["risk"] = risk_score

            # Schedule reliability: direct
            reliability_score = self._normalize_direct(
                opt.schedule_reliability_score, reliabilities
            )
            breakdown["schedule_reliability"] = reliability_score

            # Flexibility: direct
            flexibility_score = self._normalize_direct(
                opt.flexibility_score, flexibilities
            )
            breakdown["flexibility"] = flexibility_score

            # Weighted total
            total = sum(
                self.weights[k] * breakdown[k] for k in self.weights
            )

            scored.append(
                ScoredOption(
                    option=opt,
                    total_score=round(total, 2),
                    score_breakdown={k: round(v, 2) for k, v in breakdown.items()},
                )
            )

        # Sort descending by score and assign ranks
        scored.sort(key=lambda s: s.total_score, reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1

        return scored

    @staticmethod
    def _normalize_direct(value: float, all_values: List[float]) -> float:
        """Normalize to 0-100 where higher raw value = higher score."""
        mn, mx = min(all_values), max(all_values)
        if mx == mn:
            return 50.0
        return ((value - mn) / (mx - mn)) * 100

    @staticmethod
    def _normalize_inverted(value: float, all_values: List[float]) -> float:
        """Normalize to 0-100 where lower raw value = higher score."""
        mn, mx = min(all_values), max(all_values)
        if mx == mn:
            return 50.0
        return ((mx - value) / (mx - mn)) * 100

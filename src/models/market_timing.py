"""
Charter-AI — Market Timing Engine (Phase 7).

Recommends whether a dry-bulk charterer should:
- BOOK_NOW
- WAIT
- MONITOR
- START_NEGOTIATION
- HYBRID_BOOKING

Calculates the net expected economic benefit of waiting:
Expected waiting benefit =
  expected future freight savings
  minus risk-adjusted probability of rate increase
  minus deadline risk
  minus vessel availability risk
  minus port congestion penalty

Takes into account forecast quantiles (P10, P50, P90), forecast confidence,
market momentum, freight volatility, tonnage supply tightness, and delivery deadlines.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TimingAction(str, Enum):
    """Recommended timing actions for charterers."""
    BOOK_NOW = "BOOK_NOW"
    WAIT = "WAIT"
    MONITOR = "MONITOR"
    START_NEGOTIATION = "START_NEGOTIATION"
    HYBRID_BOOKING = "HYBRID_BOOKING"


@dataclass
class MarketTimingInputs:
    """
    Inputs for market timing evaluation.
    """
    current_freight_rate: float
    forecast_rate: float
    p10_forecast: float
    p50_forecast: float
    p90_forecast: float
    forecast_confidence: float = 0.85
    market_momentum: float = 0.0  # Momentum indicator ($/t or % over recent period)
    freight_volatility: float = 1.5  # Freight rate volatility ($/t)
    vessel_availability: str = "BALANCED"  # "TIGHT", "BALANCED", "SURPLUS" or numeric float
    congestion_forecast: float = 2.5  # Predicted waiting time at destination in days
    cargo_deadline: Optional[Union[datetime, date, float]] = None  # Absolute deadline or days allowed
    required_delivery_date: Optional[Union[datetime, date]] = None
    current_date: Optional[Union[datetime, date]] = None
    cargo_quantity_t: float = 75000.0  # Standard shipment volume
    route_voyage_days: float = 18.0  # Baseline sailing + handling duration


@dataclass
class MarketTimingResult:
    """
    Outcome of market timing evaluation.
    """
    recommendation: str
    recommended_booking_window: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    expected_current_cost: float
    expected_future_cost: float
    expected_savings: float
    deadline_risk: float
    vessel_availability_risk: float
    confidence: float
    reasons: List[str]

    # Internal analytical metrics
    net_waiting_benefit: float = 0.0
    rate_increase_risk: float = 0.0
    congestion_penalty: float = 0.0
    schedule_buffer_days: float = 0.0
    uncertainty_spread: float = 0.0
    market_momentum: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Returns the canonical dictionary specified by Phase 7 requirements."""
        return {
            "recommendation": self.recommendation,
            "recommended_booking_window": self.recommended_booking_window,
            "expected_current_cost": round(self.expected_current_cost, 2),
            "expected_future_cost": round(self.expected_future_cost, 2),
            "expected_savings": round(self.expected_savings, 2),
            "deadline_risk": round(self.deadline_risk, 2),
            "vessel_availability_risk": round(self.vessel_availability_risk, 2),
            "confidence": round(self.confidence, 4),
            "market_momentum": round(self.market_momentum, 2),
            "reasons": self.reasons,
        }



class MarketTimingEngine:
    """
    Evaluates expected economic benefits and operational risks to recommend market entry timing.
    """

    def evaluate_timing(self, inputs: MarketTimingInputs) -> MarketTimingResult:
        """
        Calculates expected waiting benefits, evaluates deadline and tonnage availability risks,
        and generates an explainable timing recommendation.
        """
        # ---------------------------------------------------------
        # 1. Timeline & Dates Normalization
        # ---------------------------------------------------------
        today = self._normalize_date(inputs.current_date) or datetime.now().date()
        target_delivery = self._normalize_date(inputs.required_delivery_date)
        if target_delivery is None and isinstance(inputs.cargo_deadline, (date, datetime)):
            target_delivery = self._normalize_date(inputs.cargo_deadline)

        if target_delivery is not None:
            allowed_days = max(1.0, float((target_delivery - today).days))
        elif isinstance(inputs.cargo_deadline, (int, float)):
            allowed_days = max(1.0, float(inputs.cargo_deadline))
        else:
            allowed_days = 40.0  # Safe default operational window

        # Total required operational turnaround time (voyage duration + congestion wait)
        total_required_time = inputs.route_voyage_days + max(0.0, inputs.congestion_forecast)
        schedule_buffer = allowed_days - total_required_time

        # ---------------------------------------------------------
        # 2. Freight Economics & Rate Expectations
        # ---------------------------------------------------------
        current_cost = inputs.current_freight_rate * inputs.cargo_quantity_t
        future_cost = inputs.forecast_rate * inputs.cargo_quantity_t
        # Positive expected_savings means future cost is lower (rates falling)
        expected_savings = current_cost - future_cost
        rate_delta_pct = ((inputs.forecast_rate - inputs.current_freight_rate) / max(0.1, inputs.current_freight_rate)) * 100.0


        # ---------------------------------------------------------
        # 3. Quantile Uncertainty & Rate Increase Risk
        # ---------------------------------------------------------
        p10 = inputs.p10_forecast
        p50 = inputs.p50_forecast
        p90 = inputs.p90_forecast
        spread = max(0.1, p90 - p10)
        rel_uncertainty = spread / max(1.0, inputs.forecast_rate)

        # Estimate probability that future rate exceeds current rate P(R_fut > R_curr)
        if inputs.current_freight_rate <= p10:
            prob_increase = 0.95
        elif inputs.current_freight_rate >= p90:
            prob_increase = 0.05
        else:
            std_est = max(0.1, spread / 2.56)
            z = (inputs.current_freight_rate - p50) / std_est
            prob_increase = max(0.02, min(0.98, 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

        potential_spike = max(0.0, p90 - inputs.current_freight_rate)
        # Rate increase risk penalizes waiting if market could spike
        momentum_factor = 1.0 + max(0.0, inputs.market_momentum * 0.05)
        rate_increase_risk = (
            prob_increase * potential_spike * inputs.cargo_quantity_t * momentum_factor * (1.0 - inputs.forecast_confidence * 0.4)
        )

        # ---------------------------------------------------------
        # 4. Deadline Risk Penalty
        # ---------------------------------------------------------
        if schedule_buffer <= 0.0:
            # Deadline is already breached or zero buffer — waiting is fatal
            deadline_risk = current_cost * 0.35
        elif schedule_buffer <= 3.0:
            # Critical deadline (< 3 days buffer)
            deadline_risk = current_cost * 0.20 * ((3.0 - schedule_buffer) / 3.0) + 15000.0
        elif schedule_buffer <= 7.0:
            # Moderate deadline pressure
            deadline_risk = current_cost * 0.06 * ((7.0 - schedule_buffer) / 4.0)
        else:
            deadline_risk = 0.0

        # ---------------------------------------------------------
        # 5. Vessel Availability Risk
        # ---------------------------------------------------------
        avail_str = str(inputs.vessel_availability).upper()
        if avail_str in ["TIGHT", "LOW", "SCARCE"] or (isinstance(inputs.vessel_availability, (int, float)) and inputs.vessel_availability < 0.35):
            availability_factor = 0.05  # 5% risk premium for scarce tonnage
            vessel_availability_risk = current_cost * availability_factor
            avail_label = "TIGHT"
        elif avail_str in ["BALANCED", "MODERATE"] or (isinstance(inputs.vessel_availability, (int, float)) and inputs.vessel_availability < 0.70):
            availability_factor = 0.015
            vessel_availability_risk = current_cost * availability_factor
            avail_label = "BALANCED"
        else:
            availability_factor = 0.0
            vessel_availability_risk = 0.0
            avail_label = "SURPLUS"

        # ---------------------------------------------------------
        # 6. Congestion Penalty on Schedule Buffer
        # ---------------------------------------------------------
        congestion_penalty = 0.0
        if inputs.congestion_forecast > 3.5:
            # Extended port congestion adds risk of demurrage and eats into buffers
            congestion_penalty = (inputs.congestion_forecast - 3.5) * 12000.0

        # ---------------------------------------------------------
        # 7. Net Waiting Benefit & Dynamic Decision
        # ---------------------------------------------------------
        net_waiting_benefit = (
            expected_savings
            - rate_increase_risk
            - deadline_risk
            - vessel_availability_risk
            - congestion_penalty
        )

        # Confidence adjusted for forecast uncertainty and volatility
        effective_confidence = inputs.forecast_confidence * max(0.5, 1.0 - (rel_uncertainty * 0.5))
        if inputs.freight_volatility > 3.0:
            effective_confidence *= 0.85

        reasons: List[str] = []

        # Decision Logic
        # Condition A: Critical Deadline takes absolute precedence
        if schedule_buffer <= 2.5:
            action = TimingAction.BOOK_NOW
            reasons.append(
                f"Delivery deadline is critical with only {schedule_buffer:.1f} days schedule buffer remaining. "
                f"Waiting risks missing contractual delivery date."
            )
        # Condition B: High Uncertainty / High Volatility -> HYBRID or MONITOR
        elif (rel_uncertainty > 0.30 or effective_confidence < 0.60) and abs(rate_delta_pct) < 6.0:
            if inputs.cargo_quantity_t >= 65000:
                action = TimingAction.HYBRID_BOOKING
                reasons.append(
                    f"High forecast uncertainty (P10: ${p10:.2f} vs P90: ${p90:.2f}/t, spread ${spread:.2f}/t). "
                    f"Recommend a hybrid strategy: book 50% base volume now and float/index-link remainder."
                )
            else:
                action = TimingAction.MONITOR
                reasons.append(
                    f"Wide forecast uncertainty spread (${spread:.2f}/t) with confidence {effective_confidence*100:.0f}%. "
                    f"Monitor daily arrivals and FFA fixtures before committing."
                )
        # Condition C: Low Vessel Availability Squeeze
        elif avail_label == "TIGHT" and rate_delta_pct >= -1.0:
            if rate_delta_pct >= 4.0:
                action = TimingAction.BOOK_NOW
                reasons.append(
                    f"Vessel availability in the loading basin is TIGHT and freight is forecast to rise by {rate_delta_pct:.1f}%. "
                    f"Book immediately to secure prompt tonnage and avoid spot scarcity surcharges."
                )
            else:
                action = TimingAction.START_NEGOTIATION
                reasons.append(
                    f"Tonnage availability is TIGHT (${vessel_availability_risk:,.0f} availability risk). "
                    f"Start negotiations with vessel owners now to lock in spot capacity ahead of tightening supply."
                )
        # Condition D: Falling Market with Positive Net Benefit -> WAIT
        elif rate_delta_pct <= -2.5 and net_waiting_benefit > 0 and schedule_buffer >= 6.0:
            action = TimingAction.WAIT
            reasons.append(
                f"Freight forecast indicates a {abs(rate_delta_pct):.1f}% decline from ${inputs.current_freight_rate:.2f} to ${inputs.forecast_rate:.2f}/t. "
                f"Expected freight savings of ${expected_savings:,.0f} outweigh delay risks (net benefit: ${net_waiting_benefit:,.0f})."
            )
            reasons.append(f"Sufficient schedule buffer ({schedule_buffer:.1f} days) permits waiting for market trough.")
        # Condition E: Rising Market -> BOOK_NOW or START_NEGOTIATION
        elif rate_delta_pct >= 4.0:
            action = TimingAction.BOOK_NOW
            reasons.append(
                f"14-day freight forecast indicates a {rate_delta_pct:.1f}% expected rate increase from ${inputs.current_freight_rate:.2f} to ${inputs.forecast_rate:.2f}/t. "
                f"Booking now prevents expected cost escalation of ${abs(expected_savings):,.0f}."
            )
        elif rate_delta_pct >= 1.5:
            action = TimingAction.START_NEGOTIATION
            reasons.append(
                f"Forecast indicates moderate upward momentum (+{rate_delta_pct:.1f}%). "
                f"Start negotiation now and target booking within 5 days before rates firm further."
            )
        # Condition F: High Congestion eroding buffer
        elif inputs.congestion_forecast >= 5.0 and schedule_buffer <= 6.0:
            action = TimingAction.BOOK_NOW
            reasons.append(
                f"Severe port congestion forecast ({inputs.congestion_forecast:.1f} waiting days) reduces effective schedule buffer to {schedule_buffer:.1f} days. "
                f"Book immediately to avoid operational laycan delays."
            )
        # Condition G: Balanced / Neutral Market -> MONITOR
        else:
            action = TimingAction.MONITOR
            reasons.append(
                f"Market momentum is relatively neutral ({rate_delta_pct:+.1f}%) with {avail_label.lower()} tonnage supply. "
                f"Ample schedule buffer ({schedule_buffer:.1f} days) allows monitoring market fixtures for 3-5 days."
            )

        # ---------------------------------------------------------
        # 8. Compute Recommended Booking Window
        # ---------------------------------------------------------
        window = self._calculate_booking_window(action, today, schedule_buffer, rate_delta_pct)

        return MarketTimingResult(
            recommendation=action.value,
            recommended_booking_window=window,
            expected_current_cost=round(current_cost, 2),
            expected_future_cost=round(future_cost, 2),
            expected_savings=round(expected_savings, 2),
            deadline_risk=round(deadline_risk, 2),
            vessel_availability_risk=round(vessel_availability_risk, 2),
            confidence=round(effective_confidence, 4),
            reasons=reasons,
            net_waiting_benefit=round(net_waiting_benefit, 2),
            rate_increase_risk=round(rate_increase_risk, 2),
            congestion_penalty=round(congestion_penalty, 2),
            schedule_buffer_days=round(schedule_buffer, 2),
            uncertainty_spread=round(spread, 2),
            market_momentum=round(inputs.market_momentum, 2),
        )


    def _calculate_booking_window(
        self,
        action: TimingAction,
        today: date,
        buffer_days: float,
        rate_delta_pct: float,
    ) -> Dict[str, str]:
        """Calculates precise start and end dates for the recommended booking window."""
        if action == TimingAction.BOOK_NOW:
            start_date = today
            end_date = today + timedelta(days=2)
        elif action == TimingAction.START_NEGOTIATION:
            start_date = today
            end_date = today + timedelta(days=min(5, max(2, int(buffer_days))))
        elif action == TimingAction.WAIT:
            # Target future trough (e.g. 7 to 12 days out, capped by buffer)
            trough_offset = min(12, max(5, int(buffer_days - 3.0)))
            start_date = today + timedelta(days=trough_offset)
            end_date = start_date + timedelta(days=3)
        elif action == TimingAction.HYBRID_BOOKING:
            start_date = today
            end_date = today + timedelta(days=min(4, max(2, int(buffer_days))))
        else:  # MONITOR
            start_date = today + timedelta(days=3)
            end_date = today + timedelta(days=min(7, max(4, int(buffer_days))))

        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }

    def _normalize_date(self, d: Any) -> Optional[date]:
        """Normalizes datetime or date input to date."""
        if d is None:
            return None
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        return None


# =============================================================================
# Backward Compatibility Scaffolds
# =============================================================================

@dataclass
class MarketRegime:
    """Current market regime assessment."""
    regime: str  # "trough", "rising", "peak", "falling"
    confidence: float
    regime_start_date: Optional[date] = None
    expected_duration_days: Optional[int] = None


@dataclass
class TimingRecommendation:
    """Recommended booking window (legacy dataclass)."""
    optimal_window_start: date
    optimal_window_end: date
    urgency: str
    reasoning: str
    current_regime: MarketRegime
    expected_savings_pct: Optional[float] = None


class MarketTimingDetector:
    """
    MarketTimingDetector wrapper providing compatibility with legacy calls.
    Uses MarketTimingEngine internally.
    """

    def __init__(self):
        self.engine = MarketTimingEngine()
        self.is_fitted = True

    def detect_regime(self, rate_series: pd.Series) -> MarketRegime:
        """Classify current market regime from rate series using rolling statistics."""
        if rate_series is None or len(rate_series) < 3:
            return MarketRegime(regime="rising", confidence=0.75)

        recent = rate_series.dropna()
        if len(recent) < 2:
            return MarketRegime(regime="rising", confidence=0.75)

        pct_change = (recent.iloc[-1] - recent.iloc[0]) / max(0.1, recent.iloc[0]) * 100.0
        if pct_change > 3.0:
            regime = "rising"
        elif pct_change < -3.0:
            regime = "falling"
        elif recent.iloc[-1] <= recent.min() * 1.02:
            regime = "trough"
        else:
            regime = "peak"

        return MarketRegime(regime=regime, confidence=0.85)

    def recommend_timing(
        self,
        rate_series: pd.Series,
        forecast_series: Optional[pd.Series] = None,
    ) -> TimingRecommendation:
        """Produce timing recommendation based on rate series and forecast."""
        curr_rate = float(rate_series.iloc[-1]) if len(rate_series) > 0 else 22.0
        fut_rate = float(forecast_series.iloc[-1]) if (forecast_series is not None and len(forecast_series) > 0) else curr_rate * 1.05

        inputs = MarketTimingInputs(
            current_freight_rate=curr_rate,
            forecast_rate=fut_rate,
            p10_forecast=fut_rate * 0.92,
            p50_forecast=fut_rate,
            p90_forecast=fut_rate * 1.08,
            cargo_deadline=30.0,
        )
        res = self.engine.evaluate_timing(inputs)
        start_d = datetime.strptime(res.recommended_booking_window["start"], "%Y-%m-%d").date()
        end_d = datetime.strptime(res.recommended_booking_window["end"], "%Y-%m-%d").date()

        regime = self.detect_regime(rate_series)
        return TimingRecommendation(
            optimal_window_start=start_d,
            optimal_window_end=end_d,
            urgency=res.recommendation.lower(),
            reasoning=" ".join(res.reasons),
            current_regime=regime,
            expected_savings_pct=round((res.expected_savings / max(1.0, res.expected_current_cost)) * 100.0, 2),
        )

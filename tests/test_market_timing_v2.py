"""
Charter-AI — Market Timing Engine Unit & Integration Tests (Phase 7).

Validates:
1. Rising market scenario (expected rate increase -> BOOK_NOW or START_NEGOTIATION)
2. Falling market scenario (expected savings -> WAIT for trough)
3. High uncertainty scenario (wide P10-P90 spread -> HYBRID_BOOKING / MONITOR)
4. Tight deadline scenario (eroded schedule buffer -> BOOK_NOW override)
5. Low vessel availability scenario (tight tonnage supply -> START_NEGOTIATION / BOOK_NOW)
6. High congestion scenario (eroded buffer & demurrage risk -> prompt booking)
7. Net expected economic benefit calculations
8. Explainable reasoning strings
"""

from datetime import date, datetime, timedelta
import pytest

from src.models.market_timing import (
    MarketTimingEngine,
    MarketTimingInputs,
    MarketTimingResult,
    TimingAction,
    MarketTimingDetector,
)


@pytest.fixture
def timing_engine():
    return MarketTimingEngine()


# =============================================================================
# 1. Rising Market Test
# =============================================================================

def test_rising_market_recommends_book_now_or_negotiate(timing_engine):
    """
    In a strongly rising market (+13.6% from $22 to $25/t), waiting causes financial loss.
    Engine must recommend BOOK_NOW with negative expected waiting savings.
    """
    inputs = MarketTimingInputs(
        current_freight_rate=22.0,
        forecast_rate=25.0,
        p10_forecast=24.0,
        p50_forecast=25.0,
        p90_forecast=27.5,
        forecast_confidence=0.88,
        market_momentum=2.5,
        vessel_availability="BALANCED",
        congestion_forecast=2.0,
        cargo_deadline=40.0,
        cargo_quantity_t=75000.0,
    )
    result = timing_engine.evaluate_timing(inputs)

    assert result.recommendation in [TimingAction.BOOK_NOW.value, TimingAction.START_NEGOTIATION.value]
    assert result.expected_savings < 0.0  # Future cost is higher
    assert result.net_waiting_benefit < 0.0
    assert any("increase" in r.lower() or "firm" in r.lower() for r in result.reasons)

    # Booking window starts immediately
    assert result.recommended_booking_window["start"] == datetime.now().date().strftime("%Y-%m-%d")


# =============================================================================
# 2. Falling Market Test
# =============================================================================

def test_falling_market_recommends_wait(timing_engine):
    """
    In a falling market (-18% from $28 to $23/t) with ample deadline buffer,
    waiting yields positive net economic benefit. Engine must recommend WAIT.
    """
    inputs = MarketTimingInputs(
        current_freight_rate=28.0,
        forecast_rate=23.0,
        p10_forecast=21.5,
        p50_forecast=23.0,
        p90_forecast=24.5,
        forecast_confidence=0.90,
        market_momentum=-2.0,
        vessel_availability="SURPLUS",
        congestion_forecast=2.0,
        cargo_deadline=50.0,  # 50 days deadline for 18-day voyage -> 30 days buffer
        cargo_quantity_t=75000.0,
    )
    result = timing_engine.evaluate_timing(inputs)

    assert result.recommendation == TimingAction.WAIT.value
    assert result.expected_savings > 0.0  # Saves $5/t * 75,000 = $375,000
    assert result.net_waiting_benefit > 0.0
    assert any("decline" in r.lower() or "savings" in r.lower() for r in result.reasons)

    # Booking window targets future trough (not today)
    start_date = datetime.strptime(result.recommended_booking_window["start"], "%Y-%m-%d").date()
    assert start_date > datetime.now().date()


# =============================================================================
# 3. High Uncertainty Test
# =============================================================================

def test_high_uncertainty_recommends_hybrid_or_monitor(timing_engine):
    """
    When the P10-P90 spread is wide (e.g. $17 to $31/t) and confidence is reduced,
    engine must recommend HYBRID_BOOKING for large parcels or MONITOR.
    """
    inputs = MarketTimingInputs(
        current_freight_rate=23.0,
        forecast_rate=23.5,
        p10_forecast=17.0,
        p50_forecast=23.5,
        p90_forecast=31.0,  # $14/t spread = 60% relative uncertainty
        forecast_confidence=0.60,
        market_momentum=0.0,
        freight_volatility=3.5,
        vessel_availability="BALANCED",
        cargo_deadline=35.0,
        cargo_quantity_t=80000.0,
    )
    result = timing_engine.evaluate_timing(inputs)

    assert result.recommendation in [TimingAction.HYBRID_BOOKING.value, TimingAction.MONITOR.value]
    assert result.confidence < 0.70
    assert any("uncertainty" in r.lower() or "spread" in r.lower() for r in result.reasons)


# =============================================================================
# 4. Tight Deadline Test
# =============================================================================

def test_tight_deadline_forces_book_now_even_if_market_falling(timing_engine):
    """
    Even if freight is forecast to fall, a critical delivery deadline (<3 days buffer)
    must override savings and force BOOK_NOW to prevent catastrophic delay.
    """
    today = datetime.now().date()
    # 20-day voyage + 2 days congestion = 22 days needed. Deadline is 23 days from now (buffer = 1 day!)
    delivery_date = today + timedelta(days=23)

    inputs = MarketTimingInputs(
        current_freight_rate=26.0,
        forecast_rate=21.0,  # Rates falling sharply!
        p10_forecast=20.0,
        p50_forecast=21.0,
        p90_forecast=23.0,
        forecast_confidence=0.88,
        required_delivery_date=delivery_date,
        current_date=today,
        route_voyage_days=20.0,
        congestion_forecast=2.0,
    )
    result = timing_engine.evaluate_timing(inputs)

    # Must book now despite falling rates
    assert result.recommendation == TimingAction.BOOK_NOW.value
    assert result.deadline_risk > 50000.0  # Substantial deadline risk penalty
    assert any("deadline is critical" in r.lower() for r in result.reasons)


# =============================================================================
# 5. Low Vessel Availability Test
# =============================================================================

def test_low_vessel_availability_triggers_negotiation_or_booking(timing_engine):
    """
    When regional tonnage supply is TIGHT, vessel availability risk increases,
    prompting immediate owner negotiations or booking to avoid being caught without tonnage.
    """
    inputs = MarketTimingInputs(
        current_freight_rate=22.0,
        forecast_rate=22.8,  # Mild upward tilt
        p10_forecast=21.5,
        p50_forecast=22.8,
        p90_forecast=24.5,
        vessel_availability="TIGHT",
        cargo_deadline=40.0,
    )
    result = timing_engine.evaluate_timing(inputs)

    assert result.recommendation in [TimingAction.START_NEGOTIATION.value, TimingAction.BOOK_NOW.value]
    assert result.vessel_availability_risk > 0.0
    assert any("tight" in r.lower() or "negotiat" in r.lower() for r in result.reasons)


# =============================================================================
# 6. High Congestion Test
# =============================================================================

def test_high_congestion_compresses_buffer_and_adds_penalty(timing_engine):
    """
    Severe predicted port congestion (e.g. 7 days wait) expands required port time,
    eroding schedule buffer and adding congestion demurrage penalty.
    """
    today = datetime.now().date()
    delivery_date = today + timedelta(days=30)  # 30 days total

    inputs_low_congestion = MarketTimingInputs(
        current_freight_rate=22.0,
        forecast_rate=22.0,
        p10_forecast=21.0,
        p50_forecast=22.0,
        p90_forecast=23.0,
        congestion_forecast=1.5,
        route_voyage_days=18.0,
        required_delivery_date=delivery_date,
        current_date=today,
    )
    res_low = timing_engine.evaluate_timing(inputs_low_congestion)

    inputs_high_congestion = MarketTimingInputs(
        current_freight_rate=22.0,
        forecast_rate=22.0,
        p10_forecast=21.0,
        p50_forecast=22.0,
        p90_forecast=23.0,
        congestion_forecast=7.5,  # 7.5 days congestion
        route_voyage_days=18.0,
        required_delivery_date=delivery_date,
        current_date=today,
    )
    res_high = timing_engine.evaluate_timing(inputs_high_congestion)

    assert res_high.schedule_buffer_days < res_low.schedule_buffer_days
    assert res_high.congestion_penalty > res_low.congestion_penalty


# =============================================================================
# 7. Backward Compatibility Wrapper Test
# =============================================================================

def test_legacy_market_timing_detector_wrapper():
    """Verify MarketTimingDetector scaffold provides backward compatibility without error."""
    import pandas as pd
    detector = MarketTimingDetector()
    rates = pd.Series([20.0, 21.0, 22.5, 23.0, 24.5])

    regime = detector.detect_regime(rates)
    assert regime.regime in ["rising", "peak", "trough", "falling"]
    assert 0.0 <= regime.confidence <= 1.0

    rec = detector.recommend_timing(rates)
    assert rec.urgency in ["book_now", "wait", "monitor", "start_negotiation", "hybrid_booking"]
    assert rec.optimal_window_start <= rec.optimal_window_end

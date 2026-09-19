"""
Charter-AI — Risk-Aware Contract Strategy Optimization Engine (Phase 9).

Determines the optimal quantitative allocation across:
1. 100% SPOT
2. 100% SHORT_TERM
3. 100% MEDIUM_TERM
4. 75/25 HYBRID (75% term / 25% spot)
5. 60/40 HYBRID (60% term / 40% spot)
6. 50/50 HYBRID (50% term / 50% spot)
7. Configurable custom combinations

Driven by Monte Carlo simulation, evaluating:
- Expected freight cost & total delivered cost
- P10, P50, and P90 cost percentiles
- Price volatility exposure
- Vessel availability exposure
- Delivery schedule risk
- Operational flexibility
- Expected demurrage exposure

Optimizes based on:
    Objective Score = Expected Total Cost + Risk Penalty + Schedule Penalty - Flexibility Preference
Modulated by configurable Risk Tolerance (LOW, MEDIUM, HIGH).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np

from src.risk.monte_carlo import MonteCarloSimulator, CharterPlanInputs, MonteCarloResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Legacy Data Models (Preserved for 100% Backward Compatibility)
# =============================================================================

class ContractStrategy(str, Enum):
    BOOK_NOW = "BOOK_NOW"
    WAIT = "WAIT"
    SPOT = "SPOT"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    HYBRID = "HYBRID"


class ForecastDirection(str, Enum):
    RISING = "RISING"
    FALLING = "FALLING"
    UNCERTAIN = "UNCERTAIN"
    STABLE = "STABLE"


class ForecastUncertainty(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class VesselAvailability(str, Enum):
    ABUNDANT = "ABUNDANT"
    TIGHT = "TIGHT"
    SHORTAGE = "SHORTAGE"


class RiskTolerance(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ContractStrategyType(str, Enum):
    SPOT_100 = "100% SPOT"
    SHORT_TERM_100 = "100% SHORT_TERM"
    MEDIUM_TERM_100 = "100% MEDIUM_TERM"
    HYBRID_75_25 = "75/25 HYBRID"
    HYBRID_60_40 = "60/40 HYBRID"
    HYBRID_50_50 = "50/50 HYBRID"
    CUSTOM = "CUSTOM HYBRID"


@dataclass
class ContractOptimizationInputs:
    current_freight_rate: float
    expected_future_rate: float
    forecast_direction: ForecastDirection
    forecast_uncertainty: ForecastUncertainty
    vessel_availability: VesselAvailability
    congestion_level: float  # 0.0 to 10.0
    overall_risk_score: float  # 0.0 to 100.0
    voyage_cost: float
    number_of_required_voyages: int
    risk_tolerance: Optional[RiskTolerance] = RiskTolerance.MEDIUM


@dataclass
class ContractStrategyRecommendation:
    strategy: ContractStrategy
    recommendation_text: str
    reasons: List[str]


# =============================================================================
# Phase 9 Risk-Aware Quantitative Models
# =============================================================================

@dataclass
class StrategyAllocation:
    name: str
    spot_pct: float
    short_term_pct: float
    medium_term_pct: float

    def __post_init__(self):
        tot = self.spot_pct + self.short_term_pct + self.medium_term_pct
        if tot > 0 and abs(tot - 100.0) > 0.01 and abs(tot - 1.0) < 0.01:
            # Scaled 0 to 1 -> normalize to 0-100
            self.spot_pct *= 100.0
            self.short_term_pct *= 100.0
            self.medium_term_pct *= 100.0


@dataclass
class StrategyEvaluation:
    """Detailed quantitative evaluation of a single contract strategy."""
    strategy_name: str
    spot_percentage: float
    short_term_percentage: float
    medium_term_percentage: float

    expected_freight_cost: float
    expected_total_cost: float
    p10_cost: float
    p50_cost: float
    p90_cost: float

    price_volatility_exposure: float  # (%) spread (P90-P10)/P50
    vessel_availability_exposure: float  # (0-100) availability failure exposure
    delivery_risk: float  # probability of schedule slippage
    flexibility_score: float  # (0.0 to 1.0) operational re-routing/cargo flexibility
    expected_demurrage_exposure: float

    risk_penalty: float
    schedule_penalty: float
    flexibility_preference: float
    objective_score: float  # Minimized total cost + penalties - flexibility

    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "spot_percentage": round(self.spot_percentage, 1),
            "short_term_percentage": round(self.short_term_percentage, 1),
            "medium_term_percentage": round(self.medium_term_percentage, 1),
            "expected_freight_cost": round(self.expected_freight_cost, 2),
            "expected_total_cost": round(self.expected_total_cost, 2),
            "p10_cost": round(self.p10_cost, 2),
            "p50_cost": round(self.p50_cost, 2),
            "p90_cost": round(self.p90_cost, 2),
            "price_volatility_exposure": round(self.price_volatility_exposure, 1),
            "vessel_availability_exposure": round(self.vessel_availability_exposure, 1),
            "delivery_risk": round(self.delivery_risk, 4),
            "flexibility_score": round(self.flexibility_score, 2),
            "expected_demurrage_exposure": round(self.expected_demurrage_exposure, 2),
            "objective_score": round(self.objective_score, 2),
            "reasons": self.reasons,
        }


@dataclass
class RiskAwareContractInputs:
    """Inputs for quantitative Monte Carlo contract strategy optimization."""
    cargo_quantity_t: float
    spot_freight_rate: float
    short_term_freight_rate: Optional[float] = None
    medium_term_freight_rate: Optional[float] = None
    freight_volatility_pct: float = 16.0

    base_bunker_price: float = 650.0
    bunker_volatility_pct: float = 12.0
    sea_distance_nm: float = 4500.0
    service_speed_knots: float = 12.5
    fuel_consumption_t_day: float = 28.0

    expected_wait_days: float = 2.0
    delivery_deadline_days: Optional[float] = 26.0
    number_of_voyages: int = 1
    risk_tolerance: RiskTolerance = RiskTolerance.MEDIUM
    vessel_availability: VesselAvailability = VesselAvailability.TIGHT

    custom_strategies: Optional[List[StrategyAllocation]] = None
    n_simulations: int = 5000
    seed: int = 42


@dataclass
class RiskAwareContractRecommendation:
    """Canonical API output for Phase 9 contract recommendation."""
    recommended_strategy: str
    spot_percentage: float
    short_term_percentage: float
    medium_term_percentage: float
    expected_cost: float
    p90_cost: float
    risk_score: float
    flexibility_score: float
    reasons: List[str]
    evaluated_strategies: List[StrategyEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Matches the exact API response specification required by the user prompt."""
        return {
            "recommended_strategy": self.recommended_strategy,
            "spot_percentage": round(self.spot_percentage, 1),
            "short_term_percentage": round(self.short_term_percentage, 1),
            "medium_term_percentage": round(self.medium_term_percentage, 1),
            "expected_cost": round(self.expected_cost, 2),
            "p90_cost": round(self.p90_cost, 2),
            "risk_score": round(self.risk_score, 1),
            "flexibility_score": round(self.flexibility_score, 2),
            "reasons": self.reasons,
            "evaluated_strategies": [s.to_dict() for s in self.evaluated_strategies],
        }


# =============================================================================
# Risk-Aware Contract Optimizer Implementation
# =============================================================================

class RiskAwareContractOptimizer:
    """
    Evaluates candidate contract portfolios using Monte Carlo simulations
    and selects the optimal contract allocation based on quantitative risk-adjusted utility.
    """

    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()

    def get_standard_strategies(self) -> List[StrategyAllocation]:
        """Returns the standard 6 pre-configured maritime contract strategies."""
        return [
            StrategyAllocation("100% SPOT", spot_pct=100.0, short_term_pct=0.0, medium_term_pct=0.0),
            StrategyAllocation("100% SHORT_TERM", spot_pct=0.0, short_term_pct=100.0, medium_term_pct=0.0),
            StrategyAllocation("100% MEDIUM_TERM", spot_pct=0.0, short_term_pct=0.0, medium_term_pct=100.0),
            StrategyAllocation("75/25 HYBRID", spot_pct=25.0, short_term_pct=75.0, medium_term_pct=0.0),
            StrategyAllocation("60/40 HYBRID", spot_pct=40.0, short_term_pct=60.0, medium_term_pct=0.0),
            StrategyAllocation("50/50 HYBRID", spot_pct=50.0, short_term_pct=50.0, medium_term_pct=0.0),
        ]

    def optimize_contract_strategy(self, inputs: RiskAwareContractInputs) -> RiskAwareContractRecommendation:
        """
        Runs quantitative multi-objective contract strategy optimization.
        """
        # Determine candidate strategies to evaluate
        strategies = self.get_standard_strategies()
        if inputs.custom_strategies:
            strategies.extend(inputs.custom_strategies)

        # Base rate setup
        spot_rate = max(1.0, inputs.spot_freight_rate)
        # Short-term COAs trade at ~2% premium/discount depending on rate stability
        short_rate = (
            inputs.short_term_freight_rate
            if inputs.short_term_freight_rate is not None
            else spot_rate * 1.02
        )
        # Medium-term time charters offer volume discount or fixed stability
        medium_rate = (
            inputs.medium_term_freight_rate
            if inputs.medium_term_freight_rate is not None
            else spot_rate * 0.98
        )

        # Baseline availability probabilities based on market status
        if inputs.vessel_availability == VesselAvailability.SHORTAGE:
            spot_avail_prob = 0.70  # High risk of spot fixture failure
            term_avail_prob = 0.95  # Committed contractual tonnage
        elif inputs.vessel_availability == VesselAvailability.TIGHT:
            spot_avail_prob = 0.88
            term_avail_prob = 0.98
        else:
            spot_avail_prob = 0.98
            term_avail_prob = 0.99

        # Run baseline Monte Carlo simulations for the three pure legs
        # 1. Pure Spot Simulation
        spot_plan = CharterPlanInputs(
            cargo_quantity_t=inputs.cargo_quantity_t,
            base_freight_rate=spot_rate,
            freight_volatility_pct=inputs.freight_volatility_pct,
            base_bunker_price=inputs.base_bunker_price,
            bunker_volatility_pct=inputs.bunker_volatility_pct,
            sea_distance_nm=inputs.sea_distance_nm,
            service_speed_knots=inputs.service_speed_knots,
            fuel_consumption_t_day=inputs.fuel_consumption_t_day,
            expected_wait_days=inputs.expected_wait_days,
            delivery_deadline_days=inputs.delivery_deadline_days,
            vessel_availability_probability=spot_avail_prob,
        )
        mc_spot = self.monte_carlo.run_simulation(
            spot_plan, n_simulations=inputs.n_simulations, seed=inputs.seed
        )

        # 2. Pure Short-Term Simulation (collared rate, 25% volatility)
        short_plan = CharterPlanInputs(
            cargo_quantity_t=inputs.cargo_quantity_t,
            base_freight_rate=short_rate,
            freight_volatility_pct=inputs.freight_volatility_pct * 0.30,  # 70% volatility reduction
            base_bunker_price=inputs.base_bunker_price,
            bunker_volatility_pct=inputs.bunker_volatility_pct,
            sea_distance_nm=inputs.sea_distance_nm,
            service_speed_knots=inputs.service_speed_knots,
            fuel_consumption_t_day=inputs.fuel_consumption_t_day,
            expected_wait_days=max(0.5, inputs.expected_wait_days - 0.5),  # Priority berthing
            delivery_deadline_days=inputs.delivery_deadline_days,
            vessel_availability_probability=term_avail_prob,
        )
        mc_short = self.monte_carlo.run_simulation(
            short_plan, n_simulations=inputs.n_simulations, seed=inputs.seed + 1
        )

        # 3. Pure Medium-Term Simulation (fixed rate, minimal volatility)
        med_plan = CharterPlanInputs(
            cargo_quantity_t=inputs.cargo_quantity_t,
            base_freight_rate=medium_rate,
            freight_volatility_pct=inputs.freight_volatility_pct * 0.08,  # Near zero rate risk
            base_bunker_price=inputs.base_bunker_price,
            bunker_volatility_pct=inputs.bunker_volatility_pct,
            sea_distance_nm=inputs.sea_distance_nm,
            service_speed_knots=inputs.service_speed_knots,
            fuel_consumption_t_day=inputs.fuel_consumption_t_day,
            expected_wait_days=max(0.5, inputs.expected_wait_days - 0.8),  # Long-term berth allocation
            delivery_deadline_days=inputs.delivery_deadline_days,
            vessel_availability_probability=0.995,
        )
        mc_medium = self.monte_carlo.run_simulation(
            med_plan, n_simulations=inputs.n_simulations, seed=inputs.seed + 2
        )

        # Calibrate risk tolerance weights
        if inputs.risk_tolerance == RiskTolerance.LOW:
            lambda_risk = 2.40  # Heavily penalize tail cost (P90) and volatility
            lambda_sched = 1.80  # Heavily penalize late delivery risk
            lambda_flex = 0.25  # Lower value placed on optional flexibility
        elif inputs.risk_tolerance == RiskTolerance.HIGH:
            lambda_risk = 0.20  # Tolerant of rate swings
            lambda_sched = 0.60
            lambda_flex = 2.50  # Highly prize commercial agility and flexibility
        else:  # MEDIUM
            lambda_risk = 1.00
            lambda_sched = 1.00
            lambda_flex = 1.00

        # Evaluate all candidate strategy combinations
        evaluations: List[StrategyEvaluation] = []
        for strat in strategies:
            w_spot = strat.spot_pct / 100.0
            w_short = strat.short_term_pct / 100.0
            w_med = strat.medium_term_pct / 100.0

            # Linear portfolio blending of expected costs and quantiles
            exp_cost = (
                (w_spot * mc_spot.expected_cost)
                + (w_short * mc_short.expected_cost)
                + (w_med * mc_medium.expected_cost)
            )
            p10 = (w_spot * mc_spot.p10_cost) + (w_short * mc_short.p10_cost) + (w_med * mc_medium.p10_cost)
            p50 = (w_spot * mc_spot.p50_cost) + (w_short * mc_short.p50_cost) + (w_med * mc_medium.p50_cost)
            p90 = (w_spot * mc_spot.p90_cost) + (w_short * mc_short.p90_cost) + (w_med * mc_medium.p90_cost)

            # Freight cost portion only
            exp_freight = (
                (w_spot * spot_rate * inputs.cargo_quantity_t)
                + (w_short * short_rate * inputs.cargo_quantity_t)
                + (w_med * medium_rate * inputs.cargo_quantity_t)
            )

            # Volatility exposure: normalized quantile spread
            vol_exposure = ((p90 - p10) / max(1.0, p50)) * 100.0

            # Vessel availability exposure (0-100 score)
            # Spot bears full market availability risk; term contracts hedge availability
            avail_risk = (
                (w_spot * (1.0 - spot_avail_prob) * 100.0 * 2.5)
                + (w_short * (1.0 - term_avail_prob) * 100.0 * 1.5)
                + (w_med * 0.5)
            )
            avail_risk = min(100.0, avail_risk)

            # Delivery risk
            delivery_risk = (
                (w_spot * mc_spot.late_delivery_probability)
                + (w_short * mc_short.late_delivery_probability)
                + (w_med * mc_medium.late_delivery_probability)
            )

            # Operational flexibility: Spot=1.0, Short-term=0.60, Medium-term=0.25
            flex_score = (w_spot * 1.0) + (w_short * 0.60) + (w_med * 0.25)

            # Demurrage exposure
            dem_exposure = (
                (w_spot * mc_spot.expected_demurrage_cost)
                + (w_short * mc_short.expected_demurrage_cost)
                + (w_med * mc_medium.expected_demurrage_cost)
            )

            # Quantitative Penalty Terms:
            # 1. Risk penalty: tail spread (P90 - expected) + availability exposure
            tail_risk_usd = max(0.0, p90 - exp_cost)
            avail_penalty_usd = (avail_risk / 100.0) * exp_cost * 0.25
            risk_penalty = (tail_risk_usd + avail_penalty_usd) * lambda_risk

            # 2. Schedule penalty: late delivery risk * value of freight
            schedule_penalty = delivery_risk * exp_cost * 0.35 * lambda_sched

            # 3. Flexibility preference: flexibility value subtracted from cost
            flex_benefit = flex_score * exp_cost * 0.06 * lambda_flex

            # Objective function to minimize
            objective_score = exp_cost + risk_penalty + schedule_penalty - flex_benefit

            # Formulate explainable reasons for this strategy
            strat_reasons = []
            if w_med >= 0.70:
                strat_reasons.append("Guarantees vessel availability and eliminates market rate volatility.")
            elif w_spot >= 0.70:
                strat_reasons.append("Maximizes commercial flexibility without forward volume commitments.")
            else:
                strat_reasons.append(
                    f"Diversifies exposure with {strat.short_term_pct:.0f}% term hedge and {strat.spot_pct:.0f}% spot flexibility."
                )

            if p90 > exp_cost * 1.20:
                strat_reasons.append(f"Carries elevated P90 tail risk exposure (${p90:,.0f}).")
            if avail_risk < 15.0:
                strat_reasons.append("Provides strong protection against tonnage shortage in loading basin.")

            evaluations.append(
                StrategyEvaluation(
                    strategy_name=strat.name,
                    spot_percentage=strat.spot_pct,
                    short_term_percentage=strat.short_term_pct,
                    medium_term_percentage=strat.medium_term_pct,
                    expected_freight_cost=exp_freight,
                    expected_total_cost=exp_cost,
                    p10_cost=p10,
                    p50_cost=p50,
                    p90_cost=p90,
                    price_volatility_exposure=vol_exposure,
                    vessel_availability_exposure=avail_risk,
                    delivery_risk=delivery_risk,
                    flexibility_score=flex_score,
                    expected_demurrage_exposure=dem_exposure,
                    risk_penalty=risk_penalty,
                    schedule_penalty=schedule_penalty,
                    flexibility_preference=flex_benefit,
                    objective_score=objective_score,
                    reasons=strat_reasons,
                )
            )

        # Select strategy with minimum objective score
        evaluations.sort(key=lambda s: s.objective_score)
        best = evaluations[0]

        # Overarching explainable recommendation text & reasons
        top_reasons = []
        top_reasons.append(
            f"Recommended '{best.strategy_name}' optimizes risk-adjusted total cost (${best.expected_total_cost:,.0f}) for {inputs.risk_tolerance.value} risk tolerance."
        )

        if inputs.risk_tolerance == RiskTolerance.LOW:
            top_reasons.append("Low risk tolerance prioritizes cost certainty, bounding P90 tail risk to $" + f"{best.p90_cost:,.0f}.")
        elif inputs.risk_tolerance == RiskTolerance.HIGH:
            top_reasons.append(f"High risk tolerance captures flexibility ({best.flexibility_score * 100:.0f}%) and potential spot savings.")
        else:
            top_reasons.append("Balanced risk tolerance hedges rate volatility while preserving operational maneuvering room.")

        if best.spot_percentage > 0 and (best.short_term_percentage > 0 or best.medium_term_percentage > 0):
            top_reasons.append(
                f"Hybrid strategy locks {best.short_term_percentage + best.medium_term_percentage:.0f}% volume to protect against delays, leaving {best.spot_percentage:.0f}% spot."
            )
        elif best.spot_percentage == 100.0:
            top_reasons.append("Full spot allocation avoids forward lock-in costs and provides maximum agility.")
        else:
            top_reasons.append("Full term lock-in shields cargo from market spikes and vessel positioning bottlenecks.")

        # Overall risk score derived from best strategy (0 to 100)
        risk_score = min(100.0, (best.price_volatility_exposure * 0.6) + (best.vessel_availability_exposure * 0.4))

        return RiskAwareContractRecommendation(
            recommended_strategy=best.strategy_name,
            spot_percentage=best.spot_percentage,
            short_term_percentage=best.short_term_percentage,
            medium_term_percentage=best.medium_term_percentage,
            expected_cost=best.expected_total_cost,
            p90_cost=best.p90_cost,
            risk_score=risk_score,
            flexibility_score=best.flexibility_score,
            reasons=top_reasons,
            evaluated_strategies=evaluations,
        )


# =============================================================================
# Legacy ContractOptimizer Class (Preserved for 100% Backward Compatibility)
# =============================================================================

class ContractOptimizer:
    """
    Contract Strategy Optimization Engine.
    Supports both legacy rule evaluation and Phase 9 quantitative risk-aware optimization.
    """

    def __init__(self):
        self.risk_aware_optimizer = RiskAwareContractOptimizer()

    def recommend(self, inputs: ContractOptimizationInputs) -> ContractStrategyRecommendation:
        """
        Backward-compatible recommendation method matching Phase 1-7 test signatures.
        """
        reasons = []

        # 1. Single Voyage Override
        if inputs.number_of_required_voyages == 1:
            reasons.append("Only a single voyage is required.")
            if inputs.forecast_direction == ForecastDirection.FALLING and inputs.vessel_availability == VesselAvailability.ABUNDANT:
                reasons.append("Forecast indicates falling rates and vessels are abundant. Suggest waiting slightly.")
                return ContractStrategyRecommendation(ContractStrategy.WAIT, "Delay booking briefly to capture falling spot rates.", reasons)
            else:
                reasons.append("Immediate booking recommended for single execution.")
                return ContractStrategyRecommendation(ContractStrategy.SPOT, "Book spot immediately for the single voyage.", reasons)

        # Multi-voyage Logic
        reasons.append(f"Multiple voyages required ({inputs.number_of_required_voyages}).")

        # 2. Vessel Shortage Override
        if inputs.vessel_availability == VesselAvailability.SHORTAGE:
            reasons.append("Critical vessel shortage overrides market price optimization.")
            return ContractStrategyRecommendation(
                ContractStrategy.MEDIUM_TERM,
                "Lock in a medium-term contract immediately to guarantee vessel availability.",
                reasons
            )

        # 3. High Risk / High Uncertainty -> Hybrid Diversification
        if inputs.forecast_uncertainty == ForecastUncertainty.HIGH or inputs.overall_risk_score >= 70.0:
            reasons.append("High forecast uncertainty or overall risk mandates risk diversification.")

            if inputs.forecast_direction == ForecastDirection.RISING:
                short_term_pct = 60
                spot_pct = 40
                reasons.append(f"Rates are expected to rise. {short_term_pct}% protects baseline, {spot_pct}% spot offers flexibility.")
            elif inputs.forecast_direction == ForecastDirection.FALLING:
                short_term_pct = 30
                spot_pct = 70
                reasons.append(f"Rates expected to fall. {spot_pct}% spot captures downside, {short_term_pct}% term secures core operations.")
            else:
                short_term_pct = 50
                spot_pct = 50
                reasons.append("Uncertain/stable market. Balanced 50/50 split optimizes risk.")

            recommendation_text = f"{short_term_pct}% short-term contract, {spot_pct}% spot exposure."
            return ContractStrategyRecommendation(ContractStrategy.HYBRID, recommendation_text, reasons)

        # 4. Clear Rising Trend
        if inputs.forecast_direction == ForecastDirection.RISING:
            reasons.append("Forecast indicates a clear upward trend in freight rates.")
            if inputs.forecast_uncertainty == ForecastUncertainty.LOW:
                reasons.append("High confidence in forecast allows for longer-term lock-in.")
                return ContractStrategyRecommendation(
                    ContractStrategy.MEDIUM_TERM,
                    "Secure a medium-term contract now before rates rise further.",
                    reasons
                )
            else:
                reasons.append("Moderate confidence suggests locking in near-term only.")
                return ContractStrategyRecommendation(
                    ContractStrategy.SHORT_TERM,
                    "Secure a short-term contract to hedge against immediate rate increases.",
                    reasons
                )

        # 5. Clear Falling Trend
        if inputs.forecast_direction == ForecastDirection.FALLING:
            reasons.append("Forecast indicates falling freight rates.")
            if inputs.vessel_availability == VesselAvailability.TIGHT:
                reasons.append("Vessels are somewhat tight. Secure baseline capacity.")
                return ContractStrategyRecommendation(
                    ContractStrategy.HYBRID,
                    "50% short-term to secure tight capacity, 50% spot to capture falling rates.",
                    reasons
                )
            else:
                reasons.append("Vessels are abundant. Avoid locking in long-term contracts.")
                return ContractStrategyRecommendation(
                    ContractStrategy.SPOT,
                    "Rely on spot market to capitalize on declining freight rates.",
                    reasons
                )

        # 6. Stable/Uncertain without extreme risk
        reasons.append("Market is relatively stable or flat.")
        return ContractStrategyRecommendation(
            ContractStrategy.SHORT_TERM,
            "Maintain baseline coverage with short-term contracts while monitoring the market.",
            reasons
        )

    def optimize_risk_aware(self, inputs: RiskAwareContractInputs) -> RiskAwareContractRecommendation:
        """Invokes Phase 9 quantitative risk-aware contract optimization."""
        return self.risk_aware_optimizer.optimize_contract_strategy(inputs)

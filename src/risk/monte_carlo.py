"""
Charter-AI — Probabilistic Monte Carlo Simulation Engine (Phase 8).

Simulates high-dimensional maritime uncertainty across:
1. Freight rate volatility (Log-normal distribution)
2. Bunker fuel price risk (Log-normal distribution)
3. Port waiting times & queue congestion (Gamma distribution)
4. Sea voyage duration & weather speed loss (Beta / Truncated Normal distribution)
5. Port handling & demurrage liability (Laytime threshold model)
6. Vessel availability & fixture cancellation risk (Bernoulli trial)

High-performance vectorized implementation running 10,000 iterations in <100ms.
Seed-controlled for exact mathematical reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloSimulationConfig:
    """Configuration and parameters for Monte Carlo simulation."""
    n_simulations: int = 10_000
    seed: int = 42
    cost_threshold_usd: Optional[float] = None
    spot_replacement_penalty_pct: float = 0.20  # +20% freight cost if preferred vessel fixture falls through


@dataclass
class CharterPlanInputs:
    """Inputs representing a candidate charter plan for probabilistic simulation."""
    cargo_quantity_t: float
    base_freight_rate: float  # $/MT or lumpsum
    freight_rate_is_per_tonne: bool = True
    freight_volatility_pct: float = 15.0  # Annual/voyage volatility
    freight_rate_p10: Optional[float] = None
    freight_rate_p90: Optional[float] = None

    base_bunker_price: float = 650.0  # $/MT (VLSFO)
    bunker_volatility_pct: float = 12.0
    sea_distance_nm: float = 4500.0
    service_speed_knots: float = 12.5
    fuel_consumption_t_day: float = 28.0

    expected_wait_days: float = 2.0
    p90_wait_days: Optional[float] = None
    port_handling_rate_t_day: float = 15_000.0
    agreed_laytime_days: Optional[float] = None  # If None, calculated from cargo & handling rate
    demurrage_rate_usd_day: float = 20_000.0

    port_charges_usd: float = 45_000.0
    canal_charges_usd: float = 0.0
    misc_agency_usd: float = 8_000.0

    delivery_deadline_days: Optional[float] = 25.0
    vessel_availability_probability: float = 0.95  # 95% vessel availability probability


@dataclass
class SimulationDistributionBin:
    bin_min: float
    bin_max: float
    bin_center: float
    count: int
    probability: float


@dataclass
class MonteCarloResult:
    """Results of the probabilistic Monte Carlo simulation."""
    expected_cost: float
    p10_cost: float
    p50_cost: float
    p90_cost: float
    p95_cost: float
    min_cost: float
    max_cost: float
    cost_std: float

    demurrage_probability: float
    late_delivery_probability: float
    probability_cost_exceeds_threshold: float
    probability_of_infeasibility: float

    expected_duration_days: float
    p10_duration_days: float
    p50_duration_days: float
    p90_duration_days: float

    expected_demurrage_cost: float
    p90_demurrage_cost: float

    seed: int
    n_simulations: int
    distribution: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_cost": round(self.expected_cost, 2),
            "p10_cost": round(self.p10_cost, 2),
            "p50_cost": round(self.p50_cost, 2),
            "p90_cost": round(self.p90_cost, 2),
            "p95_cost": round(self.p95_cost, 2),
            "min_cost": round(self.min_cost, 2),
            "max_cost": round(self.max_cost, 2),
            "cost_std": round(self.cost_std, 2),
            "demurrage_probability": round(self.demurrage_probability, 4),
            "late_delivery_probability": round(self.late_delivery_probability, 4),
            "probability_cost_exceeds_threshold": round(self.probability_cost_exceeds_threshold, 4),
            "probability_of_infeasibility": round(self.probability_of_infeasibility, 4),
            "expected_duration_days": round(self.expected_duration_days, 1),
            "p10_duration_days": round(self.p10_duration_days, 1),
            "p50_duration_days": round(self.p50_duration_days, 1),
            "p90_duration_days": round(self.p90_duration_days, 1),
            "expected_demurrage_cost": round(self.expected_demurrage_cost, 2),
            "p90_demurrage_cost": round(self.p90_demurrage_cost, 2),
            "seed": self.seed,
            "n_simulations": self.n_simulations,
            "distribution": self.distribution,
            "assumptions": self.assumptions,
        }


class MonteCarloSimulator:
    """
    Vectorized Monte Carlo simulation engine for charter plans.
    """

    def __init__(self, config: Optional[MonteCarloSimulationConfig] = None):
        self.config = config or MonteCarloSimulationConfig()

    def run_simulation(
        self,
        plan: CharterPlanInputs,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
        cost_threshold_usd: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Executes a vectorized Monte Carlo simulation with documented statistical assumptions.
        """
        n_sims = n_simulations or self.config.n_simulations
        sim_seed = seed if seed is not None else self.config.seed

        # Initialize NumPy random generator for reproducibility
        rng = np.random.default_rng(sim_seed)

        # ---------------------------------------------------------------------
        # 1. Freight Rate Uncertainty (Log-Normal Distribution)
        # ---------------------------------------------------------------------
        # Freight rates are strictly positive and right-skewed.
        base_rate = max(1.0, plan.base_freight_rate)
        if plan.freight_rate_p10 and plan.freight_rate_p90 and plan.freight_rate_p90 > plan.freight_rate_p10:
            # Calibrate sigma from P10-P90 spread (z_90 - z_10 = 2 * 1.28155 = 2.5631)
            sigma_freight = np.log(plan.freight_rate_p90 / plan.freight_rate_p10) / 2.5631
            mu_freight = np.log(base_rate)
        else:
            sigma_freight = max(0.05, plan.freight_volatility_pct / 100.0)
            mu_freight = np.log(base_rate) - 0.5 * (sigma_freight ** 2)

        sim_freight_rates = rng.lognormal(mean=mu_freight, sigma=sigma_freight, size=n_sims)

        if plan.freight_rate_is_per_tonne:
            sim_freight_costs = sim_freight_rates * plan.cargo_quantity_t
        else:
            sim_freight_costs = sim_freight_rates.copy()

        # ---------------------------------------------------------------------
        # 2. Bunker Price Uncertainty (Log-Normal Distribution)
        # ---------------------------------------------------------------------
        base_bunker = max(100.0, plan.base_bunker_price)
        sigma_bunker = max(0.04, plan.bunker_volatility_pct / 100.0)
        mu_bunker = np.log(base_bunker) - 0.5 * (sigma_bunker ** 2)
        sim_bunker_prices = rng.lognormal(mean=mu_bunker, sigma=sigma_bunker, size=n_sims)

        # ---------------------------------------------------------------------
        # 3. Sea Voyage Duration & Weather Speed Loss
        # ---------------------------------------------------------------------
        nominal_speed = max(8.0, plan.service_speed_knots)
        nominal_sea_hours = plan.sea_distance_nm / nominal_speed
        nominal_sea_days = nominal_sea_hours / 24.0

        # Weather induces 0% to 15% speed loss, modeled with Beta(6, 2)
        # scaled between 0.85 and 1.02
        weather_speed_factors = 0.85 + (rng.beta(a=6.0, b=2.0, size=n_sims) * 0.17)
        sim_sea_days = nominal_sea_days / weather_speed_factors

        # Bunker consumption: sailing burn + port auxiliary burn (3t/day in port)
        sim_sea_bunker_t = sim_sea_days * plan.fuel_consumption_t_day

        # ---------------------------------------------------------------------
        # 4. Port Congestion & Waiting Time Uncertainty (Gamma Distribution)
        # ---------------------------------------------------------------------
        # Port queues follow Erlang/Gamma right-skewed wait distributions.
        exp_wait = max(0.2, plan.expected_wait_days)
        # Shape k=2.5, scale theta = mean / k
        shape_k = 2.5
        scale_theta = exp_wait / shape_k
        sim_wait_days = rng.gamma(shape=shape_k, scale=scale_theta, size=n_sims)

        # ---------------------------------------------------------------------
        # 5. Port Handling Time & Demurrage Liability
        # ---------------------------------------------------------------------
        handling_rate = max(1000.0, plan.port_handling_rate_t_day)
        nominal_handling_days = (plan.cargo_quantity_t / handling_rate) * 2.0  # Loading + Discharging
        # Handling variation +/- 10%
        handling_efficiency = rng.normal(loc=1.0, scale=0.08, size=n_sims)
        handling_efficiency = np.clip(handling_efficiency, 0.75, 1.40)
        sim_handling_days = nominal_handling_days * handling_efficiency

        # Total port turnaround time = wait days + handling days
        sim_port_days = sim_wait_days + sim_handling_days

        # Laytime allowed
        if plan.agreed_laytime_days is not None and plan.agreed_laytime_days > 0:
            laytime_allowed = plan.agreed_laytime_days
        else:
            laytime_allowed = nominal_handling_days + 1.0  # standard 1-day laytime buffer

        sim_demurrage_days = np.maximum(0.0, sim_port_days - laytime_allowed)
        sim_demurrage_costs = sim_demurrage_days * plan.demurrage_rate_usd_day

        # Port auxiliary bunker consumption (approx 3.0 MT/day in port)
        sim_port_bunker_t = sim_port_days * 3.0
        total_bunker_burn_t = sim_sea_bunker_t + sim_port_bunker_t
        sim_bunker_costs = total_bunker_burn_t * sim_bunker_prices

        # ---------------------------------------------------------------------
        # 6. Total Voyage Duration & Schedule Deadline Risk
        # ---------------------------------------------------------------------
        sim_total_durations = sim_sea_days + sim_port_days

        deadline = plan.delivery_deadline_days if plan.delivery_deadline_days is not None else 9999.0
        late_delivery_mask = sim_total_durations > deadline
        late_delivery_prob = float(np.mean(late_delivery_mask))

        # ---------------------------------------------------------------------
        # 7. Vessel Availability & Cancellation / Infeasibility Risk
        # ---------------------------------------------------------------------
        avail_prob = max(0.50, min(1.0, plan.vessel_availability_probability))
        # Bernoulli trial for vessel availability
        vessel_avail_draws = rng.uniform(0.0, 1.0, size=n_sims)
        vessel_unavailable_mask = vessel_avail_draws >= avail_prob

        # Infeasibility: if vessel unavailable and cannot be replaced in time (50% lead to infeasibility, 50% spot surcharge)
        infeasible_draws = rng.uniform(0.0, 1.0, size=n_sims)
        infeasible_mask = vessel_unavailable_mask & (infeasible_draws < 0.40)
        infeasibility_prob = float(np.mean(infeasible_mask))

        # Spot replacement surcharge for unavailable but replaceable instances
        spot_surcharge_mask = vessel_unavailable_mask & (~infeasible_mask)
        spot_surcharges = np.where(
            spot_surcharge_mask,
            sim_freight_costs * self.config.spot_replacement_penalty_pct,
            0.0
        )

        # ---------------------------------------------------------------------
        # 8. Total Delivered Cost Calculation
        # ---------------------------------------------------------------------
        fixed_costs = plan.port_charges_usd + plan.canal_charges_usd + plan.misc_agency_usd
        sim_total_costs = (
            sim_freight_costs
            + sim_bunker_costs
            + sim_demurrage_costs
            + fixed_costs
            + spot_surcharges
        )

        # Statistical Aggregations
        expected_cost = float(np.mean(sim_total_costs))
        p10_cost = float(np.percentile(sim_total_costs, 10))
        p50_cost = float(np.percentile(sim_total_costs, 50))
        p90_cost = float(np.percentile(sim_total_costs, 90))
        p95_cost = float(np.percentile(sim_total_costs, 95))
        min_cost = float(np.min(sim_total_costs))
        max_cost = float(np.max(sim_total_costs))
        cost_std = float(np.std(sim_total_costs))

        demurrage_mask = sim_demurrage_costs > 0.0
        demurrage_prob = float(np.mean(demurrage_mask))

        # Cost threshold
        threshold = (
            cost_threshold_usd
            if cost_threshold_usd is not None
            else (self.config.cost_threshold_usd or (expected_cost * 1.15))
        )
        cost_exceed_prob = float(np.mean(sim_total_costs > threshold))

        # Duration percentiles
        expected_dur = float(np.mean(sim_total_durations))
        p10_dur = float(np.percentile(sim_total_durations, 10))
        p50_dur = float(np.percentile(sim_total_durations, 50))
        p90_dur = float(np.percentile(sim_total_durations, 90))

        # Demurrage cost stats
        expected_dem_cost = float(np.mean(sim_demurrage_costs))
        p90_dem_cost = float(np.percentile(sim_demurrage_costs, 90))

        # ---------------------------------------------------------------------
        # 9. Generate Distribution Histogram for UI Charts
        # ---------------------------------------------------------------------
        counts, bin_edges = np.histogram(sim_total_costs, bins=25)
        distribution_bins = []
        for i in range(len(counts)):
            b_min = float(bin_edges[i])
            b_max = float(bin_edges[i + 1])
            distribution_bins.append({
                "bin_min": round(b_min, 2),
                "bin_max": round(b_max, 2),
                "bin_center": round((b_min + b_max) / 2.0, 2),
                "count": int(counts[i]),
                "probability": round(float(counts[i]) / n_sims, 4),
            })

        assumptions = {
            "freight_rate_distribution": "Log-normal LN(mu, sigma) calibrated to forecast volatility / spread",
            "bunker_price_distribution": "Log-normal LN(mu, sigma) reflecting global bunker volatility",
            "port_waiting_distribution": "Gamma(k=2.5, theta=E[wait]/2.5) right-skewed queue model",
            "voyage_duration_distribution": "Sea passage with Beta(6, 2) weather speed degradation + port handling variation",
            "demurrage_model": "Laytime threshold: max(0, port_turnaround - laytime_allowed) * demurrage_rate",
            "vessel_availability_model": "Bernoulli trials with spot market emergency re-fixture penalty",
            "seed": f"{sim_seed} (deterministic)",
        }

        return MonteCarloResult(
            expected_cost=expected_cost,
            p10_cost=p10_cost,
            p50_cost=p50_cost,
            p90_cost=p90_cost,
            p95_cost=p95_cost,
            min_cost=min_cost,
            max_cost=max_cost,
            cost_std=cost_std,
            demurrage_probability=demurrage_prob,
            late_delivery_probability=late_delivery_prob,
            probability_cost_exceeds_threshold=cost_exceed_prob,
            probability_of_infeasibility=infeasibility_prob,
            expected_duration_days=expected_dur,
            p10_duration_days=p10_dur,
            p50_duration_days=p50_dur,
            p90_duration_days=p90_dur,
            expected_demurrage_cost=expected_dem_cost,
            p90_demurrage_cost=p90_dem_cost,
            seed=sim_seed,
            n_simulations=n_sims,
            distribution=distribution_bins,
            assumptions=assumptions,
        )

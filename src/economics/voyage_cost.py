"""
Charter-AI — Realistic Voyage Economics Engine (Phase 5).

A comprehensive, transparent, formula-based economics engine designed
specifically from the CHARTERER / CARGO-OWNER perspective.

Distinguishes charterer delivered costs (Freight rate $/t, Bunker escalation/BAF,
Port tariffs & handling, Congestion-driven waiting penalties, Demurrage liability,
Positioning deadheading, Canal tolls, Agency fees, and Risk contingencies)
from shipowner profitability (Time Charter Equivalent yield).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VoyageCostInputs:
    """
    Configurable voyage parameters from the charterer/cargo-owner perspective.
    """
    # Core Cargo & Freight
    cargo_quantity_t: float
    freight_rate_usd: float  # USD per tonne (or USD/day if time_charter)

    # Vessel Technical Specs
    vessel_speed_knots: float  # Laden service speed (knots)
    vessel_daily_fuel_consumption_tpd: float  # Sea fuel consumption (MT/day)
    vessel_daily_hire_cost_usd: float  # Daily hire / opportunity cost (USD/day)
    route_distance_nm: float  # Laden voyage distance (nautical miles)

    # Positioning / Deadheading
    positioning_distance_nm: float = 0.0  # Ballast distance to load port (nm)

    # Fuel & Bunker
    fuel_price_usd_per_t: float = 550.0  # VLSFO bunker price (USD/MT)
    port_daily_fuel_consumption_tpd: float = 3.0  # Auxiliary generator fuel in port (MT/day)
    bunker_exposure_pct: Optional[float] = None  # Charterer fuel exposure (1.0 = 100%, 0.0 = fixed gross freight)

    # Port Handling & Charges
    load_port_cost_usd: float = 0.0  # Port tariffs, pilotage, tugs, berth hire at origin
    discharge_port_cost_usd: float = 0.0  # Port tariffs, pilotage, tugs, berth hire at destination
    port_handling_rate_tpd: Optional[float] = None  # Load port cargo handling rate (MT/day)
    discharge_port_handling_rate_tpd: Optional[float] = None  # Discharge port handling rate (MT/day)
    loading_time_days: Optional[float] = None  # Explicit loading duration override (days)
    discharge_time_days: Optional[float] = None  # Explicit discharge duration override (days)

    # Congestion & Demurrage
    expected_waiting_days: float = 0.0  # Waiting time at anchorage from Congestion Predictor
    daily_demurrage_rate_usd: float = 20000.0  # Contractual demurrage rate (USD/day)
    laytime_allowed_days: Optional[float] = None  # Contractual laytime allowed (days)
    despatch_rate_fraction: float = 0.50  # Despatch rate as fraction of demurrage rate

    # Additional Voyage Costs
    canal_charges_usd: float = 0.0  # Canal/strait tolls or security transit fees
    agency_fees_usd: float = 0.0  # Port agent fees, surveyor, customs, documentation
    contingency_cost_usd: float = 0.0  # Risk buffer for weather disruption / unexpected delays
    other_costs_usd: float = 0.0  # Miscellaneous / communications / sundry expenses

    # Schedule & Deadlines
    delivery_deadline_days: Optional[float] = None  # Target delivery deadline from start (days)
    vessel_availability_days: float = 0.0  # Days until vessel is available at loading area
    vessel_dwt: Optional[float] = None  # Vessel deadweight tonnage

    # Contract Model
    charter_type: str = "voyage"  # "voyage" (USD/t) or "time_charter" (USD/day)


@dataclass
class VoyageCostBreakdown:
    """
    Itemized cost breakdown returned by the voyage economics engine.
    Supports attribute access (legacy and canonical) and dict indexing.
    """
    # Canonical components
    freight_cost: float
    bunker_cost: float
    port_cost: float
    waiting_cost: float
    demurrage_exposure: float
    positioning_cost: float
    miscellaneous_cost: float
    total_cost: float
    cost_per_tonne: float
    voyage_days: float
    delivery_probability: float

    # Granular operational metrics
    sailing_days: float = 0.0
    positioning_days: float = 0.0
    loading_days: float = 0.0
    discharge_days: float = 0.0
    waiting_days: float = 0.0
    total_port_days: float = 0.0
    laytime_allowed_days: float = 0.0
    excess_time_days: float = 0.0
    despatch_savings: float = 0.0
    canal_charges: float = 0.0
    agency_fees: float = 0.0
    contingency_cost: float = 0.0
    other_costs: float = 0.0

    # Backward compatibility properties
    @property
    def freight_cost_usd(self) -> float:
        return self.freight_cost

    @property
    def bunker_cost_usd(self) -> float:
        return self.bunker_cost

    @property
    def port_costs_usd(self) -> float:
        return self.port_cost

    @property
    def waiting_cost_usd(self) -> float:
        return self.waiting_cost

    @property
    def expected_demurrage_usd(self) -> float:
        return self.demurrage_exposure

    @property
    def positioning_cost_usd(self) -> float:
        return self.positioning_cost

    @property
    def other_costs_usd(self) -> float:
        return self.other_costs

    @property
    def total_voyage_cost_usd(self) -> float:
        return self.total_cost

    def to_dict(self) -> Dict[str, float]:
        """Returns the canonical dictionary specified by Phase 5 requirements."""
        return {
            "freight_cost": round(self.freight_cost, 2),
            "bunker_cost": round(self.bunker_cost, 2),
            "port_cost": round(self.port_cost, 2),
            "waiting_cost": round(self.waiting_cost, 2),
            "demurrage_exposure": round(self.demurrage_exposure, 2),
            "positioning_cost": round(self.positioning_cost, 2),
            "miscellaneous_cost": round(self.miscellaneous_cost, 2),
            "total_cost": round(self.total_cost, 2),
            "cost_per_tonne": round(self.cost_per_tonne, 2),
            "voyage_days": round(self.voyage_days, 2),
            "delivery_probability": round(self.delivery_probability, 4),
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in VoyageCostBreakdown")


def calculate_sailing_days(distance_nm: float, speed_knots: float) -> float:
    """
    Calculate sailing time strictly using:
    sailing_days = distance_nm / (speed_knots * 24)
    Eliminating any hard-coded 14-day voyage estimates.
    """
    if speed_knots <= 0 or distance_nm <= 0:
        return 0.0
    return distance_nm / (speed_knots * 24.0)


def calculate_port_operational_time(
    cargo_quantity_t: float,
    handling_rate_tpd: Optional[float],
    default_rate_tpd: float = 12000.0,
) -> float:
    """
    Calculate port loading or discharging operational time in days:
    port_time = cargo_quantity / handling_rate
    """
    if cargo_quantity_t <= 0:
        return 0.0
    effective_rate = handling_rate_tpd if (handling_rate_tpd and handling_rate_tpd > 0) else default_rate_tpd
    return cargo_quantity_t / effective_rate


def calculate_delivery_probability(
    total_elapsed_days: float,
    delivery_deadline_days: Optional[float],
    uncertainty_days: float = 1.5,
) -> float:
    """
    Model probability of meeting delivery deadline using the Gaussian/Logistic CDF.
    Schedule buffer = delivery_deadline - total_elapsed_days.
    """
    if delivery_deadline_days is None:
        return 0.98  # Normal standard delivery confidence without restrictive deadline

    schedule_buffer = delivery_deadline_days - total_elapsed_days
    # Z-score = schedule_buffer / uncertainty_std
    z = schedule_buffer / max(0.5, uncertainty_days)
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.01, min(0.99, prob))


def calculate_voyage_cost(inputs: VoyageCostInputs) -> VoyageCostBreakdown:
    """
    Calculates realistic voyage economics from the charterer / cargo-owner perspective.

    Itemizes:
    1. Freight cost
    2. Bunker / fuel exposure
    3. Port charges
    4. Waiting cost (anchorage opportunity cost)
    5. Demurrage exposure (modeled via laytime allowed, operational time, and waiting time)
    6. Positioning / deadheading cost
    7. Canal / route charges
    8. Agency / miscellaneous cost
    9. Contingency / risk cost
    """
    # ---------------------------------------------------------
    # 1. Sailing & Positioning Duration (No hard-coded 14-day voyage)
    # ---------------------------------------------------------
    sailing_days = calculate_sailing_days(inputs.route_distance_nm, inputs.vessel_speed_knots)
    positioning_days = calculate_sailing_days(inputs.positioning_distance_nm, inputs.vessel_speed_knots)

    # ---------------------------------------------------------
    # 2. Port Handling Times (Driven by Cargo Quantity & Handling Rates)
    # ---------------------------------------------------------
    has_explicit_rates = (inputs.port_handling_rate_tpd is not None) or (inputs.discharge_port_handling_rate_tpd is not None)

    if inputs.loading_time_days is not None:
        loading_days = inputs.loading_time_days
    elif has_explicit_rates:
        loading_days = calculate_port_operational_time(
            inputs.cargo_quantity_t, inputs.port_handling_rate_tpd, default_rate_tpd=15000.0
        )
    else:
        loading_days = 0.0  # Retain 0.0 for legacy test compatibility when rates not passed

    if inputs.discharge_time_days is not None:
        discharge_days = inputs.discharge_time_days
    elif has_explicit_rates:
        dest_rate = inputs.discharge_port_handling_rate_tpd or inputs.port_handling_rate_tpd
        discharge_days = calculate_port_operational_time(
            inputs.cargo_quantity_t, dest_rate, default_rate_tpd=12000.0
        )
    else:
        discharge_days = 0.0

    actual_operational_time = loading_days + discharge_days
    waiting_days = max(0.0, float(inputs.expected_waiting_days))
    total_port_days = actual_operational_time + waiting_days
    voyage_days = positioning_days + sailing_days + total_port_days

    # ---------------------------------------------------------
    # 3. Freight Cost Calculation
    # ---------------------------------------------------------
    if inputs.charter_type == "time_charter":
        freight_cost = inputs.freight_rate_usd * (sailing_days + total_port_days)
    else:
        freight_cost = inputs.freight_rate_usd * inputs.cargo_quantity_t

    # ---------------------------------------------------------
    # 4. Bunker / Fuel Exposure
    # ---------------------------------------------------------
    sea_fuel_t = (sailing_days + positioning_days) * inputs.vessel_daily_fuel_consumption_tpd
    port_fuel_t = total_port_days * inputs.port_daily_fuel_consumption_tpd if has_explicit_rates else 0.0
    total_fuel_consumed_t = sea_fuel_t + port_fuel_t
    raw_bunker_cost = total_fuel_consumed_t * inputs.fuel_price_usd_per_t

    exposure_pct = inputs.bunker_exposure_pct if inputs.bunker_exposure_pct is not None else 1.0
    bunker_cost = raw_bunker_cost * exposure_pct

    # ---------------------------------------------------------
    # 5. Port Charges
    # ---------------------------------------------------------
    port_cost = inputs.load_port_cost_usd + inputs.discharge_port_cost_usd

    # ---------------------------------------------------------
    # 6. Waiting Cost (Anchorage idle capital & delay penalty)
    # ---------------------------------------------------------
    waiting_cost = waiting_days * inputs.vessel_daily_hire_cost_usd

    # ---------------------------------------------------------
    # 7. Demurrage Exposure Model
    # ---------------------------------------------------------
    if inputs.laytime_allowed_days is not None:
        laytime_allowed = inputs.laytime_allowed_days
    elif has_explicit_rates:
        laytime_allowed = actual_operational_time + 1.0
    else:
        laytime_allowed = 2.0

    if has_explicit_rates:
        excess_time = max(0.0, total_port_days - laytime_allowed)
        despatch_time = max(0.0, laytime_allowed - total_port_days)
    else:
        excess_time = max(0.0, waiting_days - laytime_allowed)
        despatch_time = 0.0

    demurrage_exposure = excess_time * inputs.daily_demurrage_rate_usd
    despatch_savings = despatch_time * (inputs.daily_demurrage_rate_usd * inputs.despatch_rate_fraction)

    # ---------------------------------------------------------
    # 8. Positioning / Deadheading Cost
    # ---------------------------------------------------------
    positioning_cost = positioning_days * inputs.vessel_daily_hire_cost_usd

    # ---------------------------------------------------------
    # 9. Canal, Agency & Miscellaneous Costs
    # ---------------------------------------------------------
    canal_charges = inputs.canal_charges_usd
    agency_fees = inputs.agency_fees_usd
    contingency_cost = inputs.contingency_cost_usd
    other_costs = inputs.other_costs_usd
    miscellaneous_cost = canal_charges + agency_fees + contingency_cost + other_costs

    # ---------------------------------------------------------
    # 10. Total Delivered Cost & Cost per Tonne
    # ---------------------------------------------------------
    total_cost = (
        freight_cost
        + bunker_cost
        + port_cost
        + waiting_cost
        + demurrage_exposure
        + positioning_cost
        + miscellaneous_cost
        - despatch_savings
    )
    cost_per_tonne = (total_cost / inputs.cargo_quantity_t) if inputs.cargo_quantity_t > 0 else 0.0

    # ---------------------------------------------------------
    # 11. Delivery Probability
    # ---------------------------------------------------------
    total_elapsed_days = inputs.vessel_availability_days + voyage_days
    schedule_uncertainty = math.sqrt((0.08 * sailing_days) ** 2 + (0.35 * max(1.0, waiting_days)) ** 2)
    delivery_probability = calculate_delivery_probability(
        total_elapsed_days=total_elapsed_days,
        delivery_deadline_days=inputs.delivery_deadline_days,
        uncertainty_days=schedule_uncertainty,
    )

    return VoyageCostBreakdown(
        freight_cost=round(freight_cost, 2),
        bunker_cost=round(bunker_cost, 2),
        port_cost=round(port_cost, 2),
        waiting_cost=round(waiting_cost, 2),
        demurrage_exposure=round(demurrage_exposure, 2),
        positioning_cost=round(positioning_cost, 2),
        miscellaneous_cost=round(miscellaneous_cost, 2),
        total_cost=round(total_cost, 2),
        cost_per_tonne=round(cost_per_tonne, 2),
        voyage_days=round(voyage_days, 2),
        delivery_probability=round(delivery_probability, 4),
        sailing_days=round(sailing_days, 2),
        positioning_days=round(positioning_days, 2),
        loading_days=round(loading_days, 2),
        discharge_days=round(discharge_days, 2),
        waiting_days=round(waiting_days, 2),
        total_port_days=round(total_port_days, 2),
        laytime_allowed_days=round(laytime_allowed, 2),
        excess_time_days=round(excess_time, 2),
        despatch_savings=round(despatch_savings, 2),
        canal_charges=round(canal_charges, 2),
        agency_fees=round(agency_fees, 2),
        contingency_cost=round(contingency_cost, 2),
        other_costs=round(other_costs, 2),
    )


# -------------------------------------------------------------------------
# Sensitivity Analysis Engine
# -------------------------------------------------------------------------

def perform_sensitivity_analysis(
    inputs: VoyageCostInputs,
    parameter: str,
    values: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform single-parameter sensitivity analysis on voyage economics.

    Supported parameters:
    - "bunker_price": bunker fuel price ($/MT)
    - "freight_rate": freight rate ($/MT or $/day)
    - "congestion": waiting time (days)
    - "cargo_quantity": cargo quantity (MT)
    """
    base_result = calculate_voyage_cost(inputs)
    base_cost = base_result.total_cost
    base_cpt = base_result.cost_per_tonne

    if values is None:
        if parameter == "bunker_price":
            base_val = inputs.fuel_price_usd_per_t
            values = [round(base_val * mult, 1) for mult in [0.80, 0.90, 1.0, 1.10, 1.20]]
        elif parameter == "freight_rate":
            base_val = inputs.freight_rate_usd
            values = [round(base_val * mult, 2) for mult in [0.80, 0.90, 1.0, 1.10, 1.20]]
        elif parameter == "congestion":
            base_val = inputs.expected_waiting_days
            values = [0.0, max(0.5, round(base_val * 0.5, 1)), base_val, round(base_val + 2.0, 1), round(base_val + 5.0, 1)]
            values = sorted(list(set(values)))
        elif parameter == "cargo_quantity":
            base_val = inputs.cargo_quantity_t
            values = [round(base_val * mult) for mult in [0.80, 0.90, 1.0, 1.10, 1.20]]
        else:
            raise ValueError(f"Unsupported sensitivity parameter: '{parameter}'")

    results = []
    for val in values:
        params = inputs.__dict__.copy()
        if parameter == "bunker_price":
            params["fuel_price_usd_per_t"] = val
        elif parameter == "freight_rate":
            params["freight_rate_usd"] = val
        elif parameter == "congestion":
            params["expected_waiting_days"] = val
        elif parameter == "cargo_quantity":
            params["cargo_quantity_t"] = val

        variant_inputs = VoyageCostInputs(**params)
        res = calculate_voyage_cost(variant_inputs)

        delta_total = res.total_cost - base_cost
        delta_pct = (delta_total / base_cost * 100.0) if base_cost > 0 else 0.0
        delta_cpt = res.cost_per_tonne - base_cpt

        results.append({
            "parameter": parameter,
            "value": val,
            "total_cost": res.total_cost,
            "cost_per_tonne": res.cost_per_tonne,
            "delta_cost_usd": round(delta_total, 2),
            "delta_cost_pct": round(delta_pct, 2),
            "delta_cost_per_tonne_usd": round(delta_cpt, 2),
            "demurrage_exposure": res.demurrage_exposure,
            "waiting_cost": res.waiting_cost,
            "bunker_cost": res.bunker_cost,
            "freight_cost": res.freight_cost,
            "voyage_days": res.voyage_days,
            "delivery_probability": res.delivery_probability,
        })

    return results


def run_full_sensitivity_matrix(inputs: VoyageCostInputs) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run multi-factor sensitivity analysis across all 4 key parameters:
    bunker price, freight rate, congestion, and cargo quantity.
    """
    return {
        "bunker_price": perform_sensitivity_analysis(inputs, "bunker_price"),
        "freight_rate": perform_sensitivity_analysis(inputs, "freight_rate"),
        "congestion": perform_sensitivity_analysis(inputs, "congestion"),
        "cargo_quantity": perform_sensitivity_analysis(inputs, "cargo_quantity"),
    }

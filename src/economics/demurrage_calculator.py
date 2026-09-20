"""
DockInsights — Demurrage Cost Calculator.

Estimates demurrage exposure based on predicted idle time,
contractual laytime, and demurrage rate.

DETERMINISTIC formula, but uses ML-predicted idle time as input.
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.logging import get_logger
from src.data.vessel_repository import get_vessel_class_spec

logger = get_logger(__name__)


def _get_default_demurrage_rates() -> dict[str, float]:
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    res = {s.class_name: s.demurrage_rate_usd for s in specs}
    if "Post-Panamax" not in res:
        res["Post-Panamax"] = 25_000.0
    return res


DEFAULT_DEMURRAGE_RATES = _get_default_demurrage_rates()


def _get_default_laytime_days() -> dict[str, float]:
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    res = {s.class_name: s.default_laytime_days for s in specs}
    if "Post-Panamax" not in res:
        res["Post-Panamax"] = 6.0
    return res


DEFAULT_LAYTIME_DAYS = _get_default_laytime_days()




@dataclass
class DemurrageEstimate:
    """Demurrage cost estimate."""
    predicted_idle_days: float
    allowed_laytime_days: float
    demurrage_days: float  # max(0, idle - laytime)
    demurrage_rate_usd_per_day: float
    total_demurrage_usd: float
    despatch_days: float  # max(0, laytime - idle) — potential savings
    despatch_rate_usd_per_day: float  # Typically 50% of demurrage rate
    potential_despatch_usd: float
    net_demurrage_exposure_usd: float


class DemurrageCalculator:
    """
    Calculates demurrage cost and despatch savings.

    Demurrage: Penalty paid to shipowner when vessel stays beyond allowed laytime.
    Despatch: Reward to charterer if vessel completes loading/discharge early.

    Formula:
        demurrage = max(0, predicted_idle_days - allowed_laytime) × demurrage_rate
        despatch  = max(0, allowed_laytime - predicted_idle_days) × despatch_rate
        net_exposure = demurrage - despatch
    """

    def calculate(
        self,
        vessel_class: str,
        predicted_idle_days: float,
        allowed_laytime_days: Optional[float] = None,
        demurrage_rate: Optional[float] = None,
        despatch_rate_fraction: float = 0.50,
    ) -> DemurrageEstimate:
        """
        Calculate demurrage exposure.

        Args:
            vessel_class: Vessel class name.
            predicted_idle_days: ML-predicted waiting/idle time in days.
            allowed_laytime_days: Contractual laytime. If None, uses default.
            demurrage_rate: USD/day rate. If None, uses class default.
            despatch_rate_fraction: Despatch as fraction of demurrage rate.

        Returns:
            DemurrageEstimate with costs and potential savings.
        """
        class_spec = get_vessel_class_spec(vessel_class)
        laytime = allowed_laytime_days or DEFAULT_LAYTIME_DAYS.get(vessel_class, 6.0)
        dem_rate = demurrage_rate or class_spec.demurrage_rate_usd or DEFAULT_DEMURRAGE_RATES.get(vessel_class, 20_000)
        desp_rate = dem_rate * despatch_rate_fraction

        dem_days = max(0.0, predicted_idle_days - laytime)
        desp_days = max(0.0, laytime - predicted_idle_days)

        total_dem = dem_days * dem_rate
        potential_desp = desp_days * desp_rate
        net_exposure = total_dem - potential_desp

        return DemurrageEstimate(
            predicted_idle_days=round(predicted_idle_days, 2),
            allowed_laytime_days=laytime,
            demurrage_days=round(dem_days, 2),
            demurrage_rate_usd_per_day=dem_rate,
            total_demurrage_usd=round(total_dem, 0),
            despatch_days=round(desp_days, 2),
            despatch_rate_usd_per_day=desp_rate,
            potential_despatch_usd=round(potential_desp, 0),
            net_demurrage_exposure_usd=round(net_exposure, 0),
        )

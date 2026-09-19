"""
Charter-AI — Contract Optimization Service (Phase 9 Upgrade).

Wraps ContractOptimizer and RiskAwareContractOptimizer.
"""

from typing import Dict, Any, Optional, Union
from src.optimization.contract_optimizer import (
    ContractOptimizer,
    ContractOptimizationInputs,
    ContractStrategyRecommendation,
    RiskAwareContractOptimizer,
    RiskAwareContractInputs,
    RiskAwareContractRecommendation,
    RiskTolerance,
    VesselAvailability,
    StrategyAllocation,
)


class ContractOptimizationService:
    """
    Service for recommending quantitative contract strategies (Spot, Short-Term, Medium-Term, Hybrid).
    """

    def __init__(self):
        self.optimizer = ContractOptimizer()
        self.risk_aware_optimizer = RiskAwareContractOptimizer()

    def recommend_strategy(self, inputs: ContractOptimizationInputs) -> ContractStrategyRecommendation:
        """
        Legacy rule-compatible strategy recommendation.
        """
        return self.optimizer.recommend(inputs)

    def optimize_risk_aware(
        self,
        inputs: Union[RiskAwareContractInputs, Dict[str, Any]],
    ) -> RiskAwareContractRecommendation:
        """
        Executes quantitative Phase 9 risk-aware contract optimization.
        """
        if isinstance(inputs, dict):
            # Parse risk tolerance
            tol_str = str(inputs.get("risk_tolerance", "MEDIUM")).upper()
            tolerance = RiskTolerance[tol_str] if tol_str in RiskTolerance.__members__ else RiskTolerance.MEDIUM

            # Parse vessel availability
            avail_str = str(inputs.get("vessel_availability", "BALANCED")).upper()
            if avail_str == "TIGHT":
                avail = VesselAvailability.TIGHT
            elif avail_str == "SHORTAGE":
                avail = VesselAvailability.SHORTAGE
            else:
                avail = VesselAvailability.ABUNDANT

            # Parse custom strategies if provided
            custom_strats = None
            if "custom_strategies" in inputs and inputs["custom_strategies"]:
                custom_strats = []
                for s in inputs["custom_strategies"]:
                    custom_strats.append(
                        StrategyAllocation(
                            name=s.get("name", "CUSTOM"),
                            spot_pct=float(s.get("spot_pct", s.get("spot_percentage", 0.0))),
                            short_term_pct=float(s.get("short_term_pct", s.get("short_term_percentage", 0.0))),
                            medium_term_pct=float(s.get("medium_term_pct", s.get("medium_term_percentage", 0.0))),
                        )
                    )

            parsed_inputs = RiskAwareContractInputs(
                cargo_quantity_t=float(inputs.get("cargo_quantity_t", inputs.get("cargo_tonnage", 70000.0))),
                spot_freight_rate=float(inputs.get("spot_freight_rate", inputs.get("base_freight_rate", inputs.get("freight_rate", 20.0)))),
                short_term_freight_rate=inputs.get("short_term_freight_rate"),
                medium_term_freight_rate=inputs.get("medium_term_freight_rate"),
                freight_volatility_pct=float(inputs.get("freight_volatility_pct", 16.0)),
                base_bunker_price=float(inputs.get("base_bunker_price", 650.0)),
                bunker_volatility_pct=float(inputs.get("bunker_volatility_pct", 12.0)),
                sea_distance_nm=float(inputs.get("sea_distance_nm", inputs.get("sailing_distance_nm", 4500.0))),
                service_speed_knots=float(inputs.get("service_speed_knots", 12.5)),
                fuel_consumption_t_day=float(inputs.get("fuel_consumption_t_day", 28.0)),
                expected_wait_days=float(inputs.get("expected_wait_days", 2.0)),
                delivery_deadline_days=inputs.get("delivery_deadline_days"),
                number_of_voyages=int(inputs.get("number_of_voyages", 1)),
                risk_tolerance=tolerance,
                vessel_availability=avail,
                custom_strategies=custom_strats,
                n_simulations=int(inputs.get("n_simulations", 5000)),
                seed=int(inputs.get("seed", 42)),
            )
        else:
            parsed_inputs = inputs

        return self.risk_aware_optimizer.optimize_contract_strategy(parsed_inputs)

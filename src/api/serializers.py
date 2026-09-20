"""
DockInsights — Pydantic Response Serializers.

API response models used by FastAPI for automatic OpenAPI documentation
and response validation.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# Common
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool
    model_version: Optional[str] = None
    sih_demo_mode: bool = True
    data_mode: str = "SYNTHETIC_DEMO"
    provenance_status: str = "SYNTHETIC_DEMO"


# =============================================================================
# Ports
# =============================================================================

class PortResponse(BaseModel):
    port_id: str
    port_name: str
    state: str
    country: str
    latitude: float
    longitude: float
    port_type: str
    operator: str
    berths_total: Optional[int] = None
    max_draft_m: Optional[float] = None
    max_loa_m: Optional[float] = None
    max_beam_m: Optional[float] = None
    max_dwt: Optional[int] = None
    annual_capacity_mtpa: Optional[str] = None
    primary_cargo: Optional[str] = None


class PortCongestionResponse(BaseModel):
    port_id: str
    date: date
    vessels_waiting: int
    avg_waiting_time_days: float
    berth_occupancy_pct: float


# =============================================================================
# Routes
# =============================================================================

class RouteResponse(BaseModel):
    origin_port_id: str
    origin_port_name: str
    origin_country: str
    destination_port_id: str
    destination_port_name: str
    great_circle_nm: float
    est_sailing_distance_nm: float
    typical_cargo: str
    routing_note: Optional[str] = None


# =============================================================================
# Vessels
# =============================================================================

class VesselClassResponse(BaseModel):
    class_name: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    draft_max_m: float
    loa_max_m: float
    beam_max_m: float


class VesselCompatibilityResponse(BaseModel):
    vessel_class: str
    is_compatible: bool
    violations: List[str] = []


class VesselSelectionResponse(BaseModel):
    origin_port_id: str
    destination_port_id: str
    feasible: List[VesselCompatibilityResponse]
    excluded: List[VesselCompatibilityResponse]


# =============================================================================
# Decision Explanation Schemas
# =============================================================================
class AlternativeExplanation(BaseModel):
    vessel_class: str
    reasons_rejected: List[str]

class ExplainabilityReport(BaseModel):
    recommendation_summary: str
    primary_reasons: List[str]
    alternatives_rejected: List[AlternativeExplanation]


# =============================================================================
# Forecast
# =============================================================================

class ForecastPointResponse(BaseModel):
    date: date
    predicted_rate: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None


class FreightForecastApiResponse(BaseModel):
    current_rate: float
    forecast_rate: float
    lower_bound: float
    upper_bound: float
    trend: str  # "rising", "falling", "stable"
    confidence: float
    model_used: str
    metrics: Dict[str, Any] = {}
    data_scope: Optional[str] = None
    fallback_level: Optional[str] = None
    training_observations: Optional[int] = None
    data_quality: Optional[str] = None
    data_mode: Optional[str] = "SYNTHETIC_DEMO"
    provenance_status: Optional[str] = "SYNTHETIC_DEMO"
    is_verified_external: Optional[bool] = False


class ForecastResponse(BaseModel):
    current_rate: float
    forecast_rate: float
    lower_bound: float
    upper_bound: float
    trend: str  # "rising", "falling", "stable"
    confidence: float
    model_used: str
    metrics: Dict[str, Any] = {}
    data_scope: Optional[str] = None
    fallback_level: Optional[str] = None
    training_observations: Optional[int] = None
    data_quality: Optional[str] = None
    data_mode: Optional[str] = "SYNTHETIC_DEMO"
    provenance_status: Optional[str] = "SYNTHETIC_DEMO"
    is_verified_external: Optional[bool] = False
    # Optional backwards compatibility fields
    origin_port_id: Optional[str] = None
    destination_port_id: Optional[str] = None
    vessel_class: Optional[str] = None
    horizon_days: Optional[int] = None
    model_version: Optional[str] = None
    series: Optional[List[ForecastPointResponse]] = None


# =============================================================================
# Risk
# =============================================================================

class RiskDimensionResponse(BaseModel):
    score: float
    level: str
    detail: str


class RiskAssessmentResponse(BaseModel):
    composite_score: float
    level: str
    breakdown: Dict[str, RiskDimensionResponse]
    dominant_risk: Optional[str] = None
    recommendation: str


class RiskSimulationRequest(BaseModel):
    cargo_quantity_t: float = Field(..., gt=0, description="Cargo quantity in metric tonnes")
    base_freight_rate: float = Field(default=20.0, description="Freight rate ($/MT)")
    freight_volatility_pct: float = Field(default=15.0, description="Market freight volatility (%)")
    freight_rate_p10: Optional[float] = None
    freight_rate_p90: Optional[float] = None
    base_bunker_price: float = Field(default=650.0, description="Bunker fuel price ($/MT)")
    bunker_volatility_pct: float = Field(default=12.0, description="Bunker price volatility (%)")
    sea_distance_nm: float = Field(default=4500.0, description="Sea sailing distance (nautical miles)")
    service_speed_knots: float = Field(default=12.5, description="Vessel service speed (knots)")
    fuel_consumption_t_day: float = Field(default=28.0, description="Daily bunker consumption (MT/day)")
    expected_wait_days: float = Field(default=2.0, description="Expected port waiting days")
    p90_wait_days: Optional[float] = None
    port_handling_rate_t_day: float = Field(default=15000.0, description="Loading/discharge rate (MT/day)")
    agreed_laytime_days: Optional[float] = None
    demurrage_rate_usd_day: float = Field(default=20000.0, description="Daily demurrage rate ($/day)")
    port_charges_usd: float = Field(default=45000.0, description="Port dues and charges ($)")
    canal_charges_usd: float = Field(default=0.0, description="Canal transit tolls ($)")
    delivery_deadline_days: Optional[float] = Field(default=25.0, description="Delivery deadline window (days)")
    vessel_availability_probability: float = Field(default=0.95, description="Probability vessel remains available")
    n_simulations: int = Field(default=10000, description="Number of Monte Carlo iterations")
    seed: int = Field(default=42, description="Simulation seed for exact reproducibility")
    cost_threshold_usd: Optional[float] = None


class RiskSimulationResponse(BaseModel):
    expected_cost: float
    p10_cost: float
    p50_cost: float
    p90_cost: float
    demurrage_probability: float
    late_delivery_probability: float
    risk_score: float
    probability_of_infeasibility: float = 0.0
    scenarios: Dict[str, Any]
    cost_distribution: List[Dict[str, Any]] = []
    risk_assessment: Optional[Dict[str, Any]] = None
    seed: int = 42
    n_simulations: int = 10000


# =============================================================================
# Economics
# =============================================================================

class CostBreakdownResponse(BaseModel):
    freight_cost_usd: float
    bunker_cost_usd: float
    load_port_charges_usd: float
    discharge_port_charges_usd: float
    insurance_usd: float
    expected_demurrage_usd: float
    miscellaneous_usd: float


class VoyageEconomicsResponse(BaseModel):
    total_voyage_cost_usd: float
    cost_per_tonne_usd: float
    cargo_tonnage: int
    vessel_class: str
    sailing_days: float
    total_voyage_days: float
    breakdown: CostBreakdownResponse


class DeliveredCostResponse(BaseModel):
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
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Phase 9: Contract Optimization
# =============================================================================

class ContractOptimizationApiRequest(BaseModel):
    cargo_quantity_t: float = Field(..., gt=0, description="Cargo quantity in metric tonnes")
    spot_freight_rate: float = Field(default=20.0, description="Spot freight rate ($/MT)")
    short_term_freight_rate: Optional[float] = None
    medium_term_freight_rate: Optional[float] = None
    freight_volatility_pct: float = Field(default=16.0, description="Market freight volatility (%)")
    base_bunker_price: float = Field(default=650.0, description="Bunker fuel price ($/MT)")
    sea_distance_nm: float = Field(default=4500.0, description="Sea sailing distance (nautical miles)")
    delivery_deadline_days: Optional[float] = Field(default=26.0, description="Delivery deadline window (days)")
    risk_tolerance: str = Field(default="MEDIUM", description="Risk tolerance: LOW, MEDIUM, or HIGH")
    vessel_availability: str = Field(default="TIGHT", description="Vessel availability: ABUNDANT, TIGHT, or SHORTAGE")
    number_of_voyages: int = Field(default=1, description="Number of required voyages")
    custom_strategies: Optional[List[Dict[str, Any]]] = None
    n_simulations: int = Field(default=5000, description="Number of Monte Carlo simulation runs")
    seed: int = Field(default=42, description="Simulation seed")


class ContractOptimizationApiResponse(BaseModel):
    recommended_strategy: str
    spot_percentage: float
    short_term_percentage: float
    medium_term_percentage: float
    expected_cost: float
    p90_cost: float
    risk_score: float
    flexibility_score: float
    reasons: List[str]
    evaluated_strategies: List[Dict[str, Any]] = []


# =============================================================================
# Recommendation (Primary endpoint)
# =============================================================================

class ContractRecommendationResponse(BaseModel):
    contract_type: str
    duration_months: Optional[int] = None
    reasoning: str
    confidence: float


class PrimaryRecommendation(BaseModel):
    vessel_class: str
    optimal_booking_window: Dict[str, str]  # {"start": ..., "end": ...}
    contract: ContractRecommendationResponse
    estimated_rate_usd_per_day: float
    confidence: float
    reasoning: str


class ExplainabilityResponse(BaseModel):
    top_factors: List[Dict[str, float]]  # [{"feature": ..., "impact": ..., "direction": ...}]


class RecommendationRequest(BaseModel):
    """Request body for the /recommend endpoint."""
    origin_port_id: str
    destination_port_id: str
    cargo_type: str = "coal"
    cargo_tonnage: int = Field(..., gt=0, description="Cargo quantity in tonnes")
    earliest_date: date
    latest_date: date
    risk_appetite: str = Field(
        default="moderate",
        description="Risk tolerance: low, moderate, or high",
    )


class RecommendationResponse(BaseModel):
    request_id: str
    generated_at: datetime
    recommendation: PrimaryRecommendation
    alternatives: List[PrimaryRecommendation] = []
    forecast: Optional[ForecastResponse] = None
    economics: Optional[VoyageEconomicsResponse] = None
    risk: Optional[RiskAssessmentResponse] = None
    vessel_compatibility: Optional[VesselSelectionResponse] = None

class AnalyzeVoyageRequest(BaseModel):
    cargo_type: str
    cargo_quantity: float
    origin: str
    destination: str
    required_delivery_date: date
    number_of_voyages: int
    contract_preference: Optional[str] = "ANY"

class AnalyzeVoyageResponse(BaseModel):
    status: str
    error_message: Optional[str] = None
    market_forecast: Dict[str, Any] = Field(default_factory=dict)
    recommended_vessel: Dict[str, Any] = Field(default_factory=dict)
    port_analysis: Dict[str, Any] = Field(default_factory=dict)
    voyage_economics: Dict[str, Any] = Field(default_factory=dict)
    risk_analysis: Dict[str, Any] = Field(default_factory=dict)
    contract_strategy: Dict[str, Any] = Field(default_factory=dict)
    final_recommendation: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[ExplainabilityReport] = None


# =============================================================================
# Phase 10: Unified Decision Engine Schemas
# =============================================================================

class MarketAnalysisResponse(BaseModel):
    current_rate: float
    forecast: float
    direction: str
    confidence: float
    volatility: float


class RecommendedPlanResponse(BaseModel):
    vessel_class: str
    vessel_count: int
    voyages: int
    cargo_allocation: List[float] = Field(default_factory=list)
    port_compatibility: Dict[str, Any] = Field(default_factory=dict)
    utilization: float
    total_cost: float
    cost_per_tonne: float
    voyage_duration: float
    expected_waiting: float
    demurrage_probability: float
    delivery_probability: float
    risk_score: float


class DecisionExplanationResponse(BaseModel):
    summary: str
    primary_reasons: List[str]
    tradeoff_analysis: str
    alternatives_rejected: List[Dict[str, Any]] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    decision_id: str
    timestamp: str
    model_versions: Dict[str, str] = Field(default_factory=dict)
    data_versions: Dict[str, str] = Field(default_factory=dict)
    request_summary: Dict[str, Any] = Field(default_factory=dict)
    market_analysis: MarketAnalysisResponse
    freight_forecast: Dict[str, Any] = Field(default_factory=dict)
    market_timing: Dict[str, Any] = Field(default_factory=dict)
    recommended_plan: RecommendedPlanResponse
    alternative_plans: List[Dict[str, Any]] = Field(default_factory=list)
    economics: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    contract_strategy: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    explanation: DecisionExplanationResponse

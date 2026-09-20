"""
DockInsights — Normalized Pydantic Data Schemas (V2).

Validation schemas for ingested datasets, loaders, and API boundaries.
Supports canonical attributes and legacy aliases via ConfigDict(populate_by_name=True).
"""

from datetime import date, datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# 1. Ports
# =============================================================================

class PortSchema(BaseModel):
    """Schema for Port records."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    port_id: str = Field(..., description="Unique port identifier, e.g. IND_VZG")
    port_name: str = Field(..., description="Full name of the port")
    country: str = Field(..., description="ISO 3-letter or common country code")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    max_draft_m: float = Field(..., ge=0.0, le=35.0, alias="max_draft")
    max_loa_m: float = Field(..., ge=0.0, le=500.0, alias="max_loa")
    max_beam_m: float = Field(..., ge=0.0, le=80.0, alias="max_beam")
    cargo_handling_rate_mt_day: float = Field(default=25000.0, ge=0.0, alias="cargo_handling_rate")
    berth_count: int = Field(default=4, ge=1, alias="berthing_capacity")
    coal_terminal: bool = Field(default=True)
    iron_ore_terminal: bool = Field(default=False)
    grain_terminal: bool = Field(default=False)
    tidal_restriction: bool = Field(default=False)
    night_navigation_restriction: bool = Field(default=False)
    source: str = Field(default="SYNTHETIC_DEMO")
    source_date: Optional[date] = None
    confidence_level: str = Field(default="LOW", description="HIGH, MEDIUM, LOW")

    # Compatibility optional fields
    state: Optional[str] = None
    max_dwt: Optional[int] = None
    primary_cargo: Optional[str] = None


# =============================================================================
# 2. Vessels (Fleet Assets)
# =============================================================================

class VesselSchema(BaseModel):
    """Schema for individual dry bulk vessel specifications."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    vessel_id: str = Field(..., description="Unique vessel ID e.g. VSL_001")
    imo_number: str = Field(..., description="Unique 7-digit IMO number")
    vessel_name: str = Field(..., description="Vessel name")
    vessel_class: str = Field(..., alias="vessel_type", description="Handysize, Supramax, Panamax, Capesize, VLOC")
    dwt: int = Field(..., ge=5000, le=500000)
    loa_m: float = Field(..., ge=50.0, le=450.0, alias="loa")
    beam_m: float = Field(..., ge=10.0, le=75.0, alias="beam")
    max_draft_m: float = Field(..., ge=4.0, le=30.0, alias="draft")
    service_speed_knots: float = Field(default=13.0, ge=5.0, le=25.0, alias="speed")
    ballast_speed_knots: float = Field(default=13.5, ge=5.0, le=25.0)
    laden_speed_knots: float = Field(default=12.5, ge=5.0, le=25.0)
    fuel_consumption_mt_day: float = Field(default=30.0, ge=5.0, le=120.0, alias="fuel_consumption")
    age_years: int = Field(default=5, ge=0, le=50, alias="age")
    current_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    current_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    availability_status: str = Field(default="AVAILABLE")
    available_from: Optional[date] = None
    data_source: str = Field(default="SYNTHETIC_DEMO", alias="source")
    source_date: Optional[date] = None

    @field_validator("imo_number", mode="before")
    @classmethod
    def coerce_imo(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""


# =============================================================================
# 3. Vessel Class Models
# =============================================================================

class VesselClassSchema(BaseModel):
    """Schema for vessel class dimension boundaries."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    vessel_class: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    loa_min_m: Optional[float] = None
    loa_max_m: Optional[float] = None
    beam_min_m: Optional[float] = None
    beam_max_m: Optional[float] = None
    draft_min_m: Optional[float] = None
    draft_max_m: Optional[float] = None
    typical_cargo: Optional[str] = None
    data_type: Optional[str] = None
    source_note: Optional[str] = None


# =============================================================================
# 4. Routes
# =============================================================================

class RouteSchema(BaseModel):
    """Schema for maritime shipping routes."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    route_id: str
    origin_port: str = Field(..., alias="origin")
    destination_port: str = Field(..., alias="destination")
    distance_nm: float = Field(..., ge=10.0, le=30000.0)
    typical_duration_days: float = Field(..., ge=0.5, le=120.0)
    route_type: str = Field(default="direct")
    seasonal_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 5. Freight Rates
# =============================================================================

class FreightRateSchema(BaseModel):
    """Schema for freight rate observations."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    origin: str
    destination: str
    vessel_class: str = Field(..., alias="vessel_type")
    cargo_type: str = Field(default="thermal_coal")
    freight_rate: float = Field(..., ge=0.0, le=500.0)
    currency: str = Field(default="USD")
    unit: str = Field(default="per_tonne")
    source: str = Field(default="SYNTHETIC_DEMO")

    # Compatibility properties
    origin_port_id: Optional[str] = None
    destination_port_id: Optional[str] = None
    freight_rate_usd_per_day: Optional[float] = None
    freight_rate_usd_per_tonne: Optional[float] = None


# =============================================================================
# 6. Bunker Prices
# =============================================================================

class BunkerPriceSchema(BaseModel):
    """Schema for bunker fuel prices."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    location: str
    fuel_type: str
    price_usd_mt: float = Field(..., ge=50.0, le=2500.0)
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 7. Dry Bulk Indices
# =============================================================================

class DryBulkIndexSchema(BaseModel):
    """Schema for dry bulk index observations."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    index_name: str
    value: float = Field(..., ge=0.0, le=20000.0)
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 8. Commodity Prices
# =============================================================================

class CommodityPriceSchema(BaseModel):
    """Schema for commodity market prices."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    commodity_name: str = Field(..., alias="name")
    price: float = Field(..., ge=0.0)
    currency: str = Field(default="USD")
    unit: str = Field(default="tonne")
    source: str = Field(default="SYNTHETIC_DEMO")


class CommoditySchema(BaseModel):
    """Schema for commodity specifications."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    commodity_id: str
    commodity_name: str = Field(..., alias="name")
    category: Optional[str] = Field(default=None, alias="type")
    calorific_value_kcal_kg: Optional[str] = None
    ash_content_pct: Optional[str] = None
    typical_parcel_size_tonnes: Optional[str] = None
    typical_vessel_class: Optional[str] = None


# =============================================================================
# 9. Port Congestion
# =============================================================================

class CongestionSchema(BaseModel):
    """Schema for port congestion observations."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    port: str
    vessels_waiting: int = Field(default=0, ge=0)
    average_waiting_days: float = Field(default=0.0, ge=0.0, alias="average_waiting_time")
    berth_occupancy_pct: float = Field(default=0.0, ge=0.0, le=100.0, alias="berth_occupancy")
    congestion_index: float = Field(default=0.0, ge=0.0, le=100.0)
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 10. Weather Observations
# =============================================================================

class WeatherSchema(BaseModel):
    """Schema for maritime weather observations."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    port: str
    wind_speed_kmh: Optional[float] = Field(default=None, ge=0.0, le=350.0, alias="wind_speed")
    wave_height_m: Optional[float] = Field(default=None, ge=0.0, le=25.0, alias="wave_height")
    rainfall_mm: Optional[float] = Field(default=0.0, ge=0.0)
    cyclone_alert_level: str = Field(default="None", alias="cyclone_alert")
    source: str = Field(default="SYNTHETIC_DEMO")

    @field_validator("cyclone_alert_level", mode="before")
    @classmethod
    def coerce_cyclone_alert(cls, v: Any) -> str:
        if v is None or str(v).lower() in ("none", "nan", ""):
            return "None"
        return str(v).strip()


# =============================================================================
# 11. Economic Indicators
# =============================================================================

class EconomicIndicatorSchema(BaseModel):
    """Schema for macroeconomic indicator observations."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    indicator_name: str = Field(..., alias="indicator")
    value: float
    unit: str = Field(default="value")
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 12. Geopolitical Events
# =============================================================================

class EventSchema(BaseModel):
    """Schema for operational and geopolitical events."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: date
    event_type: str
    affected_region: Optional[str] = None
    severity: str = Field(default="low", alias="impact_severity")
    description: Optional[str] = None
    source: str = Field(default="SYNTHETIC_DEMO")


# =============================================================================
# 13. Vessel AIS Positions
# =============================================================================

class VesselAISSchema(BaseModel):
    """Schema for Vessel AIS real-time/historical telemetry."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    imo_number: str
    timestamp: datetime
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    speed_knots: float = Field(..., ge=0.0, le=35.0)
    heading: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    destination: Optional[str] = None
    eta: Optional[datetime] = None
    nav_status: str = Field(default="under_way")
    source: str = Field(default="SYNTHETIC_DEMO")

    @field_validator("imo_number", mode="before")
    @classmethod
    def coerce_imo(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

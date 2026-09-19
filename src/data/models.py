"""
Charter-AI — Normalized SQLAlchemy ORM Models (V2).

Normalized schema architecture supporting historical and current maritime data across 12 domains:
1. Ports
2. Vessels (individual fleet assets & specs)
3. Vessel Classes (class taxonomy Handysize - VLOC)
4. Routes
5. Freight Rates
6. Bunker Prices
7. Dry Bulk Indices
8. Commodity Prices & Specifications
9. Port Congestion
10. Weather Observations
11. Economic Indicators
12. Geopolitical / Operational Events
13. Vessel AIS Positions & Availability Tracking
"""

from datetime import date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import BigInteger

from src.data.db import Base


# =============================================================================
# 1. Ports
# =============================================================================

class Port(Base):
    """Normalized Port entity with bathymetry, restrictions, and PostGIS geometry."""

    __tablename__ = "ports"

    # Required canonical fields
    port_id = Column(String(20), primary_key=True)
    port_name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    max_draft_m = Column(Float, nullable=False)
    max_loa_m = Column(Float, nullable=False)
    max_beam_m = Column(Float, nullable=False)
    cargo_handling_rate_mt_day = Column(Float, default=25000.0)
    berth_count = Column(Integer, default=4)
    coal_terminal = Column(Boolean, default=True)
    iron_ore_terminal = Column(Boolean, default=False)
    grain_terminal = Column(Boolean, default=False)
    tidal_restriction = Column(Boolean, default=False)
    night_navigation_restriction = Column(Boolean, default=False)
    source = Column(String(100), default="SYNTHETIC_DEMO")
    source_date = Column(Date, nullable=True)
    confidence_level = Column(String(20), default="LOW")

    # Additional operational and spatial metadata
    state = Column(String(100), nullable=True)
    geom = Column(Geometry("POINT", srid=4326), nullable=True)
    port_type = Column(String(100), nullable=True)
    operator = Column(String(200), nullable=True)
    berths_total = Column(Integer, nullable=True)
    max_dwt = Column(Integer, nullable=True)
    annual_capacity_mtpa = Column(String(50), nullable=True)
    primary_cargo = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=True)
    source_note = Column(Text, nullable=True)


# =============================================================================
# 2. Vessels (Individual Fleet Assets)
# =============================================================================

class Vessel(Base):
    """Individual dry bulk vessel asset with physical and operational characteristics."""

    __tablename__ = "vessels"

    vessel_id = Column(String(30), primary_key=True)
    imo_number = Column(String(20), unique=True, index=True, nullable=False)
    vessel_name = Column(String(100), nullable=False)
    vessel_class = Column(String(50), nullable=False, index=True)
    dwt = Column(Integer, nullable=False)
    loa_m = Column(Float, nullable=False)
    beam_m = Column(Float, nullable=False)
    max_draft_m = Column(Float, nullable=False)
    service_speed_knots = Column(Float, nullable=False, default=13.0)
    ballast_speed_knots = Column(Float, nullable=False, default=13.5)
    laden_speed_knots = Column(Float, nullable=False, default=12.5)
    fuel_consumption_mt_day = Column(Float, nullable=False, default=30.0)
    age_years = Column(Integer, nullable=False, default=5)
    current_latitude = Column(Float, nullable=True)
    current_longitude = Column(Float, nullable=True)
    availability_status = Column(String(30), default="AVAILABLE")
    available_from = Column(Date, nullable=True)
    data_source = Column(String(100), default="SYNTHETIC_DEMO")
    source_date = Column(Date, nullable=True)


# =============================================================================
# 3. Vessel Class Specifications
# =============================================================================

class VesselClassModel(Base):
    """Vessel class specifications (Handysize through VLOC)."""

    __tablename__ = "vessel_classes"

    class_id = Column(String(30), primary_key=True)
    class_name = Column(String(50), nullable=False)
    dwt_min = Column(Integer, nullable=False)
    dwt_max = Column(Integer, nullable=False)
    typical_dwt = Column(Integer, nullable=False)
    loa_min_m = Column(Float)
    loa_max_m = Column(Float)
    beam_min_m = Column(Float)
    beam_max_m = Column(Float)
    draft_min_m = Column(Float)
    draft_max_m = Column(Float)
    typical_cargo = Column(Text)
    data_type = Column(String(50))
    source_note = Column(Text)


# =============================================================================
# 4. Routes
# =============================================================================

class Route(Base):
    """Normalized shipping route between origin and destination port."""

    __tablename__ = "routes"

    route_id = Column(String(50), primary_key=True)
    origin_port = Column(String(20), nullable=False, index=True)
    destination_port = Column(String(20), nullable=False, index=True)
    distance_nm = Column(Float, nullable=False)
    typical_duration_days = Column(Float, nullable=False)
    route_type = Column(String(50), default="direct")
    seasonal_factor = Column(Float, default=1.0)
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility attributes
    origin_port_id = Column(String(20), nullable=True)
    destination_port_id = Column(String(20), nullable=True)
    origin_port_name = Column(String(100), nullable=True)
    destination_port_name = Column(String(100), nullable=True)
    origin_country = Column(String(50), nullable=True)
    great_circle_nm = Column(Float, nullable=True)
    est_sailing_distance_nm = Column(Float, nullable=True)
    typical_cargo = Column(Text, nullable=True)
    routing_note = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=True)


# =============================================================================
# 5. Freight Rates
# =============================================================================

class FreightRate(Base):
    """Historical or spot freight rate observation."""

    __tablename__ = "freight_rates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    origin = Column(String(20), nullable=False, index=True)
    destination = Column(String(20), nullable=False, index=True)
    vessel_class = Column(String(50), nullable=False, index=True)
    cargo_type = Column(String(50), nullable=False, default="thermal_coal")
    freight_rate = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    unit = Column(String(30), default="per_tonne")
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility columns
    origin_port_id = Column(String(20), nullable=True)
    destination_port_id = Column(String(20), nullable=True)
    freight_rate_usd_per_day = Column(Float, nullable=True)
    freight_rate_usd_per_tonne = Column(Float, nullable=True)
    bdi_proxy_index = Column(Float, nullable=True)
    data_type = Column(String(50), nullable=True)


# =============================================================================
# 6. Bunker Prices
# =============================================================================

class BunkerPrice(Base):
    """Bunker fuel prices across major bunkering hubs."""

    __tablename__ = "bunker_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    location = Column(String(50), nullable=False, index=True)  # Singapore, Fujairah, Rotterdam, Visakhapatnam
    fuel_type = Column(String(30), nullable=False)  # VLSFO, MGO, IFO380
    price_usd_mt = Column(Float, nullable=False)
    source = Column(String(100), default="SYNTHETIC_DEMO")


# =============================================================================
# 7. Dry Bulk Indices
# =============================================================================

class DryBulkIndex(Base):
    """Baltic Exchange dry bulk freight indices."""

    __tablename__ = "dry_bulk_indices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    index_name = Column(String(50), nullable=False, index=True)  # BDI, BCI, BPI, BSI, BHSI
    value = Column(Float, nullable=False)
    source = Column(String(100), default="SYNTHETIC_DEMO")


# =============================================================================
# 8. Commodity Prices & Reference Data
# =============================================================================

class CommodityPrice(Base):
    """Dry bulk commodity benchmark prices (Coal, Iron Ore, Grain)."""

    __tablename__ = "commodity_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    commodity_name = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    unit = Column(String(30), default="tonne")
    source = Column(String(100), default="SYNTHETIC_DEMO")


class Commodity(Base):
    """Commodity physical reference specification."""

    __tablename__ = "commodities"

    commodity_id = Column(String(30), primary_key=True)
    commodity_name = Column(String(100), nullable=False)
    category = Column(String(50))
    typical_origin_countries = Column(Text)
    calorific_value_kcal_kg = Column(String(30))
    ash_content_pct = Column(String(30))
    typical_parcel_size_tonnes = Column(String(50))
    typical_vessel_class = Column(String(50))
    primary_indian_use = Column(String(100))
    data_type = Column(String(50))
    source_note = Column(Text)


# =============================================================================
# 9. Port Congestion
# =============================================================================

class PortCongestion(Base):
    """Port waiting queue and berth occupancy tracking."""

    __tablename__ = "port_congestion"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    port = Column(String(20), nullable=False, index=True)
    vessels_waiting = Column(Integer, default=0)
    average_waiting_days = Column(Float, default=0.0)
    berth_occupancy_pct = Column(Float, default=0.0)
    congestion_index = Column(Float, default=0.0)
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility attributes
    port_id = Column(String(20), nullable=True)
    avg_waiting_time_days = Column(Float, nullable=True)
    data_type = Column(String(50), nullable=True)


# Backwards compatibility alias
Congestion = PortCongestion


# =============================================================================
# 10. Weather Observations
# =============================================================================

class Weather(Base):
    """Weather and sea-state conditions at ports and maritime transit zones."""

    __tablename__ = "weather"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    port = Column(String(20), nullable=False, index=True)
    wind_speed_kmh = Column(Float, nullable=True)
    wave_height_m = Column(Float, nullable=True)
    rainfall_mm = Column(Float, nullable=True)
    cyclone_alert_level = Column(String(30), default="None")
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility attributes
    port_id = Column(String(20), nullable=True)
    sea_state = Column(String(30), nullable=True)
    data_type = Column(String(50), nullable=True)


# =============================================================================
# 11. Economic Indicators
# =============================================================================

class EconomicIndicator(Base):
    """Macroeconomic indicators (crude oil, FX, GDP/PMI indices)."""

    __tablename__ = "economic_indicators"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    indicator_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(30), default="value")
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility wide columns for existing code
    brent_crude_usd_bbl = Column(Float, nullable=True)
    bunker_fuel_vlsfo_usd_tonne = Column(Float, nullable=True)
    newcastle_coal_price_usd_tonne = Column(Float, nullable=True)
    india_coal_import_demand_index = Column(Float, nullable=True)
    usd_inr_rate = Column(Float, nullable=True)
    china_pmi_manufacturing = Column(Float, nullable=True)
    data_type = Column(String(50), nullable=True)


# =============================================================================
# 12. Geopolitical / Operational Events
# =============================================================================

class GeopoliticalEvent(Base):
    """Disruption events (chokepoints, strikes, storms, regulatory shifts)."""

    __tablename__ = "events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    affected_region = Column(String(100), nullable=True)
    severity = Column(String(30), default="low")
    description = Column(Text, nullable=True)
    source = Column(String(100), default="SYNTHETIC_DEMO")

    # Compatibility attributes
    event_id = Column(Integer, nullable=True)
    event_date = Column(Date, nullable=True)
    event_name = Column(String(100), nullable=True)
    event_category = Column(String(50), nullable=True)
    affected_ports_or_origins = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=True)


# Backwards compatibility alias
Event = GeopoliticalEvent


# =============================================================================
# 13. Vessel AIS Positions & Availability
# =============================================================================

class VesselAISPosition(Base):
    """Historical and real-time AIS vessel telemetry and tracking."""

    __tablename__ = "vessel_ais_positions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    imo_number = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed_knots = Column(Float, nullable=False)
    heading = Column(Float, nullable=True)
    destination = Column(String(50), nullable=True)
    eta = Column(DateTime, nullable=True)
    nav_status = Column(String(50), default="under_way")
    source = Column(String(100), default="SYNTHETIC_DEMO")


# =============================================================================
# 14. ML Model Registry & Audit Logs
# =============================================================================

class ModelRegistryEntry(Base):
    """ML model version tracking."""

    __tablename__ = "model_registry"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(30), nullable=False)
    model_type = Column(String(50))
    hyperparameters = Column(JSONB)
    metrics = Column(JSONB)
    artifact_path = Column(String(500))
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)


class RecommendationLog(Base):
    """Audit log of recommendations generated."""

    __tablename__ = "recommendation_logs"

    request_id = Column(String(36), primary_key=True)  # UUID
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    request_params = Column(JSONB)
    recommendation = Column(JSONB)
    model_version = Column(String(30))

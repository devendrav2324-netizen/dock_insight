"""
Charter-AI — Application Configuration.

Loads settings from environment variables / .env file using Pydantic BaseSettings.
All configuration is centralized here to avoid scattered os.getenv() calls.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = charter-ai/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- Database ----------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "charter_ai"
    postgres_user: str = "charter_ai"
    postgres_password: str = "changeme_in_production"
    database_url: str | None = None  # Override full URL if needed

    @property
    def db_url_async(self) -> str:
        """Construct async database URL."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def db_url_sync(self) -> str:
        """Construct sync database URL (for Alembic, ingestion scripts)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ---------- API ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    sih_demo_mode: bool = True

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    # ---------- Paths ----------
    raw_data_dir: str = str(_PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(_PROJECT_ROOT / "data" / "processed")
    trained_models_dir: str = str(_PROJECT_ROOT / "trained_models")

    # ---------- ML ----------
    model_version: str = "v0.1.0"
    forecast_default_horizon_days: int = 30

    # ---------- Logging ----------
    log_level: str = "INFO"
    log_format: str = "json"

    # ==========================================================================
    # Business Defaults — [DEMO/CONFIGURABLE]
    #
    # These were previously hard-coded across multiple modules. They are now
    # centralized here so that every module reads from one source of truth.
    # Override via environment variables or .env file.
    # ==========================================================================

    # -- Voyage Economics --
    default_route_distance_nm: float = 4500.0
    default_vessel_speed_knots: float = 12.5
    default_fuel_price_usd_per_t: float = 600.0
    default_daily_hire_cost_usd: float = 14000.0
    default_positioning_distance_nm: float = 500.0
    default_load_port_cost_usd: float = 75000.0
    default_discharge_port_cost_usd: float = 75000.0
    default_daily_demurrage_rate_usd: float = 18000.0
    default_waiting_days: float = 2.0
    default_other_costs_usd: float = 10000.0
    default_daily_fuel_consumption_tpd: float = 28.0

    # -- Forecast (demo defaults when ML model is not available) --
    default_forecast_rate_usd: float = 22.50
    default_current_freight_rate_usd: float = 21.0
    default_forecast_direction: str = "RISING"
    default_forecast_uncertainty: str = "MODERATE"

    # -- Risk (demo defaults for synthetic risk inputs) --
    default_market_volatility_pct: float = 12.0
    default_port_wait_days: float = 2.0
    default_wave_height_m: float = 2.0
    default_storm_warning: bool = False
    default_available_vessels_in_region: int = 8
    default_route_conflict_level: float = 2.0
    default_maintenance_due: bool = False

    # -- Port Constraints (generous defaults for unknown ports) --
    default_port_max_draft_m: float = 25.0
    default_port_max_loa_m: float = 350.0
    default_port_max_beam_m: float = 60.0
    default_port_max_dwt: int = 200000

    # -- Market & Vessel Decision Thresholds (Problem 6 Configurable Thresholds) --
    availability_tight_threshold: float = 1.25
    availability_balanced_threshold: float = 2.50
    momentum_rising_threshold_pct: float = 2.5
    momentum_falling_threshold_pct: float = -2.5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

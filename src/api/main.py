"""
DockInsights — FastAPI Application Factory.

Creates and configures the FastAPI app with CORS, routes, and lifespan events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import economics, forecast, health, ports, recommend, risk, routes, vessels, analyze_voyage, congestion, contracts
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger

# Import services
from src.services.freight_forecast_service import FreightForecastService
from src.services.risk_service import RiskService
from src.services.voyage_economics_service import VoyageEconomicsService
from src.services.vessel_optimization_service import VesselOptimizationService
from src.services.contract_optimization_service import ContractOptimizationService
from src.services.congestion_service import CongestionService


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    logger.info(
        f"DockInsights starting | version={settings.model_version} | "
        f"db={settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # Initialize Core Services (Singletons for the app lifecycle)
    app.state.congestion_service = CongestionService()
    app.state.forecast_service = FreightForecastService()
    app.state.risk_service = RiskService(congestion_service=app.state.congestion_service)
    app.state.economics_service = VoyageEconomicsService(congestion_service=app.state.congestion_service)
    app.state.vessel_service = VesselOptimizationService()
    app.state.contract_service = ContractOptimizationService()

    logger.info("Core services initialized.")

    yield

    logger.info("DockInsights shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="DockInsights",
        description=(
            "Advanced Bulk Vessel Chartering & Freight Intelligence Platform. "
            "Decision-support system for dry-bulk cargo chartering to India's East Coast ports."
        ),
        version=settings.model_version,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules under /api/v1 prefix
    api_prefix = "/api/v1"

    app.include_router(health.router, prefix=api_prefix)
    app.include_router(ports.router, prefix=api_prefix)
    app.include_router(routes.router, prefix=api_prefix)
    app.include_router(vessels.router, prefix=api_prefix)
    app.include_router(forecast.router, prefix=api_prefix)
    app.include_router(congestion.router, prefix=api_prefix)
    app.include_router(recommend.router, prefix=api_prefix)
    app.include_router(economics.router, prefix=api_prefix)
    app.include_router(risk.router, prefix=api_prefix)
    app.include_router(analyze_voyage.router, prefix=api_prefix)
    app.include_router(contracts.router, prefix=api_prefix)

    return app


# Application instance (used by uvicorn)
app = create_app()

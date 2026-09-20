"""
DockInsights — Health Check Route.
"""

from fastapi import APIRouter

from src.api.serializers import HealthResponse
from src.utils.config import get_settings

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.model_version,
        db_connected=True,  # TODO: actual DB ping
        model_version=settings.model_version,
        sih_demo_mode=settings.sih_demo_mode,
    )

"""
DockInsights — FastAPI Dependency Injection.

Provides database sessions, model instances, and service objects
to route handlers via FastAPI's Depends() system.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.db import get_db_session


async def get_session() -> AsyncSession:
    """Yield a database session for request-scoped use."""
    async for session in get_db_session():
        yield session

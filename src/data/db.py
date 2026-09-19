"""
Charter-AI — Database Engine & Session Management.

Provides both async (for FastAPI) and sync (for scripts/ingestion) engines.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Async engine (used by FastAPI)
# ---------------------------------------------------------------------------

_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Lazily create and return the async engine."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.db_url_async,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db_session() -> AsyncSession:
    """
    FastAPI dependency that yields an async DB session.

    Usage in route:
        async def my_route(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Sync engine (used by ingestion scripts, Alembic)
# ---------------------------------------------------------------------------

_sync_engine = None
_sync_session_factory = None


def get_sync_engine():
    """Lazily create and return the sync engine."""
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        _sync_engine = create_engine(
            settings.db_url_sync,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
    return _sync_engine


def get_sync_session_factory() -> sessionmaker[Session]:
    """Return the sync session factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            expire_on_commit=False,
        )
    return _sync_session_factory

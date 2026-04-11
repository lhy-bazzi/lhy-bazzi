"""PostgreSQL async connection management using SQLAlchemy 2.0 + asyncpg."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from app.config import get_settings

# Module-level engine and session factory (initialized on startup)
_engine = None
_AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Create the async engine and session factory. Call on app startup."""
    global _engine, _AsyncSessionLocal

    settings = get_settings().database
    logger.info("Connecting to PostgreSQL: {}", settings.url.split("@")[-1])

    _engine = create_async_engine(
        settings.url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        echo=False,
    )
    _AsyncSessionLocal = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Verify connectivity
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connected successfully.")


async def close_db() -> None:
    """Dispose the engine. Call on app shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("PostgreSQL connection closed.")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for a database session."""
    if _AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def health_check() -> dict:
    """Return DB connectivity status."""
    try:
        if _engine is None:
            return {"status": "error", "message": "not initialized"}
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

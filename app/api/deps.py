"""FastAPI dependency injection helpers for infrastructure clients."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import database, es_client, llm_provider, milvus_client, redis_client


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a SQLAlchemy async database session."""
    async with database.get_db_session() as session:
        yield session


async def get_redis_dep():
    """Return the Redis async client."""
    return redis_client.get_redis()


def get_milvus_dep():
    """Return the Milvus client."""
    return milvus_client.get_milvus()


def get_es_dep():
    """Return the AsyncElasticsearch client."""
    return es_client.get_es()


def get_llm_dep():
    """Return the LLM provider."""
    return llm_provider.get_llm()

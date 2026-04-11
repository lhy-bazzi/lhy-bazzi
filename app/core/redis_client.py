"""Redis async client with key-prefix helpers and JSON convenience methods."""

from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis
from loguru import logger

from app.config import get_settings

_redis: aioredis.Redis | None = None


async def init_redis() -> None:
    """Create Redis connection pool. Call on app startup."""
    global _redis
    settings = get_settings().redis
    logger.info("Connecting to Redis: {}", settings.url)
    _redis = aioredis.from_url(
        settings.url,
        encoding="utf-8",
        decode_responses=True,
    )
    # Verify connectivity
    await _redis.ping()
    logger.info("Redis connected successfully.")


async def close_redis() -> None:
    """Close the Redis connection pool. Call on app shutdown."""
    global _redis
    if _redis:
        await _redis.aclose()
        logger.info("Redis connection closed.")


def get_redis() -> aioredis.Redis:
    """Return the Redis client instance."""
    if _redis is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis


def _build_key(key: str) -> str:
    return get_settings().redis.prefix + key


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

async def cache_get(key: str) -> Optional[str]:
    return await get_redis().get(_build_key(key))


async def cache_set(key: str, value: str, ttl: Optional[int] = None) -> None:
    r = get_redis()
    if ttl:
        await r.setex(_build_key(key), ttl, value)
    else:
        await r.set(_build_key(key), value)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

async def cache_get_json(key: str) -> Optional[Any]:
    raw = await cache_get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def cache_set_json(key: str, obj: Any, ttl: Optional[int] = None) -> None:
    await cache_set(key, json.dumps(obj, ensure_ascii=False), ttl)


async def cache_delete(key: str) -> None:
    await get_redis().delete(_build_key(key))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check() -> dict:
    try:
        if _redis is None:
            return {"status": "error", "message": "not initialized"}
        await _redis.ping()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

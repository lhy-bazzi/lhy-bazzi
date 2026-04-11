"""System endpoints — health check, metrics."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check — verifies connectivity to all infrastructure components.

    Returns overall status:
    - healthy  : all components reachable
    - degraded : some components down
    - unhealthy: all components down
    """
    settings = get_settings()
    components: list[ComponentHealth] = []

    # Import here to avoid circular imports at module load time
    from app.core import database, es_client, minio_client, milvus_client, redis_client

    for name, checker in [
        ("postgresql",     database.health_check),
        ("redis",          redis_client.health_check),
        ("minio",          minio_client.health_check),
        ("milvus",         milvus_client.health_check),
        ("elasticsearch",  es_client.health_check),
    ]:
        t0 = time.perf_counter()
        try:
            result = await checker()
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}
        latency = (time.perf_counter() - t0) * 1000
        components.append(ComponentHealth(
            name=name,
            status=result.get("status", "error"),
            latency_ms=round(latency, 2),
            message=result.get("message") or result.get("cluster_status"),
        ))

    ok_count = sum(1 for c in components if c.status == "ok")
    if ok_count == len(components):
        overall = "healthy"
    elif ok_count > 0:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        components=components,
    )

"""MinIO object-storage client (sync SDK wrapped with asyncio.to_thread)."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

from loguru import logger
from minio import Minio
from minio.error import S3Error

from app.config import get_settings

_client: Minio | None = None
_default_bucket: str = ""


def init_minio() -> None:
    """Create MinIO client and ensure default bucket exists. Call on startup."""
    global _client, _default_bucket
    cfg = get_settings().minio
    logger.info("Connecting to MinIO: {}", cfg.endpoint)

    _client = Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )
    _default_bucket = cfg.bucket

    # Ensure bucket exists
    if not _client.bucket_exists(_default_bucket):
        _client.make_bucket(_default_bucket)
        logger.info("MinIO bucket '{}' created.", _default_bucket)
    else:
        logger.info("MinIO bucket '{}' already exists.", _default_bucket)
    logger.info("MinIO connected successfully.")


def get_minio() -> Minio:
    if _client is None:
        raise RuntimeError("MinIO not initialized. Call init_minio() first.")
    return _client


# ---------------------------------------------------------------------------
# Async-wrapped operations
# ---------------------------------------------------------------------------

async def download_file(object_name: str, local_path: str, bucket: Optional[str] = None) -> None:
    """Download an object from MinIO to a local path."""
    bucket = bucket or _default_bucket
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        get_minio().fget_object, bucket, object_name, local_path
    )
    logger.debug("Downloaded {}/{} → {}", bucket, object_name, local_path)


async def upload_file(
    object_name: str,
    file_path: str,
    content_type: str = "application/octet-stream",
    bucket: Optional[str] = None,
) -> str:
    """Upload a local file to MinIO. Returns the full object path."""
    bucket = bucket or _default_bucket
    await asyncio.to_thread(
        get_minio().fput_object, bucket, object_name, file_path, content_type
    )
    logger.debug("Uploaded {} → {}/{}", file_path, bucket, object_name)
    return f"{bucket}/{object_name}"


async def get_presigned_url(
    object_name: str,
    expires_seconds: int = 3600,
    bucket: Optional[str] = None,
) -> str:
    """Generate a presigned GET URL."""
    bucket = bucket or _default_bucket
    url = await asyncio.to_thread(
        get_minio().presigned_get_object,
        bucket,
        object_name,
        expires=timedelta(seconds=expires_seconds),
    )
    return url


async def file_exists(object_name: str, bucket: Optional[str] = None) -> bool:
    """Return True if the object exists in the bucket."""
    bucket = bucket or _default_bucket
    try:
        await asyncio.to_thread(get_minio().stat_object, bucket, object_name)
        return True
    except S3Error:
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check() -> dict:
    try:
        if _client is None:
            return {"status": "error", "message": "not initialized"}
        exists = await asyncio.to_thread(
            _client.bucket_exists, _default_bucket
        )
        return {"status": "ok"} if exists else {"status": "error", "message": "bucket not found"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

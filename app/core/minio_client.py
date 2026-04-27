"""MinIO object-storage client (sync SDK wrapped with asyncio.to_thread)."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

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
# Object reference normalization
# ---------------------------------------------------------------------------

def _configured_default_bucket() -> str:
    return _default_bucket or get_settings().minio.bucket


def _normalize_object_name(raw: str) -> str:
    normalized = unquote(raw).strip().lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise ValueError("MinIO object name is empty after normalization.")
    return normalized


def resolve_object_location(object_ref: str, bucket: str | None = None) -> tuple[str, str]:
    """Resolve object reference into ``(bucket, object_name)``.

    Supported forms:
    - DB canonical path: ``/bucket/key``
    - Full URL: ``http(s)://host:port/bucket/key`` (path-style URL)
    - ``bucket/key`` (when bucket equals configured default bucket)
    - Raw ``key`` (uses configured default bucket)

    If ``bucket`` is provided explicitly, it is always respected.
    """
    raw_ref = (object_ref or "").strip()
    if not raw_ref:
        raise ValueError("MinIO object reference cannot be empty.")

    default_bucket = bucket or _configured_default_bucket()

    # Canonical DB path format: /bucket/key
    if not bucket and raw_ref.startswith("/"):
        path = _normalize_object_name(raw_ref)
        if "/" in path:
            path_bucket, path_key = path.split("/", 1)
            if path_key:
                return path_bucket, path_key

    # Explicit bucket wins; normalize only the key portion.
    if bucket:
        parsed = urlparse(raw_ref)
        key_candidate = parsed.path if parsed.scheme and parsed.netloc else raw_ref
        key = _normalize_object_name(key_candidate)
        prefix = f"{bucket}/"
        if key.startswith(prefix):
            key = key[len(prefix) :]
        return bucket, key

    parsed = urlparse(raw_ref)
    if parsed.scheme and parsed.netloc:
        path = _normalize_object_name(parsed.path)
        if "/" in path:
            path_bucket, path_key = path.split("/", 1)
            if path_key:
                return path_bucket, path_key
        return default_bucket, path

    normalized = _normalize_object_name(raw_ref)
    default_prefix = f"{default_bucket}/"
    if normalized.startswith(default_prefix):
        return default_bucket, normalized[len(default_prefix) :]
    return default_bucket, normalized


def to_db_file_path(object_ref: str, bucket: str | None = None) -> str:
    """Convert any supported object reference into canonical DB format.

    Returns:
        ``/bucket/key``
    """
    bucket_name, key = resolve_object_location(object_ref, bucket=bucket)
    return f"/{bucket_name}/{key}"


def build_object_url(object_ref: str, bucket: str | None = None) -> str:
    """Build a full object URL from config endpoint + canonical path."""
    cfg = get_settings().minio
    endpoint = cfg.endpoint.strip().rstrip("/")
    if endpoint.startswith(("http://", "https://")):
        base = endpoint
    else:
        scheme = "https" if cfg.secure else "http"
        base = f"{scheme}://{endpoint}"
    return base + quote(to_db_file_path(object_ref, bucket=bucket), safe="/")


# ---------------------------------------------------------------------------
# Async-wrapped operations
# ---------------------------------------------------------------------------

async def download_file(object_name: str, local_path: str, bucket: str | None = None) -> None:
    """Download an object from MinIO to a local path."""
    bucket_name, key = resolve_object_location(object_name, bucket=bucket)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(get_minio().fget_object, bucket_name, key, local_path)
    logger.debug("Downloaded {}/{} -> {}", bucket_name, key, local_path)


async def upload_file(
    object_name: str,
    file_path: str,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
) -> str:
    """Upload a local file to MinIO. Returns the full object path."""
    bucket = bucket or _default_bucket
    await asyncio.to_thread(get_minio().fput_object, bucket, object_name, file_path, content_type)
    logger.debug("Uploaded {} -> {}/{}", file_path, bucket, object_name)
    return f"{bucket}/{object_name}"


async def get_presigned_url(
    object_name: str,
    expires_seconds: int = 3600,
    bucket: str | None = None,
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


async def file_exists(object_name: str, bucket: str | None = None) -> bool:
    """Return True if the object exists in the bucket."""
    bucket_name, key = resolve_object_location(object_name, bucket=bucket)
    try:
        await asyncio.to_thread(get_minio().stat_object, bucket_name, key)
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
        exists = await asyncio.to_thread(_client.bucket_exists, _default_bucket)
        return {"status": "ok"} if exists else {"status": "error", "message": "bucket not found"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

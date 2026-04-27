from __future__ import annotations

import pytest

from app.config import get_settings
from app.core.minio_client import build_object_url, resolve_object_location, to_db_file_path


def test_resolve_raw_key_uses_default_bucket() -> None:
    cfg_bucket = get_settings().minio.bucket
    bucket, key = resolve_object_location("folder/doc-a.docx")
    assert bucket == cfg_bucket
    assert key == "folder/doc-a.docx"


def test_resolve_default_bucket_prefixed_path() -> None:
    cfg_bucket = get_settings().minio.bucket
    bucket, key = resolve_object_location(f"{cfg_bucket}/folder/doc-b.docx")
    assert bucket == cfg_bucket
    assert key == "folder/doc-b.docx"


def test_resolve_canonical_db_path() -> None:
    bucket, key = resolve_object_location("/uni-ai-docs/folder/doc-c.docx")
    assert bucket == "uni-ai-docs"
    assert key == "folder/doc-c.docx"


def test_resolve_url_and_decode_key() -> None:
    bucket, key = resolve_object_location(
        "http://localhost:9000/uni-ai-docs/%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3.docx"
    )
    assert bucket == "uni-ai-docs"
    assert key == "\u4e2d\u6587\u6587\u6863.docx"


def test_explicit_bucket_has_priority() -> None:
    bucket, key = resolve_object_location(
        "http://localhost:9000/uni-ai-docs/%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3.docx",
        bucket="uni-ai-docs",
    )
    assert bucket == "uni-ai-docs"
    assert key == "\u4e2d\u6587\u6587\u6863.docx"


def test_resolve_empty_reference_raises() -> None:
    with pytest.raises(ValueError):
        resolve_object_location("")


def test_to_db_file_path_from_url() -> None:
    db_path = to_db_file_path(
        "http://localhost:9000/uni-ai-docs/%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3.docx"
    )
    assert db_path == "/uni-ai-docs/\u4e2d\u6587\u6587\u6863.docx"


def test_build_object_url_from_canonical_db_path() -> None:
    cfg = get_settings().minio
    if cfg.endpoint.startswith(("http://", "https://")):
        expected_prefix = cfg.endpoint
    else:
        scheme = "https" if cfg.secure else "http"
        expected_prefix = f"{scheme}://{cfg.endpoint}"
    full_url = build_object_url("/uni-ai-docs/folder/doc-c.docx")
    assert full_url == f"{expected_prefix}/uni-ai-docs/folder/doc-c.docx"

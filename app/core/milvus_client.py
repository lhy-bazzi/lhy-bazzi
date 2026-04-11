"""Milvus vector database client — collection management and CRUD."""

from __future__ import annotations

from typing import Optional

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.config import get_settings

_client: MilvusClient | None = None
_collection: Collection | None = None

COLLECTION_NAME = "knowledge_chunks"

# ---------------------------------------------------------------------------
# Schema definition  (mirrors tech-design §5.1)
# ---------------------------------------------------------------------------

def _build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id",            dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id",        dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="kb_id",         dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_index",   dtype=DataType.INT32),
        FieldSchema(name="content",       dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="heading_chain", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_type",    dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="dense_vector",  dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    return CollectionSchema(fields=fields, description="UniAI knowledge chunk vectors")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def init_milvus() -> None:
    """Connect to Milvus and ensure collection + indexes exist."""
    global _client, _collection
    cfg = get_settings().milvus
    logger.info("Connecting to Milvus: {}:{}", cfg.host, cfg.port)

    connections.connect(alias="default", host=cfg.host, port=cfg.port)
    _client = MilvusClient(uri=f"http://{cfg.host}:{cfg.port}")

    await ensure_collection()
    logger.info("Milvus connected successfully.")


async def close_milvus() -> None:
    global _client, _collection
    connections.disconnect("default")
    _client = None
    _collection = None
    logger.info("Milvus connection closed.")


def get_milvus() -> MilvusClient:
    if _client is None:
        raise RuntimeError("Milvus not initialized. Call init_milvus() first.")
    return _client


def get_collection() -> Collection:
    if _collection is None:
        raise RuntimeError("Milvus collection not initialized.")
    return _collection


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

async def ensure_collection() -> None:
    """Create collection and indexes if they don't exist yet."""
    global _collection

    col_name = get_settings().milvus.collection or COLLECTION_NAME

    if not utility.has_collection(col_name):
        logger.info("Creating Milvus collection '{}'...", col_name)
        schema = _build_schema()
        _collection = Collection(name=col_name, schema=schema)
        _create_indexes(_collection)
        logger.info("Milvus collection '{}' created.", col_name)
    else:
        _collection = Collection(name=col_name)
        logger.info("Milvus collection '{}' loaded.", col_name)

    _collection.load()


def _create_indexes(col: Collection) -> None:
    # Dense vector — HNSW
    col.create_index(
        field_name="dense_vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        },
    )
    # Sparse vector
    col.create_index(
        field_name="sparse_vector",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        },
    )
    # Scalar indexes for permission filtering
    col.create_index(field_name="kb_id",  index_params={"index_type": "INVERTED"})
    col.create_index(field_name="doc_id", index_params={"index_type": "INVERTED"})


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def insert_chunks(chunks: list[dict]) -> int:
    """Batch insert chunk dicts into Milvus. Returns inserted count."""
    if not chunks:
        return 0
    col = get_collection()
    result = col.insert(chunks)
    col.flush()
    return result.insert_count


async def search_dense(
    vector: list[float],
    top_k: int = 20,
    filter_expr: Optional[str] = None,
    output_fields: Optional[list[str]] = None,
) -> list[dict]:
    col = get_collection()
    output_fields = output_fields or ["id", "doc_id", "kb_id", "content", "heading_chain", "chunk_type"]
    results = col.search(
        data=[vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        expr=filter_expr,
        output_fields=output_fields,
    )
    return _format_search_results(results)


async def search_sparse(
    sparse_vector: dict,
    top_k: int = 20,
    filter_expr: Optional[str] = None,
    output_fields: Optional[list[str]] = None,
) -> list[dict]:
    col = get_collection()
    output_fields = output_fields or ["id", "doc_id", "kb_id", "content", "heading_chain", "chunk_type"]
    results = col.search(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=top_k,
        expr=filter_expr,
        output_fields=output_fields,
    )
    return _format_search_results(results)


def _format_search_results(results) -> list[dict]:
    hits = []
    for batch in results:
        for hit in batch:
            row = {k: hit.entity.get(k) for k in hit.entity.fields}
            row["score"] = hit.score
            hits.append(row)
    return hits


async def delete_by_doc_id(doc_id: str) -> int:
    col = get_collection()
    result = col.delete(expr=f'doc_id == "{doc_id}"')
    return result.delete_count


async def delete_by_kb_id(kb_id: str) -> int:
    col = get_collection()
    result = col.delete(expr=f'kb_id == "{kb_id}"')
    return result.delete_count


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check() -> dict:
    try:
        if _client is None:
            return {"status": "error", "message": "not initialized"}
        col_name = get_settings().milvus.collection or COLLECTION_NAME
        exists = utility.has_collection(col_name)
        return {"status": "ok"} if exists else {"status": "error", "message": "collection missing"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

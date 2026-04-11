"""Elasticsearch async client — index management and BM25 retrieval."""

from __future__ import annotations

from typing import Optional

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk
from loguru import logger

from app.config import get_settings

_es: AsyncElasticsearch | None = None

# ---------------------------------------------------------------------------
# Index mapping  (mirrors tech-design §5.2)
# ---------------------------------------------------------------------------

def _chunk_index_mapping() -> dict:
    cfg = get_settings().elasticsearch
    analyzer = cfg.analyzer  # e.g. "ik_max_word"
    return {
        "mappings": {
            "properties": {
                "id":            {"type": "keyword"},
                "doc_id":        {"type": "keyword"},
                "kb_id":         {"type": "keyword"},
                "content":       {"type": "text", "analyzer": analyzer, "search_analyzer": "ik_smart"},
                "heading_chain": {"type": "text", "analyzer": analyzer},
                "chunk_type":    {"type": "keyword"},
                "chunk_index":   {"type": "integer"},
                "created_at":    {"type": "date"},
            }
        }
    }


def _index_name() -> str:
    return get_settings().elasticsearch.index_prefix + "knowledge_chunks"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def init_es() -> None:
    global _es
    cfg = get_settings().elasticsearch
    logger.info("Connecting to Elasticsearch: {}", cfg.hosts)
    _es = AsyncElasticsearch(hosts=cfg.hosts)
    # Verify connectivity
    info = await _es.info()
    logger.info("Elasticsearch connected: version {}", info["version"]["number"])
    await ensure_index()


async def close_es() -> None:
    global _es
    if _es:
        await _es.close()
        logger.info("Elasticsearch connection closed.")


def get_es() -> AsyncElasticsearch:
    if _es is None:
        raise RuntimeError("Elasticsearch not initialized. Call init_es() first.")
    return _es


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

async def ensure_index() -> None:
    """Create the chunk index if it doesn't exist."""
    idx = _index_name()
    es = get_es()
    if not await es.indices.exists(index=idx):
        # Attempt mapping with ik analyzer; fall back gracefully if plugin missing
        try:
            await es.indices.create(index=idx, body=_chunk_index_mapping())
            logger.info("Elasticsearch index '{}' created.", idx)
        except Exception as exc:
            # If ik analyzer is unavailable, create with standard analyzer
            logger.warning("IK analyzer unavailable, falling back to standard: {}", exc)
            fallback = _chunk_index_mapping()
            for field in ("content", "heading_chain"):
                fallback["mappings"]["properties"][field] = {"type": "text"}
            await es.indices.create(index=idx, body=fallback)
            logger.info("Elasticsearch index '{}' created (standard analyzer).", idx)
    else:
        logger.info("Elasticsearch index '{}' already exists.", idx)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def index_chunks(chunks: list[dict]) -> int:
    """Bulk-index chunks. Each dict must contain at least: id, doc_id, kb_id, content."""
    if not chunks:
        return 0
    idx = _index_name()

    def _actions():
        for c in chunks:
            yield {
                "_index": idx,
                "_id": c["id"],
                "_source": {
                    "id":            c["id"],
                    "doc_id":        c.get("doc_id", ""),
                    "kb_id":         c.get("kb_id", ""),
                    "content":       c.get("content", ""),
                    "heading_chain": c.get("heading_chain", ""),
                    "chunk_type":    c.get("chunk_type", "text"),
                    "chunk_index":   c.get("chunk_index", 0),
                    "created_at":    c.get("created_at", "now"),
                },
            }

    success, _ = await async_bulk(get_es(), _actions())
    return success


async def search_bm25(
    query: str,
    top_k: int = 20,
    kb_ids: Optional[list[str]] = None,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """BM25 full-text search with optional permission filters."""
    must_clause: list[dict] = [
        {
            "multi_match": {
                "query": query,
                "fields": ["content^2", "heading_chain"],
            }
        }
    ]
    filters: list[dict] = []
    if kb_ids:
        filters.append({"terms": {"kb_id": kb_ids}})
    if doc_ids:
        filters.append({"terms": {"doc_id": doc_ids}})

    body: dict = {
        "query": {
            "bool": {
                "must": must_clause,
                **({"filter": filters} if filters else {}),
            }
        },
        "size": top_k,
    }

    resp = await get_es().search(index=_index_name(), body=body)
    return [
        {**hit["_source"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]


async def delete_by_doc_id(doc_id: str) -> int:
    resp = await get_es().delete_by_query(
        index=_index_name(),
        body={"query": {"term": {"doc_id": doc_id}}},
    )
    return resp.get("deleted", 0)


async def delete_by_kb_id(kb_id: str) -> int:
    resp = await get_es().delete_by_query(
        index=_index_name(),
        body={"query": {"term": {"kb_id": kb_id}}},
    )
    return resp.get("deleted", 0)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check() -> dict:
    try:
        if _es is None:
            return {"status": "error", "message": "not initialized"}
        health = await _es.cluster.health()
        return {"status": "ok", "cluster_status": health.get("status")}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

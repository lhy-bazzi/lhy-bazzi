"""Index service — orchestrates parallel Milvus + ES dual-write."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from app.models.chunk import ChunkNode
from app.services.indexing.es_indexer import ESIndexer
from app.services.indexing.milvus_indexer import MilvusIndexer


@dataclass
class IndexResult:
    milvus_count: int = 0
    es_count: int = 0
    milvus_error: str | None = None
    es_error: str | None = None

    @property
    def success(self) -> bool:
        return self.milvus_error is None and self.es_error is None


class IndexService:
    """Dual-write orchestrator: Milvus (vector) + ES (BM25)."""

    def __init__(self, milvus_indexer: MilvusIndexer, es_indexer: ESIndexer):
        self.milvus_indexer = milvus_indexer
        self.es_indexer = es_indexer

    async def index_chunks(self, chunks: list[ChunkNode]) -> IndexResult:
        result = IndexResult()

        async def _milvus():
            try:
                result.milvus_count = await self.milvus_indexer.index_chunks(chunks)
            except Exception as exc:
                logger.error("Milvus index failed: {}", exc)
                result.milvus_error = str(exc)

        async def _es():
            try:
                result.es_count = await self.es_indexer.index_chunks(chunks)
            except Exception as exc:
                logger.error("ES index failed: {}", exc)
                result.es_error = str(exc)

        await asyncio.gather(_milvus(), _es())
        logger.info(
            "Index done: milvus={} es={} errors=[milvus={}, es={}]",
            result.milvus_count, result.es_count,
            result.milvus_error, result.es_error,
        )
        return result

    async def delete_document(self, doc_id: str) -> None:
        results = await asyncio.gather(
            self.milvus_indexer.delete_by_doc_id(doc_id),
            self.es_indexer.delete_by_doc_id(doc_id),
            return_exceptions=True,
        )
        logger.info("Deleted doc_id={}: milvus={} es={}", doc_id, results[0], results[1])

    async def delete_knowledge_base(self, kb_id: str) -> None:
        results = await asyncio.gather(
            self.milvus_indexer.delete_by_kb_id(kb_id),
            self.es_indexer.delete_by_kb_id(kb_id),
            return_exceptions=True,
        )
        logger.info("Deleted kb_id={}: milvus={} es={}", kb_id, results[0], results[1])

    async def rebuild_knowledge_base(self, kb_id: str, chunks: list[ChunkNode]) -> IndexResult:
        await self.delete_knowledge_base(kb_id)
        return await self.index_chunks(chunks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_index_service: IndexService | None = None


def get_index_service() -> IndexService:
    if _index_service is None:
        raise RuntimeError("IndexService not initialized.")
    return _index_service


def init_index_service(milvus_indexer: MilvusIndexer, es_indexer: ESIndexer) -> IndexService:
    global _index_service
    _index_service = IndexService(milvus_indexer, es_indexer)
    return _index_service

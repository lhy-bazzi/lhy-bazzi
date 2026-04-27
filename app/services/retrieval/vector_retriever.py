"""Vector retrievers — Dense (Milvus COSINE) and Sparse (Milvus IP)."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from app.services.retrieval.models import RetrievedChunk


def _hits_to_chunks(hits: list[dict], *, source_leg: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=h["id"],
            doc_id=h.get("doc_id", ""),
            kb_id=h.get("kb_id", ""),
            content=h.get("content", ""),
            heading_chain=h.get("heading_chain", ""),
            chunk_type=h.get("chunk_type", "text"),
            score=float(h.get("score", 0.0)),
            source_leg=source_leg,
        )
        for h in hits
    ]


class VectorRetriever:
    """Milvus Dense vector retrieval (COSINE)."""

    def __init__(self, milvus_client, embedding_service):
        self.milvus = milvus_client
        self.embedder = embedding_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        from app.core import milvus_client as mc

        emb = await self.embedder.embed_query(query)
        hits = await mc.search_dense(
            vector=emb.dense_vector,
            top_k=top_k,
            filter_expr=filter_expr or None,
        )
        logger.debug("VectorRetriever: {} hits", len(hits))
        return _hits_to_chunks(hits, source_leg="dense")


class SparseRetriever:
    """Milvus Sparse vector retrieval (BGE-M3 sparse / IP)."""

    def __init__(self, milvus_client, embedding_service):
        self.milvus = milvus_client
        self.embedder = embedding_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        from app.core import milvus_client as mc

        emb = await self.embedder.embed_query(query)
        # sparse_vector may be empty dict when using DashScope dense-only API
        if not emb.sparse_vector:
            logger.debug("SparseRetriever: no sparse vector available, skipping")
            return []
        hits = await mc.search_sparse(
            sparse_vector=emb.sparse_vector,
            top_k=top_k,
            filter_expr=filter_expr or None,
        )
        logger.debug("SparseRetriever: {} hits", len(hits))
        return _hits_to_chunks(hits, source_leg="sparse")

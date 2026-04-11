"""Elasticsearch BM25 full-text retriever."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from app.services.retrieval.models import RetrievedChunk


class FulltextRetriever:
    """Elasticsearch BM25 full-text retrieval."""

    def __init__(self, es_client):
        self.es = es_client

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        from app.core import es_client as ec

        kb_ids = (filter_dict or {}).get("kb_ids")
        doc_ids = (filter_dict or {}).get("doc_ids")
        hits = await ec.search_bm25(query=query, top_k=top_k, kb_ids=kb_ids, doc_ids=doc_ids)
        logger.debug("FulltextRetriever: {} hits", len(hits))
        return [
            RetrievedChunk(
                chunk_id=h["id"],
                doc_id=h.get("doc_id", ""),
                kb_id=h.get("kb_id", ""),
                content=h.get("content", ""),
                heading_chain=h.get("heading_chain", ""),
                chunk_type=h.get("chunk_type", "text"),
                score=float(h.get("score", 0.0)),
            )
            for h in hits
        ]

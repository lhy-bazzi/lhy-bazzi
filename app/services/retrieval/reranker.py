"""Reranker — DashScope online rerank API (no local model)."""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from app.services.retrieval.models import RetrievedChunk

_RERANK_MAX_INPUT = 30


class RerankerService:
    """Cross-encoder reranking via DashScope text-rerank API."""

    def __init__(self, model_manager):
        self.cfg = model_manager  # holds api_key, base_url

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        candidates = chunks[:_RERANK_MAX_INPUT]
        try:
            scores = await asyncio.to_thread(self._call_rerank_api, query, candidates)
        except Exception as exc:
            logger.warning("Rerank API failed, returning original order: {}", exc)
            return candidates[:top_k]

        for chunk, score in zip(candidates, scores):
            chunk.score = score

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _call_rerank_api(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Synchronous DashScope rerank call (run in thread)."""
        documents = [c.content for c in chunks]
        payload = {
            "model": "gte-rerank",  # DashScope rerank model
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {"top_n": len(documents), "return_documents": False},
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(
            f"{self.cfg.base_url.rstrip('/')}/rerank",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # DashScope returns results sorted by relevance_score; re-map to original order
        results = data["output"]["results"]
        index_to_score = {r["index"]: r["relevance_score"] for r in results}
        return [index_to_score.get(i, 0.0) for i in range(len(chunks))]

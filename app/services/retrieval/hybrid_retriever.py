"""Hybrid retrieval engine — unified entry point."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from loguru import logger

from app.models.enums import RetrievalMode
from app.services.retrieval.models import RetrievalConfig, RetrievalResult, RetrievedChunk


class HybridRetriever:
    """Orchestrates dense + sparse + BM25 → RRF → rerank → permission filter."""

    def __init__(
        self,
        vector_retriever,
        sparse_retriever,
        fulltext_retriever,
        fusion,
        reranker,
        permission_filter,
        settings,
    ):
        self.vector = vector_retriever
        self.sparse = sparse_retriever
        self.fulltext = fulltext_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.perm = permission_filter
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        user_id: str,
        kb_ids: list[str],
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        cfg = config or RetrievalConfig()
        t0 = time.monotonic()
        debug: dict = {}

        # 1. Permission context
        user_ctx = await self.perm.get_user_context(user_id)
        milvus_filter = self.perm.build_milvus_filter(user_ctx, kb_ids)
        es_filter = self.perm.build_es_filter(user_ctx, kb_ids)
        es_filter_dict = {
            "kb_ids": list(set(user_ctx.kb_ids) & set(kb_ids)) if not user_ctx.is_admin else kb_ids,
            "doc_ids": user_ctx.doc_ids if not user_ctx.is_admin else None,
        }

        # 2. Parallel retrieval legs
        mode = cfg.retrieval_mode
        tasks: list = []
        leg_names: list[str] = []

        if mode in (RetrievalMode.HYBRID, RetrievalMode.VECTOR_ONLY):
            tasks.append(self._safe_retrieve(self.vector, query, 20, milvus_filter, "dense"))
            leg_names.append("dense")
            tasks.append(self._safe_retrieve(self.sparse, query, 20, milvus_filter, "sparse"))
            leg_names.append("sparse")

        if mode in (RetrievalMode.HYBRID, RetrievalMode.FULLTEXT_ONLY):
            tasks.append(self._safe_retrieve_ft(query, 20, es_filter_dict))
            leg_names.append("bm25")

        t_legs_start = time.monotonic()
        leg_results = await asyncio.gather(*tasks)
        debug["leg_latency_ms"] = int((time.monotonic() - t_legs_start) * 1000)
        for name, res in zip(leg_names, leg_results):
            debug[f"{name}_count"] = len(res)

        total_retrieved = sum(len(r) for r in leg_results)

        # 3. RRF fusion
        weights = [cfg.vector_weight, cfg.sparse_weight, cfg.bm25_weight]
        t_fuse = time.monotonic()
        fused = self.fusion.fuse(list(leg_results), weights=weights)
        debug["fusion_latency_ms"] = int((time.monotonic() - t_fuse) * 1000)
        debug["fused_count"] = len(fused)

        # 4. Rerank
        if cfg.rerank and fused:
            t_rerank = time.monotonic()
            reranked = await self.reranker.rerank(query, fused, top_k=cfg.top_k)
            debug["rerank_latency_ms"] = int((time.monotonic() - t_rerank) * 1000)
            debug["reranked_count"] = len(reranked)
        else:
            reranked = fused[: cfg.top_k]

        # 5. Application-layer permission post-filter
        filtered = self.perm.post_filter(reranked, user_ctx)

        # 6. Parent context expansion
        expanded = await self._expand_parent_context(filtered)

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "HybridRetriever: query='{}' mode={} total={} fused={} final={} latency={}ms",
            query[:60], mode, total_retrieved, len(fused), len(expanded), latency_ms,
        )

        return RetrievalResult(
            chunks=expanded,
            total_retrieved=total_retrieved,
            retrieval_mode=mode if isinstance(mode, str) else mode.value,
            latency_ms=latency_ms,
            debug=debug,
        )

    async def _safe_retrieve(
        self, retriever, query: str, top_k: int, filter_expr: str, name: str
    ) -> list[RetrievedChunk]:
        try:
            t = time.monotonic()
            result = await retriever.retrieve(query, top_k=top_k, filter_expr=filter_expr)
            logger.debug("{} retrieval: {} hits in {:.0f}ms", name, len(result), (time.monotonic() - t) * 1000)
            return result
        except Exception as exc:
            logger.warning("{} retrieval failed, degrading: {}", name, exc)
            return []

    async def _safe_retrieve_ft(
        self, query: str, top_k: int, filter_dict: dict
    ) -> list[RetrievedChunk]:
        try:
            t = time.monotonic()
            result = await self.fulltext.retrieve(query, top_k=top_k, filter_dict=filter_dict)
            logger.debug("bm25 retrieval: {} hits in {:.0f}ms", len(result), (time.monotonic() - t) * 1000)
            return result
        except Exception as exc:
            logger.warning("BM25 retrieval failed, degrading: {}", exc)
            return []

    async def _expand_parent_context(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Attach parent_content from Milvus metadata if available."""
        # Parent content is stored in the chunk's own record for now;
        # if a separate parent lookup is needed it can be added here.
        return chunks

"""Hybrid retrieval engine — unified entry point."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from loguru import logger

from app.models.enums import RetrievalMode
from app.services.retrieval.models import RetrievalConfig, RetrievalResult, RetrievedChunk

_LEG_PREVIEW_LIMIT = 3
_VECTOR_PREVIEW_LIMIT = 4
_SNIPPET_PREVIEW_LIMIT = 160


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

    @staticmethod
    def _preview_text(text: str | None, limit: int = _SNIPPET_PREVIEW_LIMIT) -> str:
        raw = text or ""
        # Keep user-facing snippet clean; front-end can still use snippet_highlight.
        plain = re.sub(r"</?em>", "", raw)
        compact = " ".join(plain.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def _preview_chunks(
        self,
        chunks: list[RetrievedChunk],
        *,
        source: str,
        limit: int = _LEG_PREVIEW_LIMIT,
    ) -> list[dict]:
        previews: list[dict] = []
        for c in chunks[:limit]:
            snippet_raw = c.highlight or c.content
            snippet_highlight = " ".join((c.highlight or "").split()) if c.highlight else None
            previews.append(
                {
                    "source": source,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "heading_chain": c.heading_chain,
                    "score": round(float(c.score), 4),
                    "snippet": self._preview_text(snippet_raw),
                    "snippet_highlight": snippet_highlight,
                }
            )
        return previews

    @staticmethod
    def _merge_previews(preview_lists: list[list[dict]], limit: int) -> list[dict]:
        merged: list[dict] = []
        seen: set[str] = set()
        for previews in preview_lists:
            for item in previews:
                cid = item.get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                merged.append(item)
                if len(merged) >= limit:
                    return merged
        return merged

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
        mode = cfg.retrieval_mode

        # 1. Permission context
        user_ctx = await self.perm.get_user_context(user_id)
        milvus_filter = self.perm.build_milvus_filter(user_ctx, kb_ids)
        es_filter_dict = {
            "kb_ids": list(set(user_ctx.kb_ids) & set(kb_ids)) if not user_ctx.is_admin else kb_ids,
            "doc_ids": user_ctx.doc_ids if not user_ctx.is_admin else None,
        }
        debug["query_preview"] = query[:200]
        debug["retrieval_mode"] = mode if isinstance(mode, str) else mode.value

        # Query terms used for ES side explainability.
        try:
            from app.core import es_client as ec

            debug["es_keywords"] = await ec.extract_query_terms(query, limit=10)
        except Exception as exc:
            logger.debug("Failed to extract ES keywords for retrieval explainability: {}", exc)
            debug["es_keywords"] = []

        # 2. Parallel retrieval legs
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
        leg_infos = await asyncio.gather(*tasks) if tasks else []
        debug["leg_latency_ms"] = int((time.monotonic() - t_legs_start) * 1000)
        leg_results: list[list[RetrievedChunk]] = []
        leg_previews: dict[str, list[dict]] = {}
        for name, info in zip(leg_names, leg_infos):
            hits = info.get("hits", [])
            latency_ms = info.get("latency_ms", 0)
            error = info.get("error")
            leg_results.append(hits)
            debug[f"{name}_count"] = len(hits)
            debug[f"{name}_latency_ms"] = int(latency_ms)
            if error:
                debug[f"{name}_error"] = str(error)
            previews = self._preview_chunks(hits, source=name, limit=_LEG_PREVIEW_LIMIT)
            leg_previews[name] = previews
            debug[f"{name}_preview"] = previews

        total_retrieved = sum(len(r) for r in leg_results)
        debug["vector_preview"] = self._merge_previews(
            [leg_previews.get("dense", []), leg_previews.get("sparse", [])],
            limit=_VECTOR_PREVIEW_LIMIT,
        )
        debug["bm25_preview"] = leg_previews.get("bm25", [])

        # 3. RRF fusion
        weights = [cfg.vector_weight, cfg.sparse_weight, cfg.bm25_weight]
        t_fuse = time.monotonic()
        fused = self.fusion.fuse(list(leg_results), weights=weights)
        debug["fusion_latency_ms"] = int((time.monotonic() - t_fuse) * 1000)
        debug["fused_count"] = len(fused)
        debug["fused_preview"] = self._preview_chunks(fused, source="fusion", limit=_LEG_PREVIEW_LIMIT)

        # 4. Rerank
        if cfg.rerank and fused:
            t_rerank = time.monotonic()
            reranked = await self.reranker.rerank(query, fused, top_k=cfg.top_k)
            debug["rerank_latency_ms"] = int((time.monotonic() - t_rerank) * 1000)
            debug["reranked_count"] = len(reranked)
        else:
            reranked = fused[: cfg.top_k]
            debug["rerank_latency_ms"] = 0
            debug["reranked_count"] = len(reranked)
        debug["reranked_preview"] = self._preview_chunks(reranked, source="rerank", limit=_LEG_PREVIEW_LIMIT)

        # 5. Application-layer permission post-filter
        filtered = self.perm.post_filter(reranked, user_ctx)
        debug["after_permission_count"] = len(filtered)

        # 6. Parent context expansion
        expanded = await self._expand_parent_context(filtered)
        debug["expanded_count"] = len(expanded)
        final_preview = self._preview_chunks(expanded, source="final", limit=_LEG_PREVIEW_LIMIT)
        debug["final_preview"] = final_preview
        debug["selected_evidence_preview"] = final_preview
        debug["allowed_kb_count"] = len(user_ctx.kb_ids) if not user_ctx.is_admin else -1
        debug["allowed_doc_count"] = len(user_ctx.doc_ids) if not user_ctx.is_admin else -1
        debug["is_admin"] = user_ctx.is_admin
        debug["pipeline"] = {
            "retrieval_mode": mode if isinstance(mode, str) else mode.value,
            "weights": {
                "dense": cfg.vector_weight,
                "sparse": cfg.sparse_weight,
                "bm25": cfg.bm25_weight,
            },
            "top_k": cfg.top_k,
            "rerank": cfg.rerank,
        }

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
    ) -> dict[str, object]:
        t = time.monotonic()
        try:
            result = await retriever.retrieve(query, top_k=top_k, filter_expr=filter_expr)
            latency_ms = int((time.monotonic() - t) * 1000)
            logger.debug("{} retrieval: {} hits in {}ms", name, len(result), latency_ms)
            return {"hits": result, "latency_ms": latency_ms, "error": None}
        except Exception as exc:
            latency_ms = int((time.monotonic() - t) * 1000)
            logger.warning("{} retrieval failed, degrading: {}", name, exc)
            return {"hits": [], "latency_ms": latency_ms, "error": str(exc)}

    async def _safe_retrieve_ft(
        self, query: str, top_k: int, filter_dict: dict
    ) -> dict[str, object]:
        t = time.monotonic()
        try:
            result = await self.fulltext.retrieve(query, top_k=top_k, filter_dict=filter_dict)
            latency_ms = int((time.monotonic() - t) * 1000)
            logger.debug("bm25 retrieval: {} hits in {}ms", len(result), latency_ms)
            return {"hits": result, "latency_ms": latency_ms, "error": None}
        except Exception as exc:
            latency_ms = int((time.monotonic() - t) * 1000)
            logger.warning("BM25 retrieval failed, degrading: {}", exc)
            return {"hits": [], "latency_ms": latency_ms, "error": str(exc)}

    async def _expand_parent_context(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Attach parent_content from Milvus metadata if available."""
        # Parent content is stored in the chunk's own record for now;
        # if a separate parent lookup is needed it can be added here.
        return chunks

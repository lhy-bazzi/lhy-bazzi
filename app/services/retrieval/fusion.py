"""Reciprocal Rank Fusion — multi-leg result merging."""

from __future__ import annotations

from app.services.retrieval.models import RetrievedChunk

_DEFAULT_WEIGHTS = [0.4, 0.3, 0.3]  # dense, sparse, bm25


class RRFFusion:
    """Reciprocal Rank Fusion: score(d) = Σ weight_i / (k + rank_i(d))."""

    def fuse(
        self,
        result_lists: list[list[RetrievedChunk]],
        weights: list[float] | None = None,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        if not result_lists:
            return []

        w = weights if weights and len(weights) == len(result_lists) else _DEFAULT_WEIGHTS
        # Pad weights if fewer provided than legs
        while len(w) < len(result_lists):
            w = list(w) + [1.0 / len(result_lists)]

        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for leg_idx, results in enumerate(result_lists):
            weight = w[leg_idx]
            for rank, chunk in enumerate(results, start=1):
                cid = chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
                if cid not in chunks:
                    chunks[cid] = chunk

        merged = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        out = []
        for cid in merged:
            c = chunks[cid]
            c.score = scores[cid]
            out.append(c)
        return out

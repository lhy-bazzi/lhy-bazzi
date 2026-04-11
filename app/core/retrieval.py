"""Retrieval service singleton — initialized on startup."""

from __future__ import annotations

from app.services.retrieval.hybrid_retriever import HybridRetriever

_retriever: HybridRetriever | None = None


def init_retriever(retriever: HybridRetriever) -> None:
    global _retriever
    _retriever = retriever


def get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call init_retriever() first.")
    return _retriever

"""Shared data models for the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    heading_chain: str
    chunk_type: str
    score: float
    parent_content: Optional[str] = None
    doc_name: Optional[str] = None
    page: Optional[int] = None


@dataclass
class RetrievalConfig:
    retrieval_mode: str = "hybrid"   # hybrid | vector_only | fulltext_only
    top_k: int = 10
    rerank: bool = True
    vector_weight: float = 0.4
    sparse_weight: float = 0.3
    bm25_weight: float = 0.3


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    total_retrieved: int
    retrieval_mode: str
    latency_ms: int
    debug: dict = field(default_factory=dict)  # per-leg latencies, pre/post rerank counts

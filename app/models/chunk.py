"""Chunk data models — output of the chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkNode:
    """分块节点 — 向量化和检索的基本单元"""
    chunk_id: str
    doc_id: str
    kb_id: str
    chunk_index: int
    content: str
    chunk_type: str                      # text / table / code / image
    heading_chain: str                   # "H1 > H2 > H3"
    metadata: dict = field(default_factory=dict)
    parent_content: Optional[str] = None
    token_count: int = 0


@dataclass
class ChunkConfig:
    """分块配置"""
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 128
    max_chunk_size: int = 1024
    parent_chunk_size: int = 2048

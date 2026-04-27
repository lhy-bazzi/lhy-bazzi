"""
Unified Intermediate Representation (UIR) — document parsing output models.

All document formats are parsed into this common structure before
chunking, embedding, and indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.models.enums import ElementType


@dataclass
class BBox:
    """Bounding box coordinates within a page (PDF / image)."""
    x0: float
    y0: float
    x1: float
    y1: float
    page: Optional[int] = None


@dataclass
class DocumentElement:
    """
    Atomic element produced by document parsing.

    For tables, `content` holds Markdown-formatted table text.
    For images, `content` holds a description and `metadata["minio_path"]`
    contains the stored object path.
    """
    element_type: ElementType
    content: str
    metadata: dict = field(default_factory=dict)
    bbox: Optional[BBox] = None
    children: list[DocumentElement] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Top-level metadata extracted from the document."""
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """
    Complete parsed document — the output of the parsing pipeline
    and the input to the chunking pipeline.
    """
    doc_id: str
    filename: str
    file_type: str
    elements: list[DocumentElement]
    metadata: DocumentMetadata
    parse_engine: str
    quality_score: float = 0.0
    raw_text: str = ""

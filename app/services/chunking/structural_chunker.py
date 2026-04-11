"""Structural chunker — splits ParsedDocument by heading hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models.chunk import ChunkConfig
from app.models.document import DocumentElement, ParsedDocument
from app.models.enums import ElementType


@dataclass
class RawChunk:
    """Intermediate chunk before metadata enrichment."""
    content: str
    chunk_type: str                      # text / table / code / image
    heading_chain: list[str] = field(default_factory=list)
    element: Optional[DocumentElement] = None  # original element (for table/code)
    needs_semantic_split: bool = False


class StructuralChunker:
    """基于文档结构（标题层级）做初步分块。"""

    def chunk(self, document: ParsedDocument, config: ChunkConfig) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        heading_chain: list[str] = []
        current_texts: list[str] = []

        def flush_text():
            text = "\n\n".join(current_texts).strip()
            if text:
                chunks.append(RawChunk(
                    content=text,
                    chunk_type="text",
                    heading_chain=list(heading_chain),
                    needs_semantic_split=len(text) > config.max_chunk_size * 4,
                ))
            current_texts.clear()

        for elem in document.elements:
            etype = elem.element_type

            if etype == ElementType.HEADING:
                flush_text()
                level = elem.metadata.get("level", 1)
                # Truncate chain to current level then append
                heading_chain = heading_chain[: level - 1]
                heading_chain.append(elem.content.strip())

            elif etype in (ElementType.TABLE,):
                flush_text()
                chunks.append(RawChunk(
                    content=elem.content,
                    chunk_type="table",
                    heading_chain=list(heading_chain),
                    element=elem,
                ))

            elif etype == ElementType.CODE:
                flush_text()
                chunks.append(RawChunk(
                    content=elem.content,
                    chunk_type="code",
                    heading_chain=list(heading_chain),
                    element=elem,
                ))

            elif etype == ElementType.IMAGE:
                flush_text()
                chunks.append(RawChunk(
                    content=elem.content,
                    chunk_type="image",
                    heading_chain=list(heading_chain),
                    element=elem,
                ))

            elif etype == ElementType.PAGE_BREAK:
                pass  # ignore

            else:
                # TEXT, LIST, FORMULA → accumulate
                if elem.content.strip():
                    current_texts.append(elem.content.strip())

        flush_text()
        return chunks

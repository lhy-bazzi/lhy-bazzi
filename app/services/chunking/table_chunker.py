"""Table chunker — handles table elements with header preservation."""

from __future__ import annotations

import uuid

from app.models.chunk import ChunkConfig, ChunkNode
from app.models.document import DocumentElement


_ROWS_PER_SEGMENT = 50


class TableChunker:
    """表格专用分块策略。"""

    def chunk(
        self,
        table_element: DocumentElement,
        doc_id: str,
        kb_id: str,
        heading_chain: str,
        config: ChunkConfig,
    ) -> list[ChunkNode]:
        content = table_element.content
        lines = content.splitlines()

        # Identify header lines (first line + separator line)
        header_lines: list[str] = []
        data_lines: list[str] = []
        for i, line in enumerate(lines):
            if i < 2 and (i == 0 or re.match(r'^\s*\|?[\s\-|:]+\|?\s*$', line)):
                header_lines.append(line)
            else:
                data_lines.append(line)

        header_text = "\n".join(header_lines)
        table_name = table_element.metadata.get("table_name", "")
        col_names = _extract_col_names(header_lines)

        # Small table — single chunk
        if len(content) <= config.max_chunk_size * 4:
            summary = _make_summary(table_name, col_names, len(data_lines))
            return [ChunkNode(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=0,
                content=content,
                chunk_type="table",
                heading_chain=heading_chain,
                metadata={**table_element.metadata, "table_summary": summary},
            )]

        # Large table — split by rows, keep header in each segment
        chunks: list[ChunkNode] = []
        for start in range(0, len(data_lines), _ROWS_PER_SEGMENT):
            segment_rows = data_lines[start: start + _ROWS_PER_SEGMENT]
            segment_content = header_text + "\n" + "\n".join(segment_rows)
            summary = _make_summary(table_name, col_names, len(segment_rows))
            chunks.append(ChunkNode(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=len(chunks),
                content=segment_content,
                chunk_type="table",
                heading_chain=heading_chain,
                metadata={**table_element.metadata, "table_summary": summary},
            ))
        return chunks


import re  # noqa: E402 — placed after class to keep class readable


def _extract_col_names(header_lines: list[str]) -> list[str]:
    if not header_lines:
        return []
    first = header_lines[0]
    cols = [c.strip() for c in first.strip().strip("|").split("|")]
    return [c for c in cols if c]


def _make_summary(table_name: str, col_names: list[str], row_count: int) -> str:
    parts = []
    if table_name:
        parts.append(table_name)
    if col_names:
        parts.append("列: " + ", ".join(col_names))
    parts.append(f"{row_count} 行")
    return " | ".join(parts)

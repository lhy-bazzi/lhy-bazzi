"""Table chunker with token-aware limits and header preservation."""

from __future__ import annotations

import re
import uuid

from app.models.chunk import ChunkConfig, ChunkNode
from app.models.document import DocumentElement
from app.services.chunking.token_utils import count_tokens


class TableChunker:
    """Split table elements into token-bounded chunks."""

    def chunk(
        self,
        table_element: DocumentElement,
        doc_id: str,
        kb_id: str,
        heading_chain: str,
        config: ChunkConfig,
    ) -> list[ChunkNode]:
        content = table_element.content or ""
        lines = content.splitlines()

        header_lines, data_lines = self._split_header_and_data(lines)
        header_text = "\n".join(header_lines).strip()
        table_name = table_element.metadata.get("table_name", "")
        col_names = _extract_col_names(header_lines)
        max_tokens = max(1, config.max_chunk_size)

        # Small table: keep one chunk.
        if count_tokens(content) <= max_tokens:
            summary = _make_summary(table_name, col_names, len(data_lines))
            return [
                ChunkNode(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    kb_id=kb_id,
                    chunk_index=0,
                    content=content,
                    chunk_type="table",
                    heading_chain=heading_chain,
                    metadata={**table_element.metadata, "table_summary": summary},
                )
            ]

        # Large table: split by rows while keeping header in each chunk.
        chunks: list[ChunkNode] = []
        current_rows: list[str] = []

        for row in data_lines:
            candidate_rows = current_rows + [row]
            candidate_content = self._build_segment_text(header_text, candidate_rows)

            if current_rows and count_tokens(candidate_content) > max_tokens:
                overlap_rows = _tail_rows_for_overlap(current_rows, config.chunk_overlap)
                chunks.append(
                    self._build_chunk(
                        table_element=table_element,
                        doc_id=doc_id,
                        kb_id=kb_id,
                        heading_chain=heading_chain,
                        header_text=header_text,
                        rows=current_rows,
                        chunk_index=len(chunks),
                        table_name=table_name,
                        col_names=col_names,
                    )
                )
                current_rows = overlap_rows + [row]
                # Keep token hard limit if overlap + current row is still too large.
                if count_tokens(self._build_segment_text(header_text, current_rows)) > max_tokens:
                    current_rows = [row]
            else:
                current_rows = candidate_rows

            # Guard against very large single row.
            if current_rows and count_tokens(self._build_segment_text(header_text, current_rows)) > max_tokens:
                chunks.append(
                    self._build_chunk(
                        table_element=table_element,
                        doc_id=doc_id,
                        kb_id=kb_id,
                        heading_chain=heading_chain,
                        header_text=header_text,
                        rows=current_rows,
                        chunk_index=len(chunks),
                        table_name=table_name,
                        col_names=col_names,
                    )
                )
                current_rows = []

        if current_rows:
            chunks.append(
                self._build_chunk(
                    table_element=table_element,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    heading_chain=heading_chain,
                    header_text=header_text,
                    rows=current_rows,
                    chunk_index=len(chunks),
                    table_name=table_name,
                    col_names=col_names,
                )
            )

        return chunks

    @staticmethod
    def _split_header_and_data(lines: list[str]) -> tuple[list[str], list[str]]:
        header_lines: list[str] = []
        data_lines: list[str] = []
        for i, line in enumerate(lines):
            # Keep markdown header and separator lines.
            if i == 0:
                header_lines.append(line)
                continue
            if i == 1 and re.match(r"^\s*\|?[\s\-|:]+\|?\s*$", line):
                header_lines.append(line)
                continue
            data_lines.append(line)
        return header_lines, data_lines

    @staticmethod
    def _build_segment_text(header_text: str, rows: list[str]) -> str:
        rows_text = "\n".join(rows).strip()
        if header_text and rows_text:
            return f"{header_text}\n{rows_text}"
        if header_text:
            return header_text
        return rows_text

    def _build_chunk(
        self,
        table_element: DocumentElement,
        doc_id: str,
        kb_id: str,
        heading_chain: str,
        header_text: str,
        rows: list[str],
        chunk_index: int,
        table_name: str,
        col_names: list[str],
    ) -> ChunkNode:
        segment_content = self._build_segment_text(header_text, rows)
        summary = _make_summary(table_name, col_names, len(rows))
        return ChunkNode(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            kb_id=kb_id,
            chunk_index=chunk_index,
            content=segment_content,
            chunk_type="table",
            heading_chain=heading_chain,
            metadata={**table_element.metadata, "table_summary": summary},
        )


def _extract_col_names(header_lines: list[str]) -> list[str]:
    if not header_lines:
        return []
    first = header_lines[0]
    cols = [c.strip() for c in first.strip().strip("|").split("|")]
    return [c for c in cols if c]


def _make_summary(table_name: str, col_names: list[str], row_count: int) -> str:
    parts: list[str] = []
    if table_name:
        parts.append(table_name)
    if col_names:
        parts.append("columns: " + ", ".join(col_names))
    parts.append(f"{row_count} rows")
    return " | ".join(parts)


def _tail_rows_for_overlap(rows: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0 or not rows:
        return []

    tail: list[str] = []
    token_sum = 0
    for row in reversed(rows):
        row_tokens = count_tokens(row)
        if token_sum + row_tokens > overlap_tokens and tail:
            break
        tail.insert(0, row)
        token_sum += row_tokens
    return tail

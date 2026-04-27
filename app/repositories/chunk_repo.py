"""Repository for persisting parsed chunks into PostgreSQL."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import ChunkNode
from app.models.db_models import DocumentChunkRecord

_BATCH_SIZE = 500


class ChunkRepository:
    """Persistence helpers for ``py_document_chunk``."""

    @staticmethod
    async def replace_by_task(
        session: AsyncSession,
        *,
        task_id: str,
        file_id: str | None,
        doc_id: str,
        kb_id: str,
        chunks: list[ChunkNode],
    ) -> int:
        # Keep task-level idempotency for retries.
        await session.execute(
            delete(DocumentChunkRecord).where(DocumentChunkRecord.task_id == task_id)
        )

        if not chunks:
            return 0

        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "task_id": task_id,
                    "file_id": file_id,
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "heading_chain": chunk.heading_chain or "",
                    "token_count": chunk.token_count,
                    "content": chunk.content,
                    "parent_content": chunk.parent_content,
                    "chunk_metadata": chunk.metadata or {},
                }
            )

        for start in range(0, len(rows), _BATCH_SIZE):
            batch = rows[start : start + _BATCH_SIZE]
            await session.execute(insert(DocumentChunkRecord), batch)

        return len(rows)

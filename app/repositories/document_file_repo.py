"""Repository for Python-side document-file mapping records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.minio_client import to_db_file_path
from app.models.db_models import DocumentFileMap


class DocumentFileRepository:
    """CRUD helpers for ``py_document_file``."""

    @staticmethod
    async def get_by_file_id(
        session: AsyncSession,
        file_id: str,
    ) -> DocumentFileMap | None:
        stmt = select(DocumentFileMap).where(DocumentFileMap.file_id == file_id)
        return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def upsert(
        session: AsyncSession,
        *,
        file_id: str,
        doc_id: str,
        kb_id: str,
        file_path: str,
        file_type: str,
        filename: str | None = None,
        source_system: str = "java",
        source_metadata: dict[str, Any] | None = None,
    ) -> DocumentFileMap:
        values = {
            "file_id": file_id,
            "doc_id": doc_id,
            "kb_id": kb_id,
            # Canonical storage format: /bucket/object_key
            "file_path": to_db_file_path(file_path),
            "file_type": file_type.lower().strip("."),
            "filename": filename,
            "source_system": source_system,
            "source_metadata": source_metadata or {},
        }
        stmt = pg_insert(DocumentFileMap).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[DocumentFileMap.file_id],
            set_={
                "doc_id": values["doc_id"],
                "kb_id": values["kb_id"],
                "file_path": values["file_path"],
                "file_type": values["file_type"],
                "filename": values["filename"],
                "source_system": values["source_system"],
                "source_metadata": values["source_metadata"],
            },
        )
        await session.execute(stmt)
        return await DocumentFileRepository.get_by_file_id(session, file_id)

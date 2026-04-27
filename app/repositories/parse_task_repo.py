"""Repository for persistent parse task state transitions."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import DocumentFileMap, ParseTaskRecord


class ParseTaskRepository:
    """Status transition helpers for ``py_parse_task``."""

    @staticmethod
    async def get_by_task_id(
        session: AsyncSession,
        task_id: str,
    ) -> ParseTaskRecord | None:
        stmt = select(ParseTaskRecord).where(ParseTaskRecord.task_id == task_id)
        return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def create_pending(
        session: AsyncSession,
        *,
        task_id: str,
        file_id: str | None,
        file_record: DocumentFileMap | None = None,
    ) -> ParseTaskRecord:
        values = {
            "task_id": task_id,
            "file_id": file_id,
            "doc_id": file_record.doc_id if file_record else None,
            "kb_id": file_record.kb_id if file_record else None,
            "file_path": file_record.file_path if file_record else None,
            "file_type": file_record.file_type if file_record else None,
            "status": "pending",
            "error_message": None,
        }
        stmt = pg_insert(ParseTaskRecord).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ParseTaskRecord.task_id],
            set_={
                "file_id": values["file_id"],
                "doc_id": values["doc_id"],
                "kb_id": values["kb_id"],
                "file_path": values["file_path"],
                "file_type": values["file_type"],
                "status": "pending",
                "parse_engine": None,
                "quality_score": None,
                "element_count": 0,
                "chunk_count": 0,
                "milvus_count": 0,
                "es_count": 0,
                "started_at": None,
                "finished_at": None,
                "error_message": None,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await session.execute(stmt)
        return await ParseTaskRepository.get_by_task_id(session, task_id)

    @staticmethod
    async def mark_processing(
        session: AsyncSession,
        *,
        task_id: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        stmt = (
            update(ParseTaskRecord)
            .where(ParseTaskRecord.task_id == task_id)
            .values(
                status="processing",
                started_at=now,
                finished_at=None,
                error_message=None,
                updated_at=now,
            )
        )
        await session.execute(stmt)

    @staticmethod
    async def mark_retry(
        session: AsyncSession,
        *,
        task_id: str,
        error_message: str,
        retry_count: int,
    ) -> None:
        stmt = (
            update(ParseTaskRecord)
            .where(ParseTaskRecord.task_id == task_id)
            .values(
                status="retry",
                error_message=error_message[:4000],
                retry_count=retry_count,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.execute(stmt)

    @staticmethod
    async def mark_success(
        session: AsyncSession,
        *,
        task_id: str,
        parse_engine: str,
        quality_score: float,
        element_count: int,
        chunk_count: int,
        milvus_count: int,
        es_count: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        stmt = (
            update(ParseTaskRecord)
            .where(ParseTaskRecord.task_id == task_id)
            .values(
                status="success",
                parse_engine=parse_engine,
                quality_score=quality_score,
                element_count=element_count,
                chunk_count=chunk_count,
                milvus_count=milvus_count,
                es_count=es_count,
                error_message=None,
                finished_at=now,
                updated_at=now,
            )
        )
        await session.execute(stmt)

    @staticmethod
    async def mark_failed(
        session: AsyncSession,
        *,
        task_id: str,
        error_message: str,
        retry_count: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        stmt = (
            update(ParseTaskRecord)
            .where(ParseTaskRecord.task_id == task_id)
            .values(
                status="failed",
                error_message=error_message[:4000],
                retry_count=retry_count,
                finished_at=now,
                updated_at=now,
            )
        )
        await session.execute(stmt)

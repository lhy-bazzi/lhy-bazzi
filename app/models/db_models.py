"""SQLAlchemy ORM models for PostgreSQL persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class DocumentFileMap(Base):
    """Python-side file mapping table populated by upstream systems."""

    __tablename__ = "py_document_file"

    file_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_system: Mapped[str] = mapped_column(String(32), nullable=False, default="java")
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_py_document_file_doc_id", "doc_id"),
        Index("ix_py_document_file_kb_id", "kb_id"),
    )


class ParseTaskRecord(Base):
    """Persistent parse-task status record."""

    __tablename__ = "py_parse_task"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("py_document_file.file_id"), nullable=True
    )
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    kb_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    parse_engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    element_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    milvus_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    es_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_py_parse_task_status", "status"),
        Index("ix_py_parse_task_file_id", "file_id"),
        Index("ix_py_parse_task_doc_id", "doc_id"),
        Index("ix_py_parse_task_kb_id", "kb_id"),
    )


class DocumentChunkRecord(Base):
    """Chunk persistence table (full content + metadata)."""

    __tablename__ = "py_document_chunk"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("py_parse_task.task_id"), nullable=False
    )
    file_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("py_document_file.file_id"), nullable=True
    )
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    heading_chain: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parent_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_py_document_chunk_chunk_id"),
        UniqueConstraint("task_id", "chunk_index", name="uq_py_document_chunk_task_index"),
        Index("ix_py_document_chunk_task_id", "task_id"),
        Index("ix_py_document_chunk_file_id", "file_id"),
        Index("ix_py_document_chunk_doc_id", "doc_id"),
        Index("ix_py_document_chunk_kb_id", "kb_id"),
    )

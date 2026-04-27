"""Celery task for async document parsing with PostgreSQL persistence."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from app.core.database import get_db_session
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_file_repo import DocumentFileRepository
from app.repositories.parse_task_repo import ParseTaskRepository
from app.tasks.celery_app import celery_app

if TYPE_CHECKING:
    from app.models.chunk import ChunkNode

_worker_runtime_ready = False
_worker_runtime_lock = threading.Lock()
_worker_event_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_event_loop() -> asyncio.AbstractEventLoop:
    global _worker_event_loop
    if _worker_event_loop is None or _worker_event_loop.is_closed():
        _worker_event_loop = asyncio.new_event_loop()
    return _worker_event_loop


def _run_async(coro):
    """Run an async coroutine from synchronous Celery worker context."""
    loop = _get_worker_event_loop()
    return loop.run_until_complete(coro)


@celery_app.task(
    bind=True,
    name="app.tasks.parse_task.process_parse_task",
    max_retries=3,
    default_retry_delay=60,
)
def process_parse_task(self, message: dict) -> dict:
    """Process a document parse task.

    Expected *message* keys:
        file_id, task_id(optional)

    Returns a summary dict with status, chunk_count, etc.
    """
    task_id = (message.get("task_id") or uuid4().hex)[:64]
    file_id = message.get("file_id")

    if not file_id:
        raise ValueError("message.file_id is required")

    logger.info("Parse task started: task_id={} file_id={}", task_id, file_id)

    try:
        _ensure_worker_runtime_initialized()
        result = _run_async(_do_parse(task_id=task_id, file_id=file_id))
        return result

    except Exception as exc:
        retry_count = self.request.retries + 1
        logger.exception(
            "Parse task failed: task_id={} file_id={} retry={} error={}",
            task_id,
            file_id,
            retry_count,
            exc,
        )

        # Determine if retryable
        retryable = not isinstance(exc, (ValueError, KeyError))
        if retryable and self.request.retries < self.max_retries:
            _run_async(_mark_retry(task_id=task_id, error_message=str(exc), retry_count=retry_count))
            raise self.retry(exc=exc) from exc

        _run_async(_mark_failed(task_id=task_id, error_message=str(exc), retry_count=retry_count))
        return {"task_id": task_id, "file_id": file_id, "status": "failed", "error": str(exc)}


def _ensure_worker_runtime_initialized() -> None:
    """Ensure Celery worker process has initialized infrastructure clients."""
    global _worker_runtime_ready
    if _worker_runtime_ready:
        return
    with _worker_runtime_lock:
        if _worker_runtime_ready:
            return
        _run_async(_init_worker_runtime())
        _worker_runtime_ready = True


async def _init_worker_runtime() -> None:
    """Initialize components required by parse + chunk + indexing pipeline."""
    from app.config import get_settings
    from app.core.database import init_db
    from app.core.es_client import get_es, init_es
    from app.core.milvus_client import get_milvus, init_milvus
    from app.core.minio_client import init_minio
    from app.core.redis_client import get_redis, init_redis
    from app.services.embedding.embedder import EmbeddingService
    from app.services.embedding.model_manager import init_model_manager
    from app.services.indexing.es_indexer import ESIndexer
    from app.services.indexing.indexer import get_index_service, init_index_service
    from app.services.indexing.milvus_indexer import MilvusIndexer

    await init_db()
    await init_redis()
    init_minio()
    await init_milvus()
    await init_es()

    try:
        get_index_service()
    except RuntimeError:
        settings = get_settings()
        model_mgr = await init_model_manager(settings)
        embed_svc = EmbeddingService(model_mgr, get_redis(), settings)
        init_index_service(
            MilvusIndexer(get_milvus(), embed_svc),
            ESIndexer(get_es()),
        )
        logger.info("Celery worker runtime initialized for parse pipeline.")


async def _do_parse(task_id: str, file_id: str) -> dict:
    """Core parse logic (async)."""
    from app.config import get_settings
    from app.models.chunk import ChunkConfig
    from app.services.chunking import DocumentChunker
    from app.services.indexing.indexer import get_index_service
    from app.services.parsing.engine import get_parse_engine

    await _ensure_task_pending(task_id=task_id, file_id=file_id)
    file_record = await _get_file_record_or_raise(file_id)

    # Bind file details onto task row (idempotent).
    await _ensure_task_pending(task_id=task_id, file_id=file_id, file_record=file_record)
    await _mark_processing(task_id=task_id)

    engine = get_parse_engine()
    parsed_doc = await engine.parse_document(
        doc_id=file_record.doc_id,
        kb_id=file_record.kb_id,
        file_path=file_record.file_path,
        file_type=file_record.file_type,
    )

    logger.info(
        "Parse done: task_id={} engine={} elements={} quality={:.3f}",
        task_id, parsed_doc.parse_engine,
        len(parsed_doc.elements), parsed_doc.quality_score,
    )

    chunk_cfg = get_settings().parsing.chunk
    chunker = DocumentChunker(
        ChunkConfig(
            chunk_size=chunk_cfg.size,
            chunk_overlap=chunk_cfg.overlap,
            min_chunk_size=chunk_cfg.min_size,
            max_chunk_size=chunk_cfg.max_size,
            parent_chunk_size=chunk_cfg.parent_size,
        )
    )
    chunks = chunker.chunk_document(parsed_doc, kb_id=file_record.kb_id)

    # Persist full chunk payload to PostgreSQL first (content + metadata).
    pg_chunk_count = await _persist_chunks(
        task_id=task_id,
        file_id=file_record.file_id,
        doc_id=file_record.doc_id,
        kb_id=file_record.kb_id,
        chunks=chunks,
    )

    index_result = await get_index_service().index_chunks(chunks)
    if not index_result.success:
        raise RuntimeError(
            f"Indexing failed: milvus_error={index_result.milvus_error}, es_error={index_result.es_error}"
        )

    await _mark_success(
        task_id=task_id,
        parse_engine=parsed_doc.parse_engine,
        quality_score=parsed_doc.quality_score,
        element_count=len(parsed_doc.elements),
        chunk_count=pg_chunk_count,
        milvus_count=index_result.milvus_count,
        es_count=index_result.es_count,
    )

    return {
        "task_id": task_id,
        "file_id": file_record.file_id,
        "doc_id": file_record.doc_id,
        "kb_id": file_record.kb_id,
        "status": "success",
        "parse_engine": parsed_doc.parse_engine,
        "quality_score": parsed_doc.quality_score,
        "element_count": len(parsed_doc.elements),
        "chunk_count": pg_chunk_count,
        "milvus_count": index_result.milvus_count,
        "es_count": index_result.es_count,
    }


async def _get_file_record_or_raise(file_id: str):
    async with get_db_session() as session:
        file_record = await DocumentFileRepository.get_by_file_id(session, file_id)
        if file_record is None:
            raise ValueError(f"file_id '{file_id}' not found in py_document_file")
        return file_record


async def _ensure_task_pending(task_id: str, file_id: str, file_record=None) -> None:
    async with get_db_session() as session:
        await ParseTaskRepository.create_pending(
            session,
            task_id=task_id,
            file_id=file_id,
            file_record=file_record,
        )


async def _mark_processing(task_id: str) -> None:
    async with get_db_session() as session:
        await ParseTaskRepository.mark_processing(session, task_id=task_id)


async def _mark_success(
    task_id: str,
    parse_engine: str,
    quality_score: float,
    element_count: int,
    chunk_count: int,
    milvus_count: int,
    es_count: int,
) -> None:
    async with get_db_session() as session:
        await ParseTaskRepository.mark_success(
            session,
            task_id=task_id,
            parse_engine=parse_engine,
            quality_score=quality_score,
            element_count=element_count,
            chunk_count=chunk_count,
            milvus_count=milvus_count,
            es_count=es_count,
        )


async def _mark_retry(task_id: str, error_message: str, retry_count: int) -> None:
    try:
        async with get_db_session() as session:
            await ParseTaskRepository.mark_retry(
                session,
                task_id=task_id,
                error_message=error_message,
                retry_count=retry_count,
            )
    except Exception as exc:
        logger.warning("Failed to mark retry status: task_id={} error={}", task_id, exc)


async def _mark_failed(task_id: str, error_message: str, retry_count: int) -> None:
    try:
        async with get_db_session() as session:
            await ParseTaskRepository.mark_failed(
                session,
                task_id=task_id,
                error_message=error_message,
                retry_count=retry_count,
            )
    except Exception as exc:
        logger.warning("Failed to mark failed status: task_id={} error={}", task_id, exc)


async def _persist_chunks(
    task_id: str,
    file_id: str,
    doc_id: str,
    kb_id: str,
    chunks: list[ChunkNode],
) -> int:
    async with get_db_session() as session:
        return await ChunkRepository.replace_by_task(
            session,
            task_id=task_id,
            file_id=file_id,
            doc_id=doc_id,
            kb_id=kb_id,
            chunks=chunks,
        )

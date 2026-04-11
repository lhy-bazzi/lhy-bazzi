"""Celery task for async document parsing + Java callback."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from app.tasks.celery_app import celery_app


def _run_async(coro):
    """Run an async coroutine from synchronous Celery worker context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(
    bind=True,
    name="app.tasks.parse_task.process_parse_task",
    max_retries=2,
    default_retry_delay=60,
)
def process_parse_task(self, message: dict) -> dict:
    """Process a document parse task.

    Expected *message* keys (aligned with Java backend):
        task_id, doc_id, kb_id, file_path, file_type, callback_url

    Returns a summary dict with status, chunk_count, etc.
    """
    task_id = message.get("task_id", "unknown")
    doc_id = message.get("doc_id", "")
    kb_id = message.get("kb_id", "")
    file_path = message.get("file_path", "")
    file_type = message.get("file_type", "")
    callback_url = message.get("callback_url")

    logger.info("Parse task started: task_id={} doc_id={} file={}", task_id, doc_id, file_path)

    try:
        result = _run_async(_do_parse(task_id, doc_id, kb_id, file_path, file_type))
        # Callback Java
        if callback_url:
            _run_async(_callback_java(callback_url, {
                "task_id": task_id,
                "doc_id": doc_id,
                "status": "success",
                "parse_engine": result.get("parse_engine"),
                "quality_score": result.get("quality_score"),
                "chunk_count": result.get("chunk_count", 0),
                "milvus_count": result.get("milvus_count", 0),
                "es_count": result.get("es_count", 0),
            }))
        return result

    except Exception as exc:
        logger.exception("Parse task failed: task_id={} error={}", task_id, exc)

        # Determine if retryable
        retryable = not isinstance(exc, (ValueError, KeyError))
        if retryable and self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        # Not retryable or retries exhausted — report failure
        if callback_url:
            _run_async(_callback_java(callback_url, {
                "task_id": task_id,
                "doc_id": doc_id,
                "status": "failed",
                "error_message": str(exc),
            }))
        return {"task_id": task_id, "status": "failed", "error": str(exc)}


async def _do_parse(
    task_id: str, doc_id: str, kb_id: str, file_path: str, file_type: str
) -> dict:
    """Core parse logic (async)."""
    from app.services.parsing.engine import get_parse_engine
    from app.services.chunking import DocumentChunker
    from app.services.indexing.indexer import get_index_service

    engine = get_parse_engine()
    parsed_doc = await engine.parse_document(
        doc_id=doc_id,
        kb_id=kb_id,
        file_path=file_path,
        file_type=file_type,
    )

    logger.info(
        "Parse done: task_id={} engine={} elements={} quality={:.3f}",
        task_id, parsed_doc.parse_engine,
        len(parsed_doc.elements), parsed_doc.quality_score,
    )

    chunker = DocumentChunker()
    chunks = chunker.chunk_document(parsed_doc, kb_id=kb_id)

    index_result = await get_index_service().index_chunks(chunks)

    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "status": "success",
        "parse_engine": parsed_doc.parse_engine,
        "quality_score": parsed_doc.quality_score,
        "element_count": len(parsed_doc.elements),
        "chunk_count": len(chunks),
        "milvus_count": index_result.milvus_count,
        "es_count": index_result.es_count,
    }


async def _callback_java(url: str, payload: dict) -> None:
    """POST result back to the Java service."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            logger.info("Java callback {} → {}", url, resp.status_code)
    except Exception as exc:
        logger.warning("Java callback failed: {}", exc)

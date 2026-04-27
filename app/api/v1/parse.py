"""Document parsing API endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.schemas import ParseSubmitRequest, ParseStatusResponse

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("/submit")
async def submit_parse_task(request: ParseSubmitRequest):
    """Submit a parse task.

    Workflow:
    1. Optional upsert of file mapping (when doc fields are provided)
    2. Create persistent pending parse_task row in PostgreSQL
    3. Dispatch Celery message with task_id + file_id
    """
    from app.core.database import get_db_session
    from app.repositories.document_file_repo import DocumentFileRepository
    from app.repositories.parse_task_repo import ParseTaskRepository
    from app.tasks.parse_task import process_parse_task

    task_id = (request.task_id or uuid4().hex)[:64]

    # If any mapping fields are provided, require a complete mapping payload.
    mapping_fields = [request.doc_id, request.kb_id, request.file_path, request.file_type]
    if any(mapping_fields) and not all(mapping_fields):
        raise HTTPException(
            status_code=400,
            detail="doc_id, kb_id, file_path, file_type must be provided together when registering file mapping.",
        )

    async with get_db_session() as session:
        if all(mapping_fields):
            await DocumentFileRepository.upsert(
                session,
                file_id=request.file_id,
                doc_id=request.doc_id,
                kb_id=request.kb_id,
                file_path=request.file_path,
                file_type=request.file_type,
                filename=request.filename,
                source_system="api",
                source_metadata=request.source_metadata,
            )

        file_record = await DocumentFileRepository.get_by_file_id(session, request.file_id)
        if file_record is None:
            raise HTTPException(
                status_code=404,
                detail=f"file_id '{request.file_id}' not found in py_document_file.",
            )

        await ParseTaskRepository.create_pending(
            session,
            task_id=task_id,
            file_id=request.file_id,
            file_record=file_record,
        )

    message = {"task_id": task_id, "file_id": request.file_id}
    task = process_parse_task.apply_async(args=[message], task_id=task_id)
    logger.info(
        "Parse task submitted: task_id={} file_id={} celery_task={}",
        task_id,
        request.file_id,
        task.id,
    )

    return {
        "task_id": task_id,
        "file_id": request.file_id,
        "status": "pending",
        "message": "Task submitted to queue. Status is persisted in PostgreSQL.",
    }


@router.get("/status/{task_id}", response_model=ParseStatusResponse)
async def get_parse_status(task_id: str):
    """Query parse status from PostgreSQL."""
    from app.core.database import get_db_session
    from app.models.enums import ParseStatus
    from app.repositories.parse_task_repo import ParseTaskRepository

    async with get_db_session() as session:
        task = await ParseTaskRepository.get_by_task_id(session, task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found.")

    return ParseStatusResponse(
        task_id=task.task_id,
        file_id=task.file_id,
        doc_id=task.doc_id,
        kb_id=task.kb_id,
        status=ParseStatus(task.status)
        if task.status in ParseStatus._value2member_map_
        else ParseStatus.FAILED,
        parse_engine=task.parse_engine,
        quality_score=task.quality_score,
        element_count=task.element_count,
        chunk_count=task.chunk_count,
        milvus_count=task.milvus_count,
        es_count=task.es_count,
        retry_count=task.retry_count,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        updated_at=task.updated_at,
    )


@router.post("/retry/{task_id}")
async def retry_parse_task(task_id: str):
    """Retry a parse task by creating a new task_id for the same file_id."""
    from app.core.database import get_db_session
    from app.models.enums import ParseStatus
    from app.repositories.document_file_repo import DocumentFileRepository
    from app.repositories.parse_task_repo import ParseTaskRepository
    from app.tasks.parse_task import process_parse_task

    new_task_id = uuid4().hex[:64]

    async with get_db_session() as session:
        task = await ParseTaskRepository.get_by_task_id(session, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found.")

        if task.status in {
            ParseStatus.PENDING.value,
            ParseStatus.PROCESSING.value,
            ParseStatus.RETRY.value,
        }:
            raise HTTPException(
                status_code=409,
                detail=f"task_id '{task_id}' is still running and cannot be retried now.",
            )

        if not task.file_id:
            raise HTTPException(
                status_code=400,
                detail=f"task_id '{task_id}' has no file_id and cannot be retried.",
            )

        file_record = await DocumentFileRepository.get_by_file_id(session, task.file_id)
        if file_record is None:
            raise HTTPException(
                status_code=404,
                detail=f"file_id '{task.file_id}' not found in py_document_file.",
            )

        await ParseTaskRepository.create_pending(
            session,
            task_id=new_task_id,
            file_id=file_record.file_id,
            file_record=file_record,
        )

    message = {"task_id": new_task_id, "file_id": task.file_id}
    process_parse_task.apply_async(args=[message], task_id=new_task_id)

    return {
        "previous_task_id": task_id,
        "task_id": new_task_id,
        "file_id": task.file_id,
        "status": "pending",
        "message": "Retry task submitted.",
    }

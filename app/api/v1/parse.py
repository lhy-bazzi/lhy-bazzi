"""Document parsing API endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from app.models.schemas import ParseSubmitRequest, ParseStatusResponse

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("/submit")
async def submit_parse_task(request: ParseSubmitRequest):
    """Submit a document parsing task (manual trigger, supplements MQ).

    Dispatches to Celery for async processing.
    """
    from app.tasks.parse_task import process_parse_task

    message = {
        "task_id": request.doc_id,  # Use doc_id as task_id if none provided
        "doc_id": request.doc_id,
        "kb_id": request.kb_id,
        "file_path": request.file_path,
        "file_type": request.file_type,
        "callback_url": request.callback_url,
    }

    task = process_parse_task.delay(message)
    logger.info("Parse task submitted: doc_id={} celery_task={}", request.doc_id, task.id)

    return {
        "task_id": task.id,
        "doc_id": request.doc_id,
        "status": "pending",
        "message": "Task submitted to processing queue.",
    }


@router.get("/status/{task_id}")
async def get_parse_status(task_id: str):
    """Query the status of a Celery parse task."""
    from app.tasks.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status.lower() if result.status else "unknown",
    }

    if result.ready() and result.successful():
        response.update(result.result or {})
    elif result.failed():
        response["status"] = "failed"
        response["error_message"] = str(result.result) if result.result else None

    return response


@router.post("/retry/{task_id}")
async def retry_parse_task(task_id: str):
    """Retry a failed parsing task by re-submitting to Celery.

    Requires the original message to be stored (e.g. in PgSQL).
    For now this is a placeholder — full implementation in Phase 3+.
    """
    # TODO: look up original message from parse_task table, re-submit
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Retry requires original task record from PgSQL (Phase 3+).",
    }

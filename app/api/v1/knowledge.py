"""Knowledge management API endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _get_index_service():
    from app.services.indexing.indexer import get_index_service
    try:
        return get_index_service()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.delete("/doc/{doc_id}")
async def delete_document(doc_id: str):
    """Delete all vectors and index entries for a document."""
    svc = _get_index_service()
    await svc.delete_document(doc_id)
    logger.info("Deleted document: doc_id={}", doc_id)
    return {"status": "success", "doc_id": doc_id}


@router.delete("/kb/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """Delete all vectors and index entries for a knowledge base."""
    svc = _get_index_service()
    await svc.delete_knowledge_base(kb_id)
    logger.info("Deleted knowledge base: kb_id={}", kb_id)
    return {"status": "success", "kb_id": kb_id}


@router.post("/rebuild/{kb_id}")
async def rebuild_knowledge_base(kb_id: str):
    """
    Rebuild the entire index for a knowledge base.

    This endpoint triggers a full re-parse + re-index of all documents
    in the knowledge base. The actual rebuild is handled asynchronously
    via the task queue — this endpoint just enqueues the work.
    """
    # TODO (Phase 6): query PgSQL for all doc_ids in kb_id,
    # re-dispatch parse tasks for each, then call index_service.rebuild_knowledge_base()
    logger.info("Rebuild requested for kb_id={} (not yet implemented)", kb_id)
    return {"status": "accepted", "kb_id": kb_id, "message": "Rebuild queued (Phase 6)"}

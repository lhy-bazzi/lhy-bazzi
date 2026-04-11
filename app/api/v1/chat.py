"""Intelligent Q&A API endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_redis_dep
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunkResponse,
)
from app.services.retrieval.models import RetrievalConfig

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_qa_engine():
    from app.core.qa import get_qa_engine
    return get_qa_engine()


async def _event_generator(request: ChatRequest) -> AsyncGenerator[dict, None]:
    engine = _get_qa_engine()
    cfg = {
        "retrieval_mode": request.config.retrieval_mode.value,
        "top_k": request.config.top_k,
        "rerank": request.config.rerank,
        "model": request.config.model,
    }
    try:
        async for event in engine.chat(
            query=request.query,
            kb_ids=request.kb_ids,
            user_id=request.user_id,
            chat_history=request.chat_history,
            config=cfg,
            stream=True,
        ):
            yield {"event": event.event, "data": event.json()}
    except Exception as exc:
        yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}


@router.post("/completions")
async def chat_completions(request: ChatRequest):
    """Knowledge-base Q&A — SSE streaming or JSON response."""
    if request.stream:
        return EventSourceResponse(_event_generator(request))

    # Non-streaming: collect all events
    engine = _get_qa_engine()
    cfg = {
        "retrieval_mode": request.config.retrieval_mode.value,
        "top_k": request.config.top_k,
        "rerank": request.config.rerank,
        "model": request.config.model,
    }
    answer_parts: list[str] = []
    citations: list[dict] = []

    async for event in engine.chat(
        query=request.query,
        kb_ids=request.kb_ids,
        user_id=request.user_id,
        chat_history=request.chat_history,
        config=cfg,
        stream=False,
    ):
        if event.event == "answer":
            token = event.data.get("token", "") if isinstance(event.data, dict) else ""
            answer_parts.append(token)
        elif event.event == "citations":
            citations = event.data.get("citations", []) if isinstance(event.data, dict) else []

    return ChatResponse(
        answer="".join(answer_parts),
        citations=[
            Citation(
                chunk_id=c.get("chunk_id", ""),
                doc_name=c.get("doc_name"),
                page=c.get("page"),
                highlight=c.get("highlight"),
            )
            for c in citations
        ],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_only(
    request: RetrieveRequest,
    redis=Depends(get_redis_dep),
):
    """Retrieve-only endpoint for debugging retrieval quality."""
    from app.core.retrieval import get_retriever

    retriever = get_retriever()
    cfg = RetrievalConfig(
        retrieval_mode=request.config.retrieval_mode.value,
        top_k=request.config.top_k,
        rerank=request.config.rerank,
    )
    result = await retriever.retrieve(
        query=request.query,
        user_id=request.user_id,
        kb_ids=request.kb_ids,
        config=cfg,
    )
    return RetrieveResponse(
        chunks=[
            RetrievedChunkResponse(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                kb_id=c.kb_id,
                content=c.content,
                heading_chain=c.heading_chain,
                chunk_type=c.chunk_type,
                score=c.score,
                doc_name=c.doc_name,
                page=c.page,
            )
            for c in result.chunks
        ],
        total_retrieved=result.total_retrieved,
        retrieval_mode=result.retrieval_mode,
        latency_ms=result.latency_ms,
    )

"""Pydantic request / response schemas for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models.enums import ParseStatus, QAMode, RetrievalMode, TraceLevel


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

class ParseSubmitRequest(BaseModel):
    file_id: str = Field(..., description="Upstream file identifier")
    task_id: Optional[str] = Field(None, description="Optional custom parse task ID")
    doc_id: Optional[str] = None
    kb_id: Optional[str] = None
    file_path: Optional[str] = Field(None, description="MinIO object path")
    file_type: Optional[str] = None
    filename: Optional[str] = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ParseStatusResponse(BaseModel):
    task_id: str
    file_id: Optional[str] = None
    doc_id: Optional[str] = None
    kb_id: Optional[str] = None
    status: ParseStatus
    parse_engine: Optional[str] = None
    quality_score: Optional[float] = None
    element_count: Optional[int] = None
    chunk_count: Optional[int] = None
    milvus_count: Optional[int] = None
    es_count: Optional[int] = None
    retry_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Chat / QA
# ---------------------------------------------------------------------------

class ChatConfig(BaseModel):
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    qa_mode: QAMode = QAMode.AUTO
    top_k: int = 10
    rerank: bool = True
    model: Optional[str] = None
    trace_enabled: bool = True
    trace_level: TraceLevel = TraceLevel.PRO


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    kb_ids: list[str]
    user_id: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    stream: bool = True
    config: ChatConfig = Field(default_factory=ChatConfig)


class Citation(BaseModel):
    chunk_id: str
    doc_name: Optional[str] = None
    page: Optional[int] = None
    highlight: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: Optional[dict[str, int]] = None


class ChatStreamEvent(BaseModel):
    event: str  # retrieval / reasoning / answer / citations / done
    data: Any


# ---------------------------------------------------------------------------
# Retrieve (debug)
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    kb_ids: list[str]
    user_id: str
    config: ChatConfig = Field(default_factory=ChatConfig)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    heading_chain: str = ""
    chunk_type: str = "text"
    score: float = 0.0
    doc_name: Optional[str] = None
    page: Optional[int] = None


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunkResponse]
    total_retrieved: int
    retrieval_mode: str
    latency_ms: int


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    name: str
    status: str  # "ok" / "error"
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" / "degraded" / "unhealthy"
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)

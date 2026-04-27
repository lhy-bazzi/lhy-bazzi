from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents.retrieval_agent import retrieval_agent
from app.api.v1 import chat as chat_api
from app.api.v1.chat import _normalize_model_id
from app.models.schemas import ChatRequest
from app.services.qa.qa_engine import ChatStreamEvent, QAEngine
from app.services.retrieval.models import RetrievalResult, RetrievedChunk


class _DummyRetriever:
    async def retrieve(self, query, user_id, kb_ids, config):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    kb_id=kb_ids[0],
                    content=f"context for {query}",
                    heading_chain="",
                    chunk_type="text",
                    score=0.9,
                    doc_name="doc-a",
                    page=1,
                )
            ],
            total_retrieved=3,
            retrieval_mode=config.retrieval_mode,
            latency_ms=12,
            debug={
                "dense_count": 1,
                "sparse_count": 1,
                "bm25_count": 1,
                "fused_count": 1,
                "reranked_count": 1,
                "after_permission_count": 1,
                "es_keywords": ["年假", "政策"],
                "bm25_preview": [
                    {
                        "source": "bm25",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "doc_name": "doc-a",
                        "score": 0.88,
                        "snippet": "命中片段",
                    }
                ],
                "vector_preview": [
                    {
                        "source": "dense",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "doc_name": "doc-a",
                        "score": 0.91,
                        "snippet": "向量命中片段",
                    }
                ],
                "selected_evidence_preview": [
                    {
                        "source": "final",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "doc_name": "doc-a",
                        "score": 0.95,
                        "snippet": "最终证据片段",
                    }
                ],
            },
        )


class _DummySynthesizer:
    async def synthesize(self, query, contexts, chat_history, stream, model):
        yield "ok"

    def extract_citations(self, answer, contexts):
        return []


class _DummyQueryUnderstanding:
    def __init__(self, plan):
        self._plan = plan

    async def process(self, query, chat_history):
        return self._plan


def _build_engine_with_plan(plan) -> QAEngine:
    return QAEngine(
        query_understanding=_DummyQueryUnderstanding(plan),
        hybrid_retriever=_DummyRetriever(),
        response_synthesizer=_DummySynthesizer(),
        llm_provider=SimpleNamespace(),
        settings=SimpleNamespace(qa=SimpleNamespace(max_iterations=1)),
    )


async def _collect_events(gen):
    events = []
    async for e in gen:
        events.append(e)
    return events


def test_chat_request_rejects_invalid_history_shape():
    with pytest.raises(ValidationError):
        ChatRequest(
            query="hello",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[{"additionalProp1": "x"}],
        )


def test_normalize_model_alias_case_insensitive():
    assert _normalize_model_id("qwen-max") == "openai/qwen-max"
    assert _normalize_model_id("QWEN-MAX") == "openai/qwen-max"
    assert _normalize_model_id(" openai/qwen-max ") == "openai/qwen-max"


def test_quality_grade_mapping():
    assert QAEngine._quality_grade({"relevance": 4, "completeness": 4, "faithfulness": 5}) == "high"
    assert QAEngine._quality_grade({"relevance": 3, "completeness": 3, "faithfulness": 3}) == "medium"
    assert QAEngine._quality_grade({"relevance": 2, "completeness": 3, "faithfulness": 2}) == "low"
    assert QAEngine._quality_grade({}) == "unknown"


def test_trace_event_layered_structure_is_backward_compatible():
    evt = QAEngine._build_trace_event(
        request_id="req-1",
        step="retrieve",
        title="检索完成",
        detail="已完成检索。",
        metrics={"dense_hits": 12, "latency_ms": 88},
        level="basic",
        phase="retrieval",
        status="completed",
        node="retriever",
    )
    data = evt.data

    # Backward-compatible fields are still present
    assert data["title"] == "检索完成"
    assert data["detail"] == "已完成检索。"
    assert data["metrics"]["dense_hits"] == 12

    # New layered fields for richer frontend display
    assert data["schema_version"] == "trace.v2"
    assert data["user_view"]["headline"] == "检索完成"
    assert data["user_view"]["phase"] == "retrieval"
    assert isinstance(data["user_view"]["cards"], list)
    assert data["engine_view"]["node"] == "retriever"
    assert data["timeline"]["step_total"] >= 1
    assert data["ui_hints"]["icon"] == "search"
    # basic level hides verbose internals in engine_view metrics
    assert "dense_hits" not in data["engine_view"]["metrics"]
    assert data["engine_view"]["metrics"]["latency_ms"] == 88
    # basic level keeps only concise user cards
    assert len(data["user_view"]["cards"]) <= 2


@pytest.mark.asyncio
async def test_event_generator_passes_normalized_model(monkeypatch):
    received = {}

    class _DummyEngine:
        async def chat(self, query, kb_ids, user_id, chat_history, config, stream):
            received.update(config)
            yield ChatStreamEvent(event="done", data={"answer": "", "citations": []})

    monkeypatch.setattr(chat_api, "_get_qa_engine", lambda: _DummyEngine())
    req = ChatRequest(
        query="hi",
        kb_ids=["kb1"],
        user_id="u1",
        chat_history=[],
        stream=True,
        config={"model": "qwen-max"},
    )

    events = [e async for e in chat_api._event_generator(req)]
    assert events[0]["event"] == "done"
    assert received["model"] == "openai/qwen-max"


@pytest.mark.asyncio
async def test_qa_engine_emits_trace_events_when_enabled():
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="factual"),
        strategy="simple_rag",
        sub_queries=[],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)

    events = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"trace_enabled": True, "qa_mode": "auto"},
            stream=True,
        )
    )
    trace_steps = [e.data["step"] for e in events if e.event == "trace"]
    assert "start" in trace_steps
    assert "query_understanding" in trace_steps
    assert "route" in trace_steps
    assert "retrieve" in trace_steps


@pytest.mark.asyncio
async def test_simple_path_retrieval_event_contains_explainability_fields():
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="factual"),
        strategy="simple_rag",
        sub_queries=[],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)

    events = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"trace_enabled": True, "qa_mode": "auto"},
            stream=True,
        )
    )
    retrieval_evt = next(e for e in events if e.event == "retrieval")
    data = retrieval_evt.data
    assert "retrieval_explain" in data
    assert data["retrieval_explain"]["keywords"] == ["年假", "政策"]
    assert isinstance(data["retrieval_explain"]["es_hits"], list)
    assert isinstance(data["retrieval_explain"]["vector_hits"], list)
    assert isinstance(data["retrieval_explain"]["selected_evidence"], list)


@pytest.mark.asyncio
async def test_qa_mode_deep_forces_multi_agent(monkeypatch):
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="factual"),
        strategy="simple_rag",
        sub_queries=[],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)
    called = {"simple": 0, "deep": 0}

    async def _fake_simple(self, *args, **kwargs):
        called["simple"] += 1
        yield ChatStreamEvent(event="done", data={"answer": "simple", "citations": []})

    async def _fake_deep(self, *args, **kwargs):
        called["deep"] += 1
        yield ChatStreamEvent(event="done", data={"answer": "deep", "citations": []})

    monkeypatch.setattr(QAEngine, "_simple_rag", _fake_simple, raising=True)
    monkeypatch.setattr(QAEngine, "_multi_agent", _fake_deep, raising=True)

    _ = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"qa_mode": "deep", "trace_enabled": False},
            stream=True,
        )
    )
    assert called["simple"] == 0
    assert called["deep"] == 1


@pytest.mark.asyncio
async def test_qa_mode_deep_does_not_override_conversational(monkeypatch):
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="conversational"),
        strategy="conversational",
        sub_queries=[],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)
    called = {"conv": 0, "deep": 0}

    async def _fake_conv(self, *args, **kwargs):
        called["conv"] += 1
        yield ChatStreamEvent(event="done", data={"answer": "conv", "citations": []})

    async def _fake_deep(self, *args, **kwargs):
        called["deep"] += 1
        yield ChatStreamEvent(event="done", data={"answer": "deep", "citations": []})

    monkeypatch.setattr(QAEngine, "_handle_conversational", _fake_conv, raising=True)
    monkeypatch.setattr(QAEngine, "_multi_agent", _fake_deep, raising=True)

    _ = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"qa_mode": "deep", "trace_enabled": False},
            stream=True,
        )
    )
    assert called["conv"] == 1
    assert called["deep"] == 0


@pytest.mark.asyncio
async def test_retrieval_agent_returns_per_query_traces(monkeypatch):
    from app.core import retrieval as retrieval_core

    retriever = _DummyRetriever()
    monkeypatch.setattr(retrieval_core, "get_retriever", lambda: retriever, raising=True)

    out = await retrieval_agent({
        "query": "Q0",
        "chat_history": [],
        "user_id": "u1",
        "kb_ids": ["kb1"],
        "query_plan": {"primary_query": "Q0", "hyde_text": "Q-hyde"},
        "sub_queries": ["Q-sub1"],
        "retrieved_contexts": [],
        "retrieval_rounds": 0,
        "sub_answers": [],
        "reasoning_notes": "",
        "final_answer": "",
        "citations": [],
        "stream_tokens": [],
        "routing_decision": "complex",
        "quality_check": None,
        "iteration_count": 0,
        "max_iterations": 1,
        "should_continue": False,
        "retrieval_config": {"retrieval_mode": "hybrid", "top_k": 10, "rerank": True},
        "model": None,
    })

    assert len(out["retrieval_traces"]) == 3
    assert all("latency_ms" in t for t in out["retrieval_traces"])
    assert len(out["retrieved_contexts"]) == 1


@pytest.mark.asyncio
async def test_deep_path_trace_contains_retrieval_metrics(monkeypatch):
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="multi_hop"),
        strategy="multi_agent",
        sub_queries=["q1"],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)

    class _FakeGraph:
        async def astream(self, initial_state):
            yield {
                "retriever": {
                    "retrieved_contexts": [{"chunk_id": "c1"}],
                    "retrieval_traces": [
                        {"debug": {"dense_count": 2, "sparse_count": 1, "bm25_count": 3, "fused_count": 4}}
                    ],
                }
            }
            yield {"reasoner": {"sub_answers": [{"query": "q1", "sufficient": True}]}}
            yield {"synthesizer": {"final_answer": "ans", "citations": []}}
            yield {"critic": {"quality_check": {"passed": True}}}

    monkeypatch.setattr(QAEngine, "_get_graph", lambda self: _FakeGraph(), raising=True)

    events = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"qa_mode": "deep", "trace_enabled": True},
            stream=True,
        )
    )
    retrieve_trace = next(e for e in events if e.event == "trace" and e.data.get("step") == "retrieve")
    assert retrieve_trace.data["metrics"]["query_rounds"] == 1
    assert retrieve_trace.data["metrics"]["dense_hits"] == 2
    assert retrieve_trace.data["metrics"]["query_profiles"][0]["dense_hits"] == 2
    assert retrieve_trace.data["ui_hints"]["icon"] == "search"
    assert retrieve_trace.data["timeline"]["step_total"] >= retrieve_trace.data["timeline"]["step_index"]
    assert len(retrieve_trace.data["user_view"]["cards"]) >= 1


@pytest.mark.asyncio
async def test_deep_path_retrieval_event_contains_round_profiles(monkeypatch):
    plan = SimpleNamespace(
        original_query="q",
        resolved_query="q",
        rewritten_query="q",
        intent=SimpleNamespace(value="multi_hop"),
        strategy="multi_agent",
        sub_queries=["q1"],
        hyde_text="",
        primary_query="q",
    )
    engine = _build_engine_with_plan(plan)

    class _FakeGraph:
        async def astream(self, initial_state):
            yield {
                "retriever": {
                    "retrieved_contexts": [{"chunk_id": "c1"}],
                    "retrieval_traces": [
                        {
                            "query_preview": "q1",
                            "latency_ms": 25,
                            "total_retrieved": 5,
                            "final_count": 2,
                            "debug": {
                                "dense_count": 2,
                                "sparse_count": 1,
                                "bm25_count": 2,
                                "fused_count": 3,
                                "es_keywords": ["预算", "审批"],
                                "bm25_preview": [{"chunk_id": "c1", "snippet": "es hit"}],
                                "vector_preview": [{"chunk_id": "c1", "snippet": "vec hit"}],
                                "selected_evidence_preview": [{"chunk_id": "c1", "snippet": "selected"}],
                            },
                        }
                    ],
                }
            }
            yield {"reasoner": {"sub_answers": [{"query": "q1", "sufficient": True}]}}
            yield {"synthesizer": {"final_answer": "ans", "citations": []}}
            yield {"critic": {"quality_check": {"passed": True}}}

    monkeypatch.setattr(QAEngine, "_get_graph", lambda self: _FakeGraph(), raising=True)
    events = await _collect_events(
        engine.chat(
            query="q",
            kb_ids=["kb1"],
            user_id="u1",
            chat_history=[],
            config={"qa_mode": "deep", "trace_enabled": True},
            stream=True,
        )
    )
    retrieval_evt = next(
        e
        for e in events
        if e.event == "retrieval" and isinstance(e.data, dict) and e.data.get("node") == "retriever"
    )
    assert retrieval_evt.data["query_rounds"] == 1
    assert retrieval_evt.data["rounds"][0]["keywords"] == ["预算", "审批"]

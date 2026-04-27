"""QA Engine: unified entry point for intelligent question answering."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

from app.services.retrieval.models import RetrievalConfig, RetrievalResult

_ALLOWED_QA_MODES = {"auto", "simple", "deep"}
_ALLOWED_TRACE_LEVELS = {"basic", "pro"}
_BASIC_ENGINE_METRIC_KEYS = {
    "latency_ms",
    "total_latency_ms",
    "new_contexts",
    "query_rounds",
    "citation_count",
    "answer_length",
}
_TRACE_PROGRESS = {
    "start": 5,
    "query_understanding": 20,
    "qa_mode_override": 25,
    "route": 30,
    "deep_qa": 35,
    "retrieve": 55,
    "reasoning": 72,
    "synthesize": 88,
    "quality_gate": 95,
    "done": 100,
}
_TRACE_STEP_SEQUENCE = [
    "start",
    "query_understanding",
    "qa_mode_override",
    "route",
    "deep_qa",
    "retrieve",
    "reasoning",
    "synthesize",
    "quality_gate",
    "done",
]
_TRACE_STEP_ORDER = {step: i + 1 for i, step in enumerate(_TRACE_STEP_SEQUENCE)}
_TRACE_UI_HINTS = {
    "start": {"icon": "inbox", "tone": "info"},
    "query_understanding": {"icon": "brain", "tone": "info"},
    "qa_mode_override": {"icon": "tune", "tone": "info"},
    "route": {"icon": "route", "tone": "info"},
    "deep_qa": {"icon": "sparkles", "tone": "info"},
    "retrieve": {"icon": "search", "tone": "info"},
    "reasoning": {"icon": "workflow", "tone": "info"},
    "synthesize": {"icon": "message-square", "tone": "success"},
    "quality_gate": {"icon": "shield-check", "tone": "success"},
    "done": {"icon": "check-circle", "tone": "success"},
}


def _normalize_qa_mode(value: Any) -> str:
    raw = str(value or "auto").lower()
    return raw if raw in _ALLOWED_QA_MODES else "auto"


def _normalize_trace_level(value: Any) -> str:
    raw = str(value or "pro").lower()
    return raw if raw in _ALLOWED_TRACE_LEVELS else "pro"


@dataclass
class ChatStreamEvent:
    event: str  # trace | retrieval | reasoning | answer | citations | done
    data: Any

    def json(self) -> str:
        import json

        return json.dumps(self.data, ensure_ascii=False, default=str)


class QAEngine:
    def __init__(self, query_understanding, hybrid_retriever, response_synthesizer, llm_provider, settings):
        self.query_understanding = query_understanding
        self.retriever = hybrid_retriever
        self.synthesizer = response_synthesizer
        self.llm = llm_provider
        self.settings = settings
        self._graph = None  # lazy init

    def _get_graph(self):
        if self._graph is None:
            from app.agents.graph import build_deep_qa_graph

            self._graph = build_deep_qa_graph()
        return self._graph

    @staticmethod
    def _build_trace_event(
        *,
        request_id: str,
        step: str,
        title: str,
        detail: str,
        metrics: Optional[dict[str, Any]] = None,
        level: str = "pro",
        phase: Optional[str] = None,
        status: str = "running",
        node: Optional[str] = None,
        next_step: Optional[str] = None,
        engine_detail: Optional[str] = None,
        tags: Optional[list[str]] = None,
        summary_cards: Optional[list[dict[str, Any]]] = None,
    ) -> ChatStreamEvent:
        raw_metrics = metrics or {}
        if level == "basic":
            engine_metrics = {
                k: v
                for k, v in raw_metrics.items()
                if k in _BASIC_ENGINE_METRIC_KEYS
            }
        else:
            engine_metrics = raw_metrics

        trace_phase = phase or step
        progress = _TRACE_PROGRESS.get(step)
        step_index = _TRACE_STEP_ORDER.get(step)
        ui_hints = _TRACE_UI_HINTS.get(step, {"icon": "activity", "tone": "info"})
        cards = summary_cards or []
        if level == "basic":
            cards = cards[:2]
        return ChatStreamEvent(
            event="trace",
            data={
                "request_id": request_id,
                "step": step,
                "title": title,
                "detail": detail,
                "metrics": raw_metrics,
                "level": level,
                "ts": int(time.time() * 1000),
                "schema_version": "trace.v2",
                "phase": trace_phase,
                "status": status,
                "progress": progress,
                "tags": tags or [],
                "timeline": {
                    "step_index": step_index,
                    "step_total": len(_TRACE_STEP_SEQUENCE),
                },
                "ui_hints": {
                    "icon": ui_hints["icon"],
                    "tone": ui_hints["tone"],
                    "collapsible_metrics": level == "pro",
                },
                "user_view": {
                    "phase": trace_phase,
                    "status": status,
                    "progress": progress,
                    "headline": title,
                    "message": detail,
                    "cards": cards,
                },
                "engine_view": {
                    "phase": trace_phase,
                    "node": node or step,
                    "summary": engine_detail or detail,
                    "next_step": next_step,
                    "metrics": engine_metrics,
                },
            },
        )

    @staticmethod
    def _preview_text(text: Any, limit: int = 120) -> str:
        raw = "" if text is None else str(text)
        clean = " ".join(raw.split())
        if len(clean) <= limit:
            return clean
        return f"{clean[: limit - 3]}..."

    @staticmethod
    def _top_sources_from_retrieved_chunks(chunks: list[Any], limit: int = 3) -> list[dict[str, Any]]:
        ranked = sorted(chunks, key=lambda x: x.score, reverse=True)[:limit]
        return [
            {
                "source": c.doc_name or c.doc_id,
                "chunk_id": c.chunk_id,
                "score": round(float(c.score), 4),
                "page": c.page,
            }
            for c in ranked
        ]

    @staticmethod
    def _top_sources_from_state_contexts(contexts: list[dict], limit: int = 3) -> list[dict[str, Any]]:
        ranked = sorted(contexts, key=lambda x: x.get("score", 0), reverse=True)[:limit]
        out = []
        for c in ranked:
            score = c.get("score", 0)
            try:
                score = round(float(score), 4)
            except Exception:
                score = 0.0
            out.append({
                "source": c.get("doc_name") or c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "score": score,
                "page": c.get("page"),
            })
        return out

    @staticmethod
    def _quality_grade(quality: dict[str, Any]) -> str:
        if not quality:
            return "unknown"
        scores = []
        for key in ("relevance", "completeness", "faithfulness"):
            value = quality.get(key)
            if isinstance(value, (int, float)):
                scores.append(float(value))
        if not scores:
            return "unknown"
        avg = sum(scores) / len(scores)
        if avg >= 4.0:
            return "high"
        if avg >= 3.0:
            return "medium"
        return "low"

    @staticmethod
    def _apply_qa_mode(plan, qa_mode: str) -> tuple[str, str]:
        original_strategy = plan.strategy

        if qa_mode == "simple" and plan.strategy != "conversational":
            plan.strategy = "simple_rag"
        elif qa_mode == "deep" and plan.strategy != "conversational":
            plan.strategy = "multi_agent"
            if not plan.sub_queries:
                seed_query = plan.rewritten_query or plan.resolved_query or plan.original_query
                plan.sub_queries = [seed_query]
            if not plan.primary_query:
                plan.primary_query = plan.rewritten_query or plan.resolved_query or plan.original_query

        return original_strategy, plan.strategy

    async def chat(
        self,
        query: str,
        kb_ids: list[str],
        user_id: str,
        chat_history: list[dict] | None = None,
        config: Optional[dict] = None,
        stream: bool = True,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        cfg = config or {}
        request_id = uuid4().hex[:12]
        qa_mode = _normalize_qa_mode(cfg.get("qa_mode"))
        trace_enabled = bool(cfg.get("trace_enabled", True))
        trace_level = _normalize_trace_level(cfg.get("trace_level"))
        model = cfg.get("model")

        retrieval_cfg = RetrievalConfig(
            retrieval_mode=cfg.get("retrieval_mode", "hybrid"),
            top_k=cfg.get("top_k", 10),
            rerank=cfg.get("rerank", True),
        )

        if trace_enabled:
            yield self._build_trace_event(
                request_id=request_id,
                step="start",
                title="问题已接入",
                detail="正在理解问题并规划检索/推理执行路径。",
                metrics={
                    "qa_mode": qa_mode,
                    "retrieval_mode": retrieval_cfg.retrieval_mode,
                    "top_k": retrieval_cfg.top_k,
                    "rerank": retrieval_cfg.rerank,
                },
                level=trace_level,
                phase="intake",
                status="completed",
                node="qa_engine",
                next_step="query_understanding",
                engine_detail=(
                    "Request accepted. Building execution plan with qa_mode="
                    f"{qa_mode}, retrieval_mode={retrieval_cfg.retrieval_mode}."
                ),
                tags=["planning", "routing"],
                summary_cards=[
                    {"label": "问答模式", "value": qa_mode.upper()},
                    {"label": "检索策略", "value": str(retrieval_cfg.retrieval_mode)},
                    {"label": "召回条数", "value": retrieval_cfg.top_k},
                ],
            )

        # 1. Query understanding
        plan = await self.query_understanding.process(query, chat_history)
        original_strategy, final_strategy = self._apply_qa_mode(plan, qa_mode)

        logger.info(
            "QA intent={} strategy={} (requested_mode={}) query='{}'",
            plan.intent,
            final_strategy,
            qa_mode,
            query[:60],
        )

        if trace_enabled:
            intent_value = plan.intent.value if hasattr(plan.intent, "value") else str(plan.intent)
            yield self._build_trace_event(
                request_id=request_id,
                step="query_understanding",
                title="查询理解完成",
                detail=(
                    "已完成意图识别、查询改写和路由决策，"
                    f"当前采用 {final_strategy} 执行路径。"
                ),
                metrics={
                    "intent": intent_value,
                    "strategy": final_strategy,
                    "sub_query_count": len(plan.sub_queries),
                    "resolved_query_preview": self._preview_text(plan.resolved_query),
                    "rewritten_query_preview": self._preview_text(plan.rewritten_query),
                    "primary_query_preview": self._preview_text(plan.primary_query),
                    "sub_queries_preview": [self._preview_text(s, 80) for s in (plan.sub_queries or [])[:4]],
                },
                level=trace_level,
                phase="understanding",
                status="completed",
                node="query_understanding",
                next_step="route",
                engine_detail=(
                    f"Intent={intent_value}, strategy={final_strategy}, "
                    f"sub_queries={len(plan.sub_queries)}."
                ),
                tags=["intent", "rewrite", "route"],
                summary_cards=[
                    {"label": "识别意图", "value": intent_value},
                    {"label": "执行路径", "value": final_strategy},
                    {"label": "子查询数", "value": len(plan.sub_queries)},
                ],
            )
            if original_strategy != final_strategy:
                yield self._build_trace_event(
                    request_id=request_id,
                    step="qa_mode_override",
                    title="策略模式已锁定",
                    detail=f"根据 qa_mode={qa_mode}，执行路径已切换为 {final_strategy}。",
                    metrics={"requested_mode": qa_mode},
                    level=trace_level,
                    phase="routing",
                    status="completed",
                    node="qa_engine",
                    next_step=final_strategy,
                    engine_detail=(
                        f"Strategy overridden by qa_mode. {original_strategy} -> "
                        f"{final_strategy}."
                    ),
                    tags=["override", "qa_mode"],
                    summary_cards=[
                        {"label": "请求模式", "value": qa_mode},
                        {"label": "策略切换", "value": f"{original_strategy} -> {final_strategy}"},
                    ],
                )

        if plan.strategy == "conversational":
            if trace_enabled:
                yield self._build_trace_event(
                    request_id=request_id,
                    step="route",
                    title="进入对话生成模式",
                    detail="该问题无需知识库检索，正在直接生成回复。",
                    level=trace_level,
                    phase="routing",
                    status="completed",
                    node="router",
                    next_step="synthesize",
                    engine_detail="Routed to conversational path (no retrieval required).",
                    tags=["conversational"],
                    summary_cards=[{"label": "执行路径", "value": "conversational"}],
                )
            async for event in self._handle_conversational(query, chat_history, model):
                yield event
            return

        if plan.strategy == "simple_rag":
            if trace_enabled:
                yield self._build_trace_event(
                    request_id=request_id,
                    step="route",
                    title="进入标准检索模式",
                    detail="将执行单轮混合检索并直接生成答案。",
                    level=trace_level,
                    phase="routing",
                    status="completed",
                    node="router",
                    next_step="retrieve",
                    engine_detail="Routed to simple_rag path.",
                    tags=["simple_rag"],
                    summary_cards=[{"label": "执行路径", "value": "simple_rag"}],
                )
            async for event in self._simple_rag(
                plan,
                kb_ids,
                user_id,
                retrieval_cfg,
                chat_history,
                model,
                request_id=request_id,
                trace_enabled=trace_enabled,
                trace_level=trace_level,
            ):
                yield event
        else:
            async for event in self._multi_agent(
                plan,
                kb_ids,
                user_id,
                retrieval_cfg,
                chat_history,
                model,
                request_id=request_id,
                trace_enabled=trace_enabled,
                trace_level=trace_level,
            ):
                yield event

    async def _handle_conversational(self, query, chat_history, model) -> AsyncGenerator[ChatStreamEvent, None]:
        answer_parts = []
        async for token in self.synthesizer.synthesize(
            query=query,
            contexts=[],
            chat_history=chat_history,
            stream=True,
            model=model,
        ):
            answer_parts.append(token)
            yield ChatStreamEvent(event="answer", data={"token": token})
        yield ChatStreamEvent(event="done", data={"answer": "".join(answer_parts), "citations": []})

    async def _simple_rag(
        self,
        plan,
        kb_ids,
        user_id,
        retrieval_cfg,
        chat_history,
        model,
        *,
        request_id: str,
        trace_enabled: bool,
        trace_level: str,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        t0 = time.monotonic()
        result = await self.retriever.retrieve(
            query=plan.primary_query,
            user_id=user_id,
            kb_ids=kb_ids,
            config=retrieval_cfg,
        )
        yield ChatStreamEvent(
            event="retrieval",
            data={
                "count": len(result.chunks),
                "latency_ms": result.latency_ms,
                "chunks": [
                    {"chunk_id": c.chunk_id, "score": c.score, "doc_name": c.doc_name}
                    for c in result.chunks[:5]
                ],
            },
        )

        if trace_enabled:
            debug = result.debug or {}
            top_sources = self._top_sources_from_retrieved_chunks(result.chunks, limit=3)
            trace_metrics = {
                "latency_ms": result.latency_ms,
                "total_retrieved": result.total_retrieved,
                "dense_hits": debug.get("dense_count", 0),
                "sparse_hits": debug.get("sparse_count", 0),
                "bm25_hits": debug.get("bm25_count", 0),
                "fused_count": debug.get("fused_count", 0),
                "reranked_count": debug.get("reranked_count", 0),
                "after_permission_count": debug.get("after_permission_count", len(result.chunks)),
            }
            if trace_level == "pro" and top_sources:
                trace_metrics["top_sources"] = top_sources
            yield self._build_trace_event(
                request_id=request_id,
                step="retrieve",
                title="混合检索完成",
                detail="已完成多路召回、融合与权限过滤，正在生成答案。",
                metrics=trace_metrics,
                level=trace_level,
                phase="retrieval",
                status="completed",
                node="hybrid_retriever",
                next_step="synthesize",
                engine_detail=(
                    f"Dense={debug.get('dense_count', 0)}, Sparse={debug.get('sparse_count', 0)}, "
                    f"BM25={debug.get('bm25_count', 0)}, Fused={debug.get('fused_count', 0)}, "
                    f"Final={len(result.chunks)}."
                ),
                tags=["dense", "sparse", "bm25", "rerank"],
                summary_cards=[
                    {"label": "检索轮次", "value": 1},
                    {"label": "有效证据", "value": len(result.chunks)},
                    {"label": "检索耗时", "value": f"{result.latency_ms} ms"},
                ],
            )

        answer_parts = []
        async for token in self.synthesizer.synthesize(
            query=plan.resolved_query,
            contexts=result.chunks,
            chat_history=chat_history,
            stream=True,
            model=model,
        ):
            answer_parts.append(token)
            yield ChatStreamEvent(event="answer", data={"token": token})

        full_answer = "".join(answer_parts)
        citations = self.synthesizer.extract_citations(full_answer, result.chunks)
        yield ChatStreamEvent(event="citations", data={"citations": citations})
        if trace_enabled:
            yield self._build_trace_event(
                request_id=request_id,
                step="synthesize",
                title="答案生成完成",
                detail=f"回答已生成并完成证据绑定，共 {len(citations)} 条引用。",
                metrics={
                    "answer_length": len(full_answer),
                    "citation_count": len(citations),
                    "total_latency_ms": int((time.monotonic() - t0) * 1000),
                },
                level=trace_level,
                phase="synthesis",
                status="completed",
                node="response_synthesizer",
                next_step="done",
                engine_detail=(
                    f"Answer synthesized. answer_length={len(full_answer)}, "
                    f"citations={len(citations)}."
                ),
                tags=["generation", "citation"],
                summary_cards=[
                    {"label": "回答长度", "value": len(full_answer)},
                    {"label": "引用数量", "value": len(citations)},
                ],
            )
            yield self._build_trace_event(
                request_id=request_id,
                step="done",
                title="问答完成",
                detail="问答流程已完成，结果与引用已返回。",
                metrics={
                    "total_latency_ms": int((time.monotonic() - t0) * 1000),
                    "citation_count": len(citations),
                },
                level=trace_level,
                phase="finalize",
                status="completed",
                node="qa_engine",
                next_step=None,
                engine_detail=(
                    "Simple RAG finished. "
                    f"total_latency_ms={int((time.monotonic() - t0) * 1000)}, "
                    f"citation_count={len(citations)}."
                ),
                tags=["final"],
                summary_cards=[
                    {"label": "总耗时", "value": f"{int((time.monotonic() - t0) * 1000)} ms"},
                    {"label": "引用数量", "value": len(citations)},
                ],
            )
        yield ChatStreamEvent(
            event="done",
            data={
                "answer": full_answer,
                "citations": citations,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            },
        )

    async def _multi_agent(
        self,
        plan,
        kb_ids,
        user_id,
        retrieval_cfg,
        chat_history,
        model,
        *,
        request_id: str,
        trace_enabled: bool,
        trace_level: str,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        from app.core.retrieval import init_retriever

        t0 = time.monotonic()
        init_retriever(self.retriever)

        if trace_enabled:
            yield self._build_trace_event(
                request_id=request_id,
                step="deep_qa",
                title="进入深度研判模式",
                detail="正在并行执行子查询检索、证据汇聚与多步推理。",
                metrics={"sub_query_count": len(plan.sub_queries)},
                level=trace_level,
                phase="routing",
                status="completed",
                node="deep_qa_graph",
                next_step="retrieve",
                engine_detail=(
                    "Deep QA graph activated: retriever -> reasoner -> synthesizer -> critic."
                ),
                tags=["multi_agent", "deep_mode"],
                summary_cards=[
                    {"label": "执行路径", "value": "multi_agent"},
                    {"label": "子查询数", "value": len(plan.sub_queries)},
                ],
            )

        initial_state = {
            "query": plan.resolved_query,
            "chat_history": chat_history or [],
            "user_id": user_id,
            "kb_ids": kb_ids,
            "query_plan": {
                "primary_query": plan.primary_query,
                "hyde_text": plan.hyde_text,
                "intent": plan.intent.value if hasattr(plan.intent, "value") else plan.intent,
                "strategy": plan.strategy,
            },
            "sub_queries": plan.sub_queries,
            "retrieved_contexts": [],
            "retrieval_rounds": 0,
            "retrieval_traces": [],
            "sub_answers": [],
            "reasoning_notes": "",
            "final_answer": "",
            "citations": [],
            "stream_tokens": [],
            "routing_decision": "complex",
            "quality_check": None,
            "iteration_count": 0,
            "max_iterations": self.settings.qa.max_iterations,
            "should_continue": False,
            "retrieval_config": {
                "retrieval_mode": retrieval_cfg.retrieval_mode,
                "top_k": retrieval_cfg.top_k,
                "rerank": retrieval_cfg.rerank,
            },
            "model": model,
        }

        graph = self._get_graph()
        final_state = None
        last_quality: dict[str, Any] = {}

        async for step in graph.astream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            if node_name == "retriever":
                ctx_count = len(node_output.get("retrieved_contexts") or [])
                traces = node_output.get("retrieval_traces") or []
                yield ChatStreamEvent(event="retrieval", data={"node": node_name, "new_contexts": ctx_count})

                if trace_enabled:
                    dense_hits = 0
                    sparse_hits = 0
                    bm25_hits = 0
                    fused = 0
                    per_query_profiles = []
                    for trace in traces:
                        debug = trace.get("debug") or {}
                        dense_hits += int(debug.get("dense_count", 0))
                        sparse_hits += int(debug.get("sparse_count", 0))
                        bm25_hits += int(debug.get("bm25_count", 0))
                        fused += int(debug.get("fused_count", 0))
                        if trace_level == "pro":
                            per_query_profiles.append({
                                "query": trace.get("query_preview", ""),
                                "latency_ms": trace.get("latency_ms", 0),
                                "total_retrieved": trace.get("total_retrieved", 0),
                                "final_count": trace.get("final_count", 0),
                                "dense_hits": debug.get("dense_count", 0),
                                "sparse_hits": debug.get("sparse_count", 0),
                                "bm25_hits": debug.get("bm25_count", 0),
                                "fused_count": debug.get("fused_count", 0),
                            })

                    metrics = {
                        "query_rounds": len(traces),
                        "new_contexts": ctx_count,
                        "dense_hits": dense_hits,
                        "sparse_hits": sparse_hits,
                        "bm25_hits": bm25_hits,
                        "fused_count": fused,
                    }
                    top_sources = self._top_sources_from_state_contexts(
                        node_output.get("retrieved_contexts") or [], limit=3
                    )
                    if trace_level == "pro" and per_query_profiles:
                        metrics["query_profiles"] = per_query_profiles[:6]
                    if trace_level == "pro" and top_sources:
                        metrics["top_sources"] = top_sources

                    yield self._build_trace_event(
                        request_id=request_id,
                        step="retrieve",
                        title="多轮检索完成",
                        detail=(
                            f"已完成 {len(traces)} 轮检索，新增有效证据 {ctx_count} 条。"
                            if traces
                            else f"检索节点已执行，新增有效证据 {ctx_count} 条。"
                        ),
                        metrics=metrics,
                        level=trace_level,
                        phase="retrieval",
                        status="completed",
                        node="retriever",
                        next_step="reasoning",
                        engine_detail=(
                            f"Retrieval rounds={len(traces)}, new_contexts={ctx_count}, "
                            f"dense={dense_hits}, sparse={sparse_hits}, bm25={bm25_hits}."
                        ),
                        tags=["multi_query", "fusion", "permission_filter"],
                        summary_cards=[
                            {"label": "检索轮次", "value": len(traces)},
                            {"label": "有效证据", "value": ctx_count},
                            {"label": "Dense命中", "value": dense_hits},
                        ],
                    )

            elif node_name == "reasoner":
                sub_answers = node_output.get("sub_answers") or []
                sufficient_count = sum(1 for a in sub_answers if a.get("sufficient"))
                yield ChatStreamEvent(
                    event="reasoning",
                    data={"sub_answers": [{"query": a["query"], "sufficient": a["sufficient"]} for a in sub_answers]},
                )
                if trace_enabled:
                    yield self._build_trace_event(
                        request_id=request_id,
                        step="reasoning",
                        title="多步推理完成",
                        detail=(
                            f"已完成 {len(sub_answers)} 个子问题推理，"
                            f"其中 {sufficient_count} 个证据充分。"
                        ),
                        metrics={
                            "sub_answers": len(sub_answers),
                            "sufficient_answers": sufficient_count,
                        },
                        level=trace_level,
                        phase="reasoning",
                        status="completed",
                        node="reasoner",
                        next_step="synthesize",
                        engine_detail=(
                            f"Sub-question reasoning finished. total={len(sub_answers)}, "
                            f"sufficient={sufficient_count}."
                        ),
                        tags=["decomposition", "reasoning"],
                        summary_cards=[
                            {"label": "子问题数", "value": len(sub_answers)},
                            {"label": "充分回答", "value": sufficient_count},
                        ],
                    )

            elif node_name == "synthesizer":
                answer = node_output.get("final_answer", "")
                citations = node_output.get("citations", [])
                # Emit answer as single token (graph runs non-streaming internally)
                yield ChatStreamEvent(event="answer", data={"token": answer})
                yield ChatStreamEvent(event="citations", data={"citations": citations})
                final_state = node_output
                if trace_enabled:
                    yield self._build_trace_event(
                        request_id=request_id,
                        step="synthesize",
                        title="答案整合完成",
                        detail=f"已融合多智能体结果并生成回答，引用 {len(citations)} 条。",
                        metrics={
                            "answer_length": len(answer),
                            "citation_count": len(citations),
                        },
                        level=trace_level,
                        phase="synthesis",
                        status="completed",
                        node="synthesizer",
                        next_step="quality_gate",
                        engine_detail=(
                            f"Synthesizer completed. answer_length={len(answer)}, "
                            f"citation_count={len(citations)}."
                        ),
                        tags=["answer", "grounding"],
                        summary_cards=[
                            {"label": "回答长度", "value": len(answer)},
                            {"label": "引用数量", "value": len(citations)},
                        ],
                    )

            elif node_name == "critic" and trace_enabled:
                quality = node_output.get("quality_check") or {}
                if isinstance(quality, dict):
                    last_quality = quality
                    quality["quality_grade"] = self._quality_grade(quality)
                yield self._build_trace_event(
                    request_id=request_id,
                    step="quality_gate",
                    title="质量评估完成",
                    detail="已完成相关性、完整性和忠实度校验。",
                    metrics=quality if isinstance(quality, dict) else {},
                    level=trace_level,
                    phase="quality",
                    status="completed",
                    node="critic",
                    next_step="done",
                    engine_detail=(
                        f"Quality gate result: {quality}"
                        if isinstance(quality, dict)
                        else "Quality gate completed."
                    ),
                    tags=["evaluation", "quality_gate"],
                    summary_cards=[
                        {"label": "质量等级", "value": self._quality_grade(quality) if isinstance(quality, dict) else "unknown"},
                        {"label": "是否通过", "value": bool(quality.get("passed", True)) if isinstance(quality, dict) else True},
                    ],
                )

        answer = (final_state or {}).get("final_answer", "")
        citations = (final_state or {}).get("citations", [])
        if trace_enabled:
            total_latency_ms = int((time.monotonic() - t0) * 1000)
            yield self._build_trace_event(
                request_id=request_id,
                step="done",
                title="问答完成",
                detail="问答流程已完成，结果与引用已返回。",
                metrics={
                    "total_latency_ms": total_latency_ms,
                    "citation_count": len(citations),
                    "quality_passed": last_quality.get("passed") if isinstance(last_quality, dict) else None,
                    "quality_grade": self._quality_grade(last_quality),
                },
                level=trace_level,
                phase="finalize",
                status="completed",
                node="qa_engine",
                next_step=None,
                engine_detail=(
                    "Pipeline finished. "
                    f"total_latency_ms={total_latency_ms}, "
                    f"citation_count={len(citations)}."
                ),
                tags=["final"],
                summary_cards=[
                    {"label": "总耗时", "value": f"{total_latency_ms} ms"},
                    {"label": "引用数量", "value": len(citations)},
                    {"label": "质量等级", "value": self._quality_grade(last_quality)},
                ],
            )
        yield ChatStreamEvent(event="done", data={"answer": answer, "citations": citations})

    async def retrieve_only(
        self,
        query: str,
        kb_ids: list[str],
        user_id: str,
        config: Optional[dict] = None,
    ) -> RetrievalResult:
        cfg_dict = config or {}
        retrieval_cfg = RetrievalConfig(
            retrieval_mode=cfg_dict.get("retrieval_mode", "hybrid"),
            top_k=cfg_dict.get("top_k", 10),
            rerank=cfg_dict.get("rerank", True),
        )
        return await self.retriever.retrieve(
            query=query,
            user_id=user_id,
            kb_ids=kb_ids,
            config=retrieval_cfg,
        )

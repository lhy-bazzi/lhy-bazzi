"""QA Engine — unified entry point for intelligent question answering."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from app.services.retrieval.models import RetrievalConfig, RetrievalResult


@dataclass
class ChatStreamEvent:
    event: str  # retrieval | reasoning | answer | citations | done
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
        model = cfg.get("model")
        retrieval_cfg = RetrievalConfig(
            retrieval_mode=cfg.get("retrieval_mode", "hybrid"),
            top_k=cfg.get("top_k", 10),
            rerank=cfg.get("rerank", True),
        )

        # 1. Query understanding
        plan = await self.query_understanding.process(query, chat_history)
        logger.info("QA intent={} strategy={} query='{}'", plan.intent, plan.strategy, query[:60])

        if plan.strategy == "conversational":
            async for event in self._handle_conversational(query, chat_history, model):
                yield event
            return

        if plan.strategy == "simple_rag":
            async for event in self._simple_rag(plan, kb_ids, user_id, retrieval_cfg, chat_history, model):
                yield event
        else:
            async for event in self._multi_agent(plan, kb_ids, user_id, retrieval_cfg, chat_history, model, cfg):
                yield event

    async def _handle_conversational(self, query, chat_history, model) -> AsyncGenerator[ChatStreamEvent, None]:
        answer_parts = []
        async for token in self.synthesizer.synthesize(
            query=query, contexts=[], chat_history=chat_history, stream=True, model=model
        ):
            answer_parts.append(token)
            yield ChatStreamEvent(event="answer", data={"token": token})
        yield ChatStreamEvent(event="done", data={"answer": "".join(answer_parts), "citations": []})

    async def _simple_rag(
        self, plan, kb_ids, user_id, retrieval_cfg, chat_history, model
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        t0 = time.monotonic()
        result = await self.retriever.retrieve(
            query=plan.primary_query, user_id=user_id, kb_ids=kb_ids, config=retrieval_cfg
        )
        yield ChatStreamEvent(event="retrieval", data={
            "count": len(result.chunks),
            "latency_ms": result.latency_ms,
            "chunks": [{"chunk_id": c.chunk_id, "score": c.score, "doc_name": c.doc_name} for c in result.chunks[:5]],
        })

        answer_parts = []
        async for token in self.synthesizer.synthesize(
            query=plan.resolved_query, contexts=result.chunks,
            chat_history=chat_history, stream=True, model=model,
        ):
            answer_parts.append(token)
            yield ChatStreamEvent(event="answer", data={"token": token})

        full_answer = "".join(answer_parts)
        citations = self.synthesizer.extract_citations(full_answer, result.chunks)
        yield ChatStreamEvent(event="citations", data={"citations": citations})
        yield ChatStreamEvent(event="done", data={
            "answer": full_answer,
            "citations": citations,
            "latency_ms": int((time.monotonic() - t0) * 1000),
        })

    async def _multi_agent(
        self, plan, kb_ids, user_id, retrieval_cfg, chat_history, model, cfg
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        from app.core.retrieval import init_retriever
        init_retriever(self.retriever)

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

        async for step in graph.astream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            if node_name == "retriever":
                ctx_count = len(node_output.get("retrieved_contexts") or [])
                yield ChatStreamEvent(event="retrieval", data={"node": node_name, "new_contexts": ctx_count})

            elif node_name == "reasoner":
                sub_answers = node_output.get("sub_answers") or []
                yield ChatStreamEvent(event="reasoning", data={
                    "sub_answers": [{"query": a["query"], "sufficient": a["sufficient"]} for a in sub_answers]
                })

            elif node_name == "synthesizer":
                answer = node_output.get("final_answer", "")
                citations = node_output.get("citations", [])
                # Emit answer as single token (graph runs non-streaming internally)
                yield ChatStreamEvent(event="answer", data={"token": answer})
                yield ChatStreamEvent(event="citations", data={"citations": citations})
                final_state = node_output

        answer = (final_state or {}).get("final_answer", "")
        citations = (final_state or {}).get("citations", [])
        yield ChatStreamEvent(event="done", data={"answer": answer, "citations": citations})

    async def retrieve_only(
        self, query: str, kb_ids: list[str], user_id: str, config: Optional[dict] = None
    ) -> RetrievalResult:
        cfg_dict = config or {}
        retrieval_cfg = RetrievalConfig(
            retrieval_mode=cfg_dict.get("retrieval_mode", "hybrid"),
            top_k=cfg_dict.get("top_k", 10),
            rerank=cfg_dict.get("rerank", True),
        )
        return await self.retriever.retrieve(query=query, user_id=user_id, kb_ids=kb_ids, config=retrieval_cfg)

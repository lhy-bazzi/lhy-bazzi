"""Retrieval agent — executes hybrid retrieval for simple or multi-query plans."""

from __future__ import annotations

from loguru import logger

from app.agents.state import AgentState
from app.services.retrieval.models import RetrievalConfig


async def retrieval_agent(state: AgentState) -> dict:
    from app.services.retrieval.hybrid_retriever import HybridRetriever
    from app.core.retrieval import get_retriever

    retriever: HybridRetriever = get_retriever()
    plan = state.get("query_plan") or {}
    cfg_dict = state.get("retrieval_config") or {}
    cfg = RetrievalConfig(
        retrieval_mode=cfg_dict.get("retrieval_mode", "hybrid"),
        top_k=cfg_dict.get("top_k", 10),
        rerank=cfg_dict.get("rerank", True),
    )

    queries: list[str] = []
    primary = plan.get("primary_query") or state["query"]
    queries.append(primary)

    # HyDE text as additional query
    if plan.get("hyde_text"):
        queries.append(plan["hyde_text"])

    # Sub-queries for multi-agent path
    for sq in state.get("sub_queries") or []:
        if sq not in queries:
            queries.append(sq)

    seen: set[str] = {c["chunk_id"] for c in state.get("retrieved_contexts") or []}
    new_contexts: list[dict] = []

    for q in queries:
        try:
            result = await retriever.retrieve(
                query=q,
                user_id=state["user_id"],
                kb_ids=state["kb_ids"],
                config=cfg,
            )
            for chunk in result.chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    new_contexts.append({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "kb_id": chunk.kb_id,
                        "content": chunk.content,
                        "heading_chain": chunk.heading_chain,
                        "chunk_type": chunk.chunk_type,
                        "score": chunk.score,
                        "doc_name": chunk.doc_name,
                        "page": chunk.page,
                    })
        except Exception as exc:
            logger.warning("Retrieval failed for query '{}': {}", q[:60], exc)

    logger.debug("retrieval_agent: {} new contexts", len(new_contexts))
    return {
        "retrieved_contexts": new_contexts,
        "retrieval_rounds": (state.get("retrieval_rounds") or 0) + 1,
    }

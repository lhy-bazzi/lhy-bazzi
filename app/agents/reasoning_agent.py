"""Reasoning agent — answers each sub-question from retrieved contexts."""

from __future__ import annotations

from loguru import logger

from app.agents.state import AgentState

_SUB_ANSWER_PROMPT = """基于以下参考资料，回答子问题。如果资料不足，说明"资料不足"。

参考资料：
{context}

子问题：{sub_query}
回答："""


async def reasoning_agent(state: AgentState) -> dict:
    from app.core.llm_provider import get_llm

    llm = get_llm()
    sub_queries = state.get("sub_queries") or [state["query"]]
    contexts = state.get("retrieved_contexts") or []
    model = state.get("model")

    # Use top contexts (by score) for each sub-query
    top_contexts = sorted(contexts, key=lambda c: c.get("score", 0), reverse=True)[:8]
    context_text = "\n\n".join(
        f"[{i+1}] {c['content']}" for i, c in enumerate(top_contexts)
    )

    sub_answers: list[dict] = []
    for sq in sub_queries:
        try:
            answer = await llm.completion(
                messages=[{"role": "user", "content": _SUB_ANSWER_PROMPT.format(
                    context=context_text, sub_query=sq
                )}],
                temperature=0.1, max_tokens=1024, model=model,
            )
            sub_answers.append({"query": sq, "answer": answer.strip(), "sufficient": "资料不足" not in answer})
        except Exception as exc:
            logger.warning("Reasoning failed for sub-query '{}': {}", sq[:60], exc)
            sub_answers.append({"query": sq, "answer": "", "sufficient": False})

    return {"sub_answers": sub_answers}

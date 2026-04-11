"""Critic agent — quality self-check with optional retry signal."""

from __future__ import annotations

from loguru import logger

from app.agents.state import AgentState

_CRITIC_PROMPT = """评估以下问答的质量，从1-5分打分（5分最高）。
只返回JSON格式：{{"relevance": N, "completeness": N, "faithfulness": N, "passed": true/false}}
passed=true 当三项均>=3分时。

问题：{query}
答案：{answer}
参考资料摘要：{context_summary}

评分："""


async def critic_agent(state: AgentState) -> dict:
    from app.core.llm_provider import get_llm
    import json

    llm = get_llm()
    answer = state.get("final_answer", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    model = state.get("model")

    # Skip critic if max iterations reached
    if iteration >= max_iter or not answer:
        return {"quality_check": {"passed": True}, "should_continue": False, "iteration_count": iteration + 1}

    contexts = state.get("retrieved_contexts") or []
    context_summary = " | ".join(c["content"][:100] for c in contexts[:3])

    try:
        result = await llm.completion(
            messages=[{"role": "user", "content": _CRITIC_PROMPT.format(
                query=state["query"], answer=answer, context_summary=context_summary
            )}],
            temperature=0.0, max_tokens=128, model=model,
        )
        qc = json.loads(result.strip())
    except Exception as exc:
        logger.warning("Critic agent failed: ", exc)
        qc = {"passed": True}

    passed = qc.get("passed", True)
    return {
        "quality_check": qc,
        "should_continue": not passed,
        "iteration_count": iteration + 1,
    }

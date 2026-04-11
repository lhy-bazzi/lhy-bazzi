"""Synthesizer agent — merges sub-answers into a final streamed answer."""

from __future__ import annotations

from loguru import logger

from app.agents.state import AgentState

_SYNTHESIZE_PROMPT = """你是企业知识库智能助手。请基于以下子问题分析和参考资料，综合回答用户的原始问题。
- 整合各子问题的答案，给出完整连贯的回答
- 回答末尾用 [序号] 标注引用来源
- 如果信息不足，诚实说明

原始问题：{query}

子问题分析：
{sub_answers}

参考资料：
{context}

综合回答："""

_SIMPLE_PROMPT = """你是企业知识库智能助手。请基于以下参考资料回答问题。
- 如果资料中没有相关信息，说明"根据现有资料，暂无相关信息"
- 回答末尾用 [序号] 标注引用来源

参考资料：
{context}

问题：{query}
回答："""


async def synthesizer_agent(state: AgentState) -> dict:
    from app.core.llm_provider import get_llm
    import re

    llm = get_llm()
    contexts = state.get("retrieved_contexts") or []
    sub_answers = state.get("sub_answers") or []
    model = state.get("model")

    top_contexts = sorted(contexts, key=lambda c: c.get("score", 0), reverse=True)[:10]
    context_text = "\n\n".join(
        f"[{i+1}] 来源：{c.get('doc_name') or c['doc_id']}\n{c['content']}"
        for i, c in enumerate(top_contexts)
    )

    if sub_answers:
        sub_text = "\n".join(f"Q: {a['query']}\nA: {a['answer']}" for a in sub_answers)
        prompt = _SYNTHESIZE_PROMPT.format(
            query=state["query"], sub_answers=sub_text, context=context_text
        )
    else:
        prompt = _SIMPLE_PROMPT.format(query=state["query"], context=context_text)

    try:
        answer = await llm.completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2048, model=model,
        )
    except Exception as exc:
        logger.error("Synthesizer LLM call failed: {}", exc)
        answer = "抱歉，生成答案时发生错误，请稍后重试。"

    # Extract citations
    indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    citations = []
    for idx in sorted(indices):
        if 1 <= idx <= len(top_contexts):
            c = top_contexts[idx - 1]
            citations.append({
                "chunk_id": c["chunk_id"],
                "doc_name": c.get("doc_name") or c["doc_id"],
                "page": c.get("page"),
                "highlight": c["content"][:200],
            })

    return {
        "final_answer": answer.strip(),
        "citations": citations,
        "stream_tokens": [answer.strip()],
    }

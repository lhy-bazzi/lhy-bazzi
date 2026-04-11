"""Query understanding — intent classification, rewriting, decomposition, HyDE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.models.enums import IntentType

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_COREFERENCE_PROMPT = """你是一个对话理解助手。根据对话历史，将最新问题改写为独立完整的问题（消解代词和指代）。
如果问题已经独立完整，原样返回。只返回改写后的问题，不要解释。

对话历史：
{history}

最新问题：{query}
改写后的问题："""

_INTENT_PROMPT = """将以下问题分类为一种意图类型，只返回类型名称（大写）。

意图类型：
- FACTUAL: 简单事实查询
- ANALYTICAL: 需要分析推理
- COMPARATIVE: 对比类问题
- SUMMARY: 总结类问题
- MULTI_HOP: 需要多步推理
- CONVERSATIONAL: 闲聊或非知识查询

问题：{query}
意图："""

_REWRITE_PROMPT = """优化以下查询以提高检索效果：扩展缩写、补充隐含关键词、去除口语化表达。
只返回改写后的查询，不要解释。

原始查询：{query}
优化后的查询："""

_DECOMPOSE_PROMPT = """将以下复杂问题分解为2-4个可独立回答的子问题。
以JSON数组格式返回，例如：["子问题1", "子问题2"]

复杂问题：{query}
子问题列表："""

_HYDE_PROMPT = """请为以下问题生成一段假设性的参考文档片段（100-200字），
就像这个问题的答案会出现在企业知识库文档中一样。只返回文档片段内容。

问题：{query}
假设文档片段："""

_FAST_MODEL_TEMP = 0.0


@dataclass
class QueryPlan:
    original_query: str
    resolved_query: str
    rewritten_query: str
    intent: IntentType
    strategy: str  # simple_rag | enhanced_rag | multi_agent
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: Optional[str] = None
    primary_query: str = ""

    def __post_init__(self):
        if not self.primary_query:
            self.primary_query = self.hyde_text or self.rewritten_query


class QueryUnderstanding:
    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def process(self, query: str, chat_history: list[dict] | None = None) -> QueryPlan:
        resolved = await self.resolve_coreference(query, chat_history or [])
        intent = await self.classify_intent(resolved)
        rewritten = await self.rewrite_query(resolved)

        if intent == IntentType.CONVERSATIONAL:
            return QueryPlan(
                original_query=query, resolved_query=resolved,
                rewritten_query=rewritten, intent=intent,
                strategy="conversational", primary_query=rewritten,
            )

        if intent in (IntentType.MULTI_HOP, IntentType.ANALYTICAL, IntentType.COMPARATIVE):
            sub_queries = await self.decompose_query(rewritten)
            hyde = await self.generate_hyde(rewritten)
            return QueryPlan(
                original_query=query, resolved_query=resolved,
                rewritten_query=rewritten, intent=intent,
                strategy="multi_agent", sub_queries=sub_queries,
                hyde_text=hyde, primary_query=hyde or rewritten,
            )

        # FACTUAL / SUMMARY / default
        return QueryPlan(
            original_query=query, resolved_query=resolved,
            rewritten_query=rewritten, intent=intent,
            strategy="simple_rag", primary_query=rewritten,
        )

    async def resolve_coreference(self, query: str, chat_history: list[dict]) -> str:
        if not chat_history:
            return query
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in chat_history[-6:]
        )
        try:
            result = await self.llm.completion(
                messages=[{"role": "user", "content": _COREFERENCE_PROMPT.format(
                    history=history_text, query=query
                )}],
                temperature=_FAST_MODEL_TEMP, max_tokens=256,
            )
            return result.strip() or query
        except Exception as exc:
            logger.warning("Coreference resolution failed: {}", exc)
            return query

    async def classify_intent(self, query: str) -> IntentType:
        try:
            result = await self.llm.completion(
                messages=[{"role": "user", "content": _INTENT_PROMPT.format(query=query)}],
                temperature=_FAST_MODEL_TEMP, max_tokens=32,
            )
            label = result.strip().upper()
            return IntentType(label.lower()) if label.lower() in IntentType._value2member_map_ else IntentType.FACTUAL
        except Exception as exc:
            logger.warning("Intent classification failed: {}", exc)
            return IntentType.FACTUAL

    async def rewrite_query(self, query: str) -> str:
        try:
            result = await self.llm.completion(
                messages=[{"role": "user", "content": _REWRITE_PROMPT.format(query=query)}],
                temperature=_FAST_MODEL_TEMP, max_tokens=256,
            )
            return result.strip() or query
        except Exception as exc:
            logger.warning("Query rewrite failed: {}", exc)
            return query

    async def decompose_query(self, query: str) -> list[str]:
        try:
            result = await self.llm.completion(
                messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(query=query)}],
                temperature=_FAST_MODEL_TEMP, max_tokens=512,
            )
            sub = json.loads(result.strip())
            return sub if isinstance(sub, list) else [query]
        except Exception as exc:
            logger.warning("Query decomposition failed: {}", exc)
            return [query]

    async def generate_hyde(self, query: str) -> str:
        try:
            result = await self.llm.completion(
                messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
                temperature=0.3, max_tokens=512,
            )
            return result.strip()
        except Exception as exc:
            logger.warning("HyDE generation failed: {}", exc)
            return ""

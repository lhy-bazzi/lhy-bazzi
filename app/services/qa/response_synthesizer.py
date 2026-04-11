"""Response synthesizer — stream answer generation with citations."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Optional

from app.services.retrieval.models import RetrievedChunk

_SYSTEM_PROMPT = """你是企业知识库智能助手。请严格基于以下参考资料回答用户问题。
- 如果参考资料中没有相关信息，请明确说明"根据现有资料，暂无相关信息"
- 回答末尾用 [序号] 标注引用来源，例如 [1][3]
- 回答简洁准确，使用中文"""


class ResponseSynthesizer:
    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def synthesize(
        self,
        query: str,
        contexts: list[RetrievedChunk],
        chat_history: list[dict] | None = None,
        stream: bool = True,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if contexts:
            messages.append({"role": "system", "content": self._build_context_prompt(contexts)})

        # Truncated chat history (last 10 turns)
        for msg in (chat_history or [])[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        if stream:
            async for token in self.llm.stream_completion(messages=messages, model=model):
                yield token
        else:
            result = await self.llm.completion(messages=messages, model=model)
            yield result

    def _build_context_prompt(self, contexts: list[RetrievedChunk]) -> str:
        parts = ["参考资料：\n"]
        for i, c in enumerate(contexts, 1):
            source = c.doc_name or c.doc_id
            heading = f" | 章节：{c.heading_chain}" if c.heading_chain else ""
            parts.append(f"[{i}] 来源：{source}{heading}\n{c.content}\n")
        return "\n".join(parts)

    def extract_citations(
        self, answer: str, contexts: list[RetrievedChunk]
    ) -> list[dict]:
        indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        citations = []
        for idx in sorted(indices):
            if 1 <= idx <= len(contexts):
                c = contexts[idx - 1]
                citations.append({
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name or c.doc_id,
                    "page": c.page,
                    "highlight": c.content[:200],
                })
        return citations

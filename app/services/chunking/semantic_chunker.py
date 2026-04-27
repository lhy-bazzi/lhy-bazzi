"""Semantic chunker: sentence-aware split with token hard limits and overlap."""

from __future__ import annotations

import re

from app.models.chunk import ChunkConfig
from app.services.chunking.token_utils import count_tokens, split_text_by_token_window, tail_tokens_text


class SemanticChunker:
    """Split oversized text chunks with sentence boundaries when possible."""

    def chunk(self, text: str, config: ChunkConfig) -> list[str]:
        text = text.strip()
        if not text:
            return []

        max_tokens = max(1, config.max_chunk_size)
        overlap_tokens = max(0, min(config.chunk_overlap, max_tokens - 1))
        target_tokens = max(1, min(config.chunk_size, max_tokens))
        min_tokens = max(1, min(config.min_chunk_size, max_tokens))

        sentences = self._split_sentences(text)
        if not sentences:
            return split_text_by_token_window(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        results: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def flush_current() -> None:
            nonlocal current, current_tokens
            if not current:
                return

            chunk_text = "".join(current).strip()
            if chunk_text:
                results.append(chunk_text)
                overlap_text = tail_tokens_text(chunk_text, overlap_tokens)
                if overlap_text:
                    current = [overlap_text]
                    current_tokens = count_tokens(overlap_text)
                else:
                    current = []
                    current_tokens = 0
            else:
                current = []
                current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            # A single sentence can exceed token limit; force window split.
            if sent_tokens > max_tokens:
                flush_current()
                results.extend(
                    split_text_by_token_window(
                        sent,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
                current = []
                current_tokens = 0
                continue

            should_flush_hard = current_tokens > 0 and current_tokens + sent_tokens > max_tokens
            should_flush_soft = (
                current_tokens >= min_tokens and current_tokens + sent_tokens > target_tokens
            )
            if should_flush_hard or should_flush_soft:
                flush_current()

            current.append(sent)
            current_tokens += sent_tokens

        remainder = "".join(current).strip()
        if remainder:
            results.append(remainder)

        # Final safety pass: enforce hard max token limit strictly.
        normalized: list[str] = []
        for chunk in results:
            normalized.extend(
                split_text_by_token_window(
                    chunk,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )

        return normalized or [text]

    def _split_sentences(self, text: str) -> list[str]:
        """Split by common Chinese/English sentence-ending punctuation."""
        # Keep delimiter in the result so punctuation stays with sentence.
        parts = re.split(r"([。！？；!?])", text)
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(r"[。！？；!?]", parts[i + 1]):
                sent = parts[i] + parts[i + 1]
                i += 2
            else:
                sent = parts[i]
                i += 1
            sent = sent.strip()
            if sent:
                sentences.append(sent)
        return sentences

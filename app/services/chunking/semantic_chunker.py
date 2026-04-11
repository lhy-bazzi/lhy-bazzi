"""Semantic chunker — splits oversized text blocks at sentence boundaries."""

from __future__ import annotations

import re

from app.models.chunk import ChunkConfig


# Sentence-ending punctuation (Chinese + English)
_SENT_END = re.compile(r'(?<=[。！？；!?])|(?<=\.(?!\s*\d))(?=\s+[A-Z])')
# Abbreviations to avoid splitting on
_ABBREV = re.compile(r'\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.$', re.IGNORECASE)


class SemanticChunker:
    """对超长文本块做语义感知的精细切分。"""

    def chunk(self, text: str, config: ChunkConfig) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [text] if text.strip() else []

        results: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > config.chunk_size * 4 and current_len >= config.min_chunk_size * 2:
                chunk_text = "".join(current).strip()
                if chunk_text:
                    results.append(chunk_text)
                # Overlap: keep last overlap/2 chars worth of sentences
                overlap_target = config.chunk_overlap * 2
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap_target:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        remainder = "".join(current).strip()
        if remainder:
            results.append(remainder)

        return results or [text]

    def _split_sentences(self, text: str) -> list[str]:
        """中英文分句。"""
        # Split on Chinese punctuation or English sentence boundaries
        parts = re.split(r'([。！？；!?])', text)
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(r'[。！？；!?]', parts[i + 1]):
                sent = parts[i] + parts[i + 1]
                i += 2
            else:
                sent = parts[i]
                i += 1
            if sent.strip():
                sentences.append(sent)
        return sentences

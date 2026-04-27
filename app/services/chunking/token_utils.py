"""Token utility helpers for chunking."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Best-effort token count with a safe fallback."""
    if not text:
        return 0

    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass

    # Fallback heuristic for environments without tiktoken.
    return max(1, len(text) // 2)


def tail_tokens_text(text: str, token_count: int) -> str:
    """Return the last `token_count` tokens (decoded back to text)."""
    if not text or token_count <= 0:
        return ""

    enc = _get_encoder()
    if enc is not None:
        try:
            ids = enc.encode(text)
            if not ids:
                return ""
            return enc.decode(ids[-token_count:])
        except Exception:
            pass

    # Fallback: approximate by characters.
    approx_chars = max(1, token_count * 2)
    return text[-approx_chars:]


def split_text_by_token_window(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into fixed token windows with overlap."""
    if not text:
        return []
    if max_tokens <= 0:
        return [text]

    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    step = max(1, max_tokens - overlap_tokens)

    enc = _get_encoder()
    if enc is not None:
        try:
            ids = enc.encode(text)
            if len(ids) <= max_tokens:
                return [text]

            chunks: list[str] = []
            i = 0
            while i < len(ids):
                piece_ids = ids[i : i + max_tokens]
                if not piece_ids:
                    break
                piece = enc.decode(piece_ids).strip()
                if piece:
                    chunks.append(piece)
                if i + max_tokens >= len(ids):
                    break
                i += step
            return chunks
        except Exception:
            pass

    # Fallback by character windows.
    max_chars = max(1, max_tokens * 2)
    overlap_chars = max(0, overlap_tokens * 2)
    char_step = max(1, max_chars - overlap_chars)

    chunks: list[str] = []
    i = 0
    while i < len(text):
        piece = text[i : i + max_chars].strip()
        if piece:
            chunks.append(piece)
        if i + max_chars >= len(text):
            break
        i += char_step
    return chunks

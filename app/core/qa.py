"""QA engine singleton."""

from __future__ import annotations

_engine = None


def init_qa_engine(engine) -> None:
    global _engine
    _engine = engine


def get_qa_engine():
    if _engine is None:
        raise RuntimeError("QA engine not initialized.")
    return _engine

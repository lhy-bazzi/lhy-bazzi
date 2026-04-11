"""
Embedding model manager — DashScope online API (text-embedding-v3).

Replaces the local BGE-M3 model with DashScope's hosted embedding API.
No GPU / local model download required.
"""

from __future__ import annotations

import os

from loguru import logger


class EmbeddingModelManager:
    """
    Thin wrapper that holds DashScope config instead of a local model.
    The actual HTTP calls are made in EmbeddingService._batch_embed.
    """

    def __init__(self, settings):
        self.api_key: str = settings.dashscope.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url: str = settings.dashscope.base_url
        self.embedding_model: str = settings.dashscope.embedding_model
        self.dimension: int = settings.embedding.dimension

    async def load_embedding_model(self) -> None:
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY not set — embedding calls will fail.")
        else:
            logger.info("DashScope embedding model '{}' ready (online API).", self.embedding_model)

    async def load_rerank_model(self) -> None:
        # DashScope rerank is handled via LLM prompt; no model to load.
        logger.info("Rerank: using DashScope API (no local model).")

    def get_embed_model(self):
        """Returns self — DashScope config is the 'model'."""
        return self

    def get_rerank_model(self):
        return self


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: EmbeddingModelManager | None = None


def get_model_manager() -> EmbeddingModelManager:
    if _manager is None:
        raise RuntimeError("EmbeddingModelManager not initialized.")
    return _manager


async def init_model_manager(settings) -> EmbeddingModelManager:
    global _manager
    _manager = EmbeddingModelManager(settings)
    await _manager.load_embedding_model()
    await _manager.load_rerank_model()
    return _manager

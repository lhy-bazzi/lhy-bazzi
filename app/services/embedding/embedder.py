"""Embedding service — batch vectorization with Redis caching."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

from loguru import logger

from app.services.embedding.model_manager import EmbeddingModelManager

_DASHSCOPE_MAX_BATCH_SIZE = 10


@dataclass
class EmbeddingResult:
    dense_vector: list[float]
    sparse_vector: dict[int, float]


class EmbeddingService:
    """Batch text vectorization using BGE-M3 (dense + sparse)."""

    def __init__(self, model_manager: EmbeddingModelManager, redis_client, settings):
        self.model_manager = model_manager
        self.redis = redis_client
        self.batch_size: int = settings.embedding.batch_size
        self.api_batch_size: int = min(self.batch_size, _DASHSCOPE_MAX_BATCH_SIZE)
        if self.batch_size > _DASHSCOPE_MAX_BATCH_SIZE:
            logger.warning(
                "embedding.batch_size={} exceeds DashScope limit {}, fallback to {}",
                self.batch_size,
                _DASHSCOPE_MAX_BATCH_SIZE,
                self.api_batch_size,
            )
        self.cache_ttl: int = settings.embedding.cache_ttl
        self._cache_prefix = "emb:"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_texts(
        self, texts: list[str], use_cache: bool = True
    ) -> list[EmbeddingResult]:
        if not texts:
            return []

        keys = [self._cache_key(t) for t in texts]
        results: list[EmbeddingResult | None] = [None] * len(texts)

        # 1. Cache lookup
        if use_cache:
            cached = await self._mget(keys)
            for i, val in enumerate(cached):
                if val is not None:
                    results[i] = pickle.loads(val)  # noqa: S301

        # 2. Collect misses
        miss_indices = [i for i, r in enumerate(results) if r is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            embeddings = await self._batch_embed(miss_texts)

            # 3. Fill results + write cache
            pipe_data: list[tuple[str, bytes]] = []
            for idx, emb in zip(miss_indices, embeddings, strict=False):
                results[idx] = emb
                pipe_data.append((keys[idx], pickle.dumps(emb)))

            if use_cache:
                await self._mset(pipe_data, ttl=self.cache_ttl)

        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Single query embedding — no cache."""
        results = await self._batch_embed([query])
        return results[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _batch_embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Call DashScope text-embedding-v3 API in batches."""
        import httpx

        cfg = self.model_manager  # EmbeddingModelManager holds DashScope config
        results: list[EmbeddingResult] = []

        for start in range(0, len(texts), self.api_batch_size):
            batch = texts[start: start + self.api_batch_size]
            logger.debug("DashScope embed batch {}/{} ({} texts)",
                         start // self.api_batch_size + 1, -(-len(texts) // self.api_batch_size), len(batch))

            payload = {
                "model": cfg.embedding_model,
                "input": batch,
                "encoding_format": "float",
            }
            headers = {
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{cfg.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            # Sort by index to preserve order
            items = sorted(data["data"], key=lambda x: x["index"])
            for item in items:
                dense = item["embedding"]
                results.append(EmbeddingResult(
                    dense_vector=dense,
                    sparse_vector={},  # DashScope v3 is dense-only; sparse unused for now
                ))

        return results

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()
        return self._cache_prefix + h

    async def _mget(self, keys: list[str]) -> list[bytes | None]:
        try:
            # redis_client uses decode_responses=True, so we need raw bytes client
            # Fall back to individual gets if pipeline unavailable
            r = self.redis
            # Switch to bytes mode for pickle
            raw_client = r.client if hasattr(r, "client") else r
            vals = await raw_client.mget(*keys)
            return [v.encode("latin-1") if isinstance(v, str) else v for v in vals]
        except Exception as exc:
            logger.debug("Cache mget failed: {}", exc)
            return [None] * len(keys)

    async def _mset(self, pairs: list[tuple[str, bytes]], ttl: int) -> None:
        try:
            pipe = self.redis.pipeline()
            for key, val in pairs:
                # Store as latin-1 string to work with decode_responses=True client
                pipe.setex(key, ttl, val.decode("latin-1"))
            await pipe.execute()
        except Exception as exc:
            logger.debug("Cache mset failed: {}", exc)

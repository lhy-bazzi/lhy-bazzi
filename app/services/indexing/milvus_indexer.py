"""Milvus indexer — writes ChunkNode vectors to Milvus."""

from __future__ import annotations

from loguru import logger

from app.models.chunk import ChunkNode
from app.services.embedding.embedder import EmbeddingService

_BATCH_SIZE = 100
_CONTENT_MAX = 8192
_SPARSE_FALLBACK_TOP_K = 64


def _dense_to_sparse_fallback(dense: list[float], top_k: int = _SPARSE_FALLBACK_TOP_K) -> dict[int, float]:
    """Build a non-empty sparse vector from dense values.

    DashScope embedding API returns dense vectors only. Milvus sparse field
    rejects empty rows, so we fallback to top-k absolute dense dimensions.
    """
    if not dense:
        return {0: 1e-9}

    indexed = [(i, abs(float(v))) for i, v in enumerate(dense)]
    indexed.sort(key=lambda item: abs(item[1]), reverse=True)

    sparse = {i: v for i, v in indexed[:top_k] if v > 0.0}
    if sparse:
        return sparse

    # Defensive fallback: keep at least one non-zero entry.
    i, v = indexed[0]
    return {i: v if v != 0.0 else 1e-9}


def _resolve_sparse_vector(dense_vector: list[float], sparse_vector: dict[int, float]) -> dict[int, float]:
    if sparse_vector:
        return sparse_vector
    return _dense_to_sparse_fallback(dense_vector)


class MilvusIndexer:
    def __init__(self, milvus_client, embedding_service: EmbeddingService):
        self.milvus = milvus_client
        self.embedder = embedding_service

    async def index_chunks(self, chunks: list[ChunkNode]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed_texts(texts)

        total = 0
        for start in range(0, len(chunks), _BATCH_SIZE):
            batch_chunks = chunks[start: start + _BATCH_SIZE]
            batch_embs = embeddings[start: start + _BATCH_SIZE]

            records = [
                {
                    "id":            c.chunk_id,
                    "doc_id":        c.doc_id,
                    "kb_id":         c.kb_id,
                    "chunk_index":   c.chunk_index,
                    "content":       c.content[:_CONTENT_MAX],
                    "heading_chain": c.heading_chain,
                    "chunk_type":    c.chunk_type,
                    "dense_vector":  e.dense_vector,
                    "sparse_vector": _resolve_sparse_vector(e.dense_vector, e.sparse_vector),
                }
                for c, e in zip(batch_chunks, batch_embs, strict=False)
            ]

            from app.core import milvus_client as mc
            inserted = await mc.insert_chunks(records)
            total += inserted
            logger.info("Milvus: inserted batch {}/{} ({} chunks)",
                        start // _BATCH_SIZE + 1, -(-len(chunks) // _BATCH_SIZE), inserted)

        return total

    async def delete_by_doc_id(self, doc_id: str) -> int:
        from app.core import milvus_client as mc
        return await mc.delete_by_doc_id(doc_id)

    async def delete_by_kb_id(self, kb_id: str) -> int:
        from app.core import milvus_client as mc
        return await mc.delete_by_kb_id(kb_id)

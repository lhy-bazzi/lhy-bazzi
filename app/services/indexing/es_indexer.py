"""ES indexer — writes ChunkNode text to Elasticsearch for BM25 retrieval."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from app.models.chunk import ChunkNode

_BATCH_SIZE = 100


class ESIndexer:
    def __init__(self, es_client):
        self.es = es_client

    async def index_chunks(self, chunks: list[ChunkNode]) -> int:
        if not chunks:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        total = 0

        for start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[start: start + _BATCH_SIZE]
            records = [
                {
                    "id":            c.chunk_id,
                    "doc_id":        c.doc_id,
                    "kb_id":         c.kb_id,
                    "content":       c.content,
                    "heading_chain": c.heading_chain,
                    "chunk_type":    c.chunk_type,
                    "chunk_index":   c.chunk_index,
                    "created_at":    now,
                }
                for c in batch
            ]

            from app.core import es_client as ec
            inserted = await ec.index_chunks(records)
            total += inserted
            logger.info("ES: indexed batch {}/{} ({} chunks)",
                        start // _BATCH_SIZE + 1, -(-len(chunks) // _BATCH_SIZE), inserted)

        return total

    async def delete_by_doc_id(self, doc_id: str) -> int:
        from app.core import es_client as ec
        return await ec.delete_by_doc_id(doc_id)

    async def delete_by_kb_id(self, kb_id: str) -> int:
        from app.core import es_client as ec
        return await ec.delete_by_kb_id(kb_id)

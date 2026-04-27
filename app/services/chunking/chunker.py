"""Document chunker — orchestrates structural → semantic → table chunking."""

from __future__ import annotations

import uuid
from typing import Optional

from loguru import logger

from app.models.chunk import ChunkConfig, ChunkNode
from app.models.document import ParsedDocument
from app.services.chunking.semantic_chunker import SemanticChunker
from app.services.chunking.structural_chunker import StructuralChunker
from app.services.chunking.table_chunker import TableChunker
from app.services.chunking.token_utils import count_tokens


class DocumentChunker:
    """文档分块调度器 — 统一入口。"""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        self._structural = StructuralChunker()
        self._semantic = SemanticChunker()
        self._table = TableChunker()

    def chunk_document(
        self,
        document: ParsedDocument,
        kb_id: str,
        config: Optional[ChunkConfig] = None,
    ) -> list[ChunkNode]:
        cfg = config or self.config
        raw_chunks = self._structural.chunk(document, cfg)

        nodes: list[ChunkNode] = []

        for raw in raw_chunks:
            heading_chain_str = " > ".join(raw.heading_chain)

            if raw.chunk_type == "table" and raw.element is not None:
                table_nodes = self._table.chunk(
                    raw.element, document.doc_id, kb_id, heading_chain_str, cfg
                )
                nodes.extend(table_nodes)

            elif raw.chunk_type == "text" and raw.needs_semantic_split:
                sub_texts = self._semantic.chunk(raw.content, cfg)
                for sub in sub_texts:
                    nodes.append(ChunkNode(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=document.doc_id,
                        kb_id=kb_id,
                        chunk_index=0,
                        content=sub,
                        chunk_type="text",
                        heading_chain=heading_chain_str,
                    ))

            else:
                nodes.append(ChunkNode(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=document.doc_id,
                    kb_id=kb_id,
                    chunk_index=0,
                    content=raw.content,
                    chunk_type=raw.chunk_type,
                    heading_chain=heading_chain_str,
                ))

        # Assign index + token count
        for i, node in enumerate(nodes):
            node.chunk_index = i
            node.token_count = count_tokens(node.content)

        # Build parent-child relationships
        self._build_parent_chunks(nodes, cfg)

        # Log stats
        if nodes:
            sizes = [n.token_count for n in nodes]
            logger.info(
                "Chunking done: doc_id={} chunks={} avg_tokens={:.0f} min={} max={}",
                document.doc_id, len(nodes),
                sum(sizes) / len(sizes), min(sizes), max(sizes),
            )

        return nodes

    def _build_parent_chunks(self, chunks: list[ChunkNode], cfg: ChunkConfig) -> None:
        """Merge adjacent chunks into parent windows; set parent_content on children."""
        if not chunks:
            return

        i = 0
        while i < len(chunks):
            parent_parts: list[str] = []
            parent_len = 0
            j = i
            while j < len(chunks) and parent_len + chunks[j].token_count <= cfg.parent_chunk_size:
                parent_parts.append(chunks[j].content)
                parent_len += chunks[j].token_count
                j += 1
            if j == i:
                j = i + 1  # at least one chunk per parent

            parent_text = "\n\n".join(parent_parts)
            for k in range(i, j):
                chunks[k].parent_content = parent_text
            i = j

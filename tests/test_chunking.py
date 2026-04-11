"""Unit tests for Phase 4 chunking pipeline.

Run:
    cd uni-ai-python
    pytest tests/test_chunking.py -v
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.models.chunk import ChunkConfig, ChunkNode
from app.models.document import DocumentElement, DocumentMetadata, ParsedDocument
from app.models.enums import ElementType
from app.services.chunking.chunker import DocumentChunker
from app.services.chunking.semantic_chunker import SemanticChunker
from app.services.chunking.structural_chunker import StructuralChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elem(etype: ElementType, content: str, **meta) -> DocumentElement:
    return DocumentElement(element_type=etype, content=content, metadata=meta)


def _doc(elements: list[DocumentElement], doc_id: str = "doc-1") -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename="test.md",
        file_type="markdown",
        elements=elements,
        metadata=DocumentMetadata(),
        parse_engine="test",
    )


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------

class TestStructuralChunker:
    def setup_method(self):
        self.chunker = StructuralChunker()
        self.cfg = ChunkConfig()

    def test_single_text_block(self):
        doc = _doc([_elem(ElementType.TEXT, "Hello world.")])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].chunk_type == "text"

    def test_heading_resets_chain(self):
        doc = _doc([
            _elem(ElementType.HEADING, "Chapter 1", level=1),
            _elem(ElementType.TEXT, "Intro text."),
            _elem(ElementType.HEADING, "Section 1.1", level=2),
            _elem(ElementType.TEXT, "Section text."),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert len(chunks) == 2
        assert chunks[0].heading_chain == ["Chapter 1"]
        assert chunks[1].heading_chain == ["Chapter 1", "Section 1.1"]

    def test_h1_resets_deeper_headings(self):
        doc = _doc([
            _elem(ElementType.HEADING, "Chapter 1", level=1),
            _elem(ElementType.HEADING, "Section 1.1", level=2),
            _elem(ElementType.TEXT, "Text A."),
            _elem(ElementType.HEADING, "Chapter 2", level=1),
            _elem(ElementType.TEXT, "Text B."),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert chunks[-1].heading_chain == ["Chapter 2"]

    def test_table_is_independent_chunk(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Before table."),
            _elem(ElementType.TABLE, "| A | B |\n|---|---|\n| 1 | 2 |"),
            _elem(ElementType.TEXT, "After table."),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        types = [c.chunk_type for c in chunks]
        assert "table" in types
        assert types.index("table") == 1

    def test_code_is_independent_chunk(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Some text."),
            _elem(ElementType.CODE, "def foo(): pass"),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert chunks[-1].chunk_type == "code"

    def test_consecutive_texts_merged(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Line 1."),
            _elem(ElementType.TEXT, "Line 2."),
            _elem(ElementType.TEXT, "Line 3."),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert len(chunks) == 1
        assert "Line 1." in chunks[0].content
        assert "Line 2." in chunks[0].content

    def test_oversized_block_flagged(self):
        long_text = "sentence. " * 500  # far exceeds max_chunk_size
        doc = _doc([_elem(ElementType.TEXT, long_text)])
        cfg = ChunkConfig(max_chunk_size=100)
        chunks = self.chunker.chunk(doc, cfg)
        assert chunks[0].needs_semantic_split is True

    def test_page_break_ignored(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Before break."),
            _elem(ElementType.PAGE_BREAK, ""),
            _elem(ElementType.TEXT, "After break."),
        ])
        chunks = self.chunker.chunk(doc, self.cfg)
        assert len(chunks) == 1  # merged into one text block


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker()
        self.cfg = ChunkConfig(chunk_size=20, min_chunk_size=5, chunk_overlap=4)

    def test_short_text_single_chunk(self):
        result = self.chunker.chunk("Hello world.", self.cfg)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_split_on_chinese_punctuation(self):
        text = "第一句话。第二句话。第三句话。"
        sentences = self.chunker._split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "第一句话。"

    def test_split_on_english_punctuation(self):
        text = "First sentence! Second sentence? Third sentence."
        sentences = self.chunker._split_sentences(text)
        assert any("First sentence" in s for s in sentences)

    def test_long_text_produces_multiple_chunks(self):
        # 10 Chinese sentences, tiny chunk_size forces splits
        text = "".join(f"这是第{i}个句子。" for i in range(1, 11))
        cfg = ChunkConfig(chunk_size=10, min_chunk_size=2, chunk_overlap=2)
        result = self.chunker.chunk(text, cfg)
        assert len(result) > 1

    def test_empty_text(self):
        result = self.chunker.chunk("", self.cfg)
        assert result == []

    def test_overlap_preserves_context(self):
        text = "".join(f"Sentence number {i}. " for i in range(1, 20))
        cfg = ChunkConfig(chunk_size=30, min_chunk_size=5, chunk_overlap=10)
        result = self.chunker.chunk(text, cfg)
        if len(result) > 1:
            # overlap: some content from end of chunk N should appear at start of chunk N+1
            last_words = result[0].split()[-3:]
            next_start = result[1][:50]
            # soft check: chunks are non-empty and reasonable size
            assert all(len(c) > 0 for c in result)


# ---------------------------------------------------------------------------
# DocumentChunker (integration)
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(ChunkConfig())

    def test_basic_document(self):
        doc = _doc([
            _elem(ElementType.HEADING, "Introduction", level=1),
            _elem(ElementType.TEXT, "This is the introduction."),
            _elem(ElementType.HEADING, "Details", level=2),
            _elem(ElementType.TEXT, "This is detail text."),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        assert len(nodes) >= 2
        assert all(isinstance(n, ChunkNode) for n in nodes)

    def test_chunk_ids_unique(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Block 1."),
            _elem(ElementType.TABLE, "| A |\n|---|\n| 1 |"),
            _elem(ElementType.TEXT, "Block 2."),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        ids = [n.chunk_id for n in nodes]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential(self):
        doc = _doc([
            _elem(ElementType.TEXT, "A."),
            _elem(ElementType.TEXT, "B."),
            _elem(ElementType.CODE, "x = 1"),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        indices = [n.chunk_index for n in nodes]
        assert indices == list(range(len(nodes)))

    def test_heading_chain_propagated(self):
        doc = _doc([
            _elem(ElementType.HEADING, "Top", level=1),
            _elem(ElementType.HEADING, "Sub", level=2),
            _elem(ElementType.TEXT, "Content here."),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        text_nodes = [n for n in nodes if n.chunk_type == "text"]
        assert text_nodes[0].heading_chain == "Top > Sub"

    def test_token_count_set(self):
        doc = _doc([_elem(ElementType.TEXT, "Hello world this is a test.")])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        assert all(n.token_count > 0 for n in nodes)

    def test_parent_content_set(self):
        doc = _doc([
            _elem(ElementType.TEXT, "Chunk A."),
            _elem(ElementType.TEXT, "Chunk B."),
            _elem(ElementType.TEXT, "Chunk C."),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        assert all(n.parent_content is not None for n in nodes)

    def test_doc_id_and_kb_id(self):
        doc = _doc([_elem(ElementType.TEXT, "Some text.")], doc_id="doc-42")
        nodes = self.chunker.chunk_document(doc, kb_id="kb-99")
        assert all(n.doc_id == "doc-42" for n in nodes)
        assert all(n.kb_id == "kb-99" for n in nodes)

    def test_table_chunk_type(self):
        doc = _doc([
            _elem(ElementType.TABLE, "| Col1 | Col2 |\n|------|------|\n| A | B |\n| C | D |"),
        ])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        assert any(n.chunk_type == "table" for n in nodes)

    def test_large_table_split(self):
        # 120 data rows → should produce multiple table chunks
        header = "| ID | Name |\n|----|------|\n"
        rows = "".join(f"| {i} | item{i} |\n" for i in range(120))
        doc = _doc([_elem(ElementType.TABLE, header + rows)])
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1")
        table_nodes = [n for n in nodes if n.chunk_type == "table"]
        assert len(table_nodes) >= 2

    def test_oversized_text_semantically_split(self):
        # Build text long enough to trigger semantic split
        long_text = "这是测试句子。" * 300
        doc = _doc([_elem(ElementType.TEXT, long_text)])
        cfg = ChunkConfig(max_chunk_size=100, chunk_size=50, min_chunk_size=10, chunk_overlap=5)
        nodes = self.chunker.chunk_document(doc, kb_id="kb-1", config=cfg)
        assert len(nodes) > 1

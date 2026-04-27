#!/usr/bin/env python
"""
Local parse + chunk test script (no MinIO / Celery / database required).

Usage:
    python scripts/test_parse_chunk.py <file-path> [--kb-id <kb>] [--chunk-size <n>]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from collections import Counter

# Make sure `app` is importable when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def _parse_local(file_path: str, doc_id: str) -> "ParsedDocument":  # noqa: F821
    """Directly invoke the matching parser without MinIO."""
    from app.config import get_settings
    from app.services.parsing.docx_parser import DocxParser
    from app.services.parsing.excel_parser import ExcelParser
    from app.services.parsing.html_parser import HTMLParser
    from app.services.parsing.markdown_parser import MarkdownParser
    from app.services.parsing.pdf_parser import PDFParser
    from app.services.parsing.quality import QualityAssessor
    from app.services.parsing.text_parser import TextParser

    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        ".pdf": PDFParser(get_settings().parsing),
        ".docx": DocxParser(),
        ".xlsx": ExcelParser(),
        ".xls": ExcelParser(),
        ".csv": ExcelParser(),
        ".html": HTMLParser(),
        ".htm": HTMLParser(),
        ".md": MarkdownParser(),
        ".markdown": MarkdownParser(),
        ".txt": TextParser(),
    }

    parser = ext_map.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported extension: {ext}")

    result = await parser.parse(file_path, doc_id=doc_id)
    if result.quality_score == 0.0:
        result.quality_score = QualityAssessor().assess(result)
    return result


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _safe_console_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local parse + chunk tester")
    ap.add_argument("file", help="Local file to parse (pdf/docx/xlsx/md/txt/html)")
    ap.add_argument("--kb-id", default="test-kb", help="Knowledge-base ID (default: test-kb)")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)",
    )
    ap.add_argument("--max-chunk-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument(
        "--show-content",
        action="store_true",
        help="Print full chunk content instead of a short preview",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    from app.models.chunk import ChunkConfig
    from app.services.chunking.chunker import DocumentChunker

    doc_id = str(uuid.uuid4())[:8]
    cfg = ChunkConfig(
        chunk_size=args.chunk_size,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.overlap,
    )

    # ---- Parse ----
    _print_section("PARSING")
    print(f"  File   : {args.file}")
    print(f"  Doc ID : {doc_id}")
    parsed = asyncio.run(_parse_local(args.file, doc_id))
    print(f"  Engine : {parsed.parse_engine}")
    print(f"  Quality: {parsed.quality_score:.3f}")
    print(f"  Elements: {len(parsed.elements)}")

    type_counts = Counter(e.element_type.value for e in parsed.elements)
    for etype, cnt in sorted(type_counts.items()):
        print(f"    {etype:12s}: {cnt}")

    # ---- Chunk ----
    _print_section("CHUNKING")
    chunker = DocumentChunker(cfg)
    nodes = chunker.chunk_document(parsed, kb_id=args.kb_id)

    print(f"  Total chunks: {len(nodes)}")
    if nodes:
        sizes = [n.token_count for n in nodes]
        print(f"  Tokens  avg={sum(sizes) / len(sizes):.0f}  min={min(sizes)}  max={max(sizes)}")

    type_dist = Counter(n.chunk_type for n in nodes)
    for chunk_type, cnt in sorted(type_dist.items()):
        print(f"    {chunk_type:10s}: {cnt} chunks")

    # ---- Per-chunk detail ----
    _print_section("CHUNK DETAIL")
    for n in nodes:
        chain = n.heading_chain or "(no heading)"
        has_parent = "Y" if n.parent_content else "N"

        if args.show_content:
            preview = n.content
        else:
            preview = n.content[:8000].replace("\n", " ")
            if len(n.content) > 8000:
                preview += "..."

        print(
            f"[{n.chunk_index:03d}] {n.chunk_type:6s} "
            f"tok={n.token_count:4d} "
            f"parent={has_parent} "
            f"chain={chain!r}"
        )
        print(f"       {_safe_console_text(preview)}")
        print()


if __name__ == "__main__":
    main()

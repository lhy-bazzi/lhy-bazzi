"""Markdown parser — converts .md files into DocumentElements via markdown-it-py."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import chardet
from loguru import logger

from app.models.document import DocumentMetadata, ParsedDocument, DocumentElement
from app.models.enums import ElementType, FileType
from app.services.parsing.base import BaseParser
from app.utils.exceptions import ParseError


class MarkdownParser(BaseParser):
    """Markdown document parser using markdown-it-py token stream."""

    def supported_types(self) -> list[FileType]:
        return [FileType.MARKDOWN]

    async def parse(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)
        logger.info("Parsing Markdown '{}'", filename)

        try:
            elements = await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as exc:
            raise ParseError(f"Markdown parse failed: {exc}") from exc

        metadata = DocumentMetadata(file_size_bytes=self._file_size(file_path))
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="markdown",
            elements=elements,
            parse_engine="markdown-it-py",
            metadata=metadata,
        )

    def _parse_sync(self, file_path: str) -> list[DocumentElement]:
        text = self._read_file(file_path)
        return self._tokenize(text)

    @staticmethod
    def _read_file(file_path: str) -> str:
        with open(file_path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        return raw.decode(encoding, errors="replace")

    def _tokenize(self, text: str) -> list[DocumentElement]:
        from markdown_it import MarkdownIt

        md = MarkdownIt()
        tokens = md.parse(text)
        elements: list[DocumentElement] = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # Heading
            if tok.type == "heading_open":
                level = int(tok.tag[1]) if tok.tag and tok.tag[0] == "h" else 1
                # Next token should be inline content
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    elements.append(self._build_element(
                        ElementType.HEADING,
                        tokens[i + 1].content.strip(),
                        level=level,
                    ))
                    i += 3  # heading_open, inline, heading_close
                    continue

            # Paragraph
            if tok.type == "paragraph_open":
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    content = tokens[i + 1].content.strip()
                    if content:
                        # Check for image inside inline
                        if content.startswith("!["):
                            elements.append(self._build_element(ElementType.IMAGE, content))
                        else:
                            elements.append(self._build_element(ElementType.TEXT, content))
                    i += 3
                    continue

            # Code block / fence
            if tok.type == "fence" or tok.type == "code_block":
                lang = tok.info.strip() if tok.info else ""
                elements.append(self._build_element(
                    ElementType.CODE,
                    tok.content.rstrip("\n"),
                    language=lang,
                ))
                i += 1
                continue

            # Table (markdown-it doesn't enable tables by default; handle raw)
            # We check for "html_block" or fall through to next token
            if tok.type == "html_block":
                content = tok.content.strip()
                if "<table" in content.lower():
                    elements.append(self._build_element(ElementType.TABLE, content))
                else:
                    elements.append(self._build_element(ElementType.TEXT, content))
                i += 1
                continue

            # Bullet / ordered lists
            if tok.type in ("bullet_list_open", "ordered_list_open"):
                list_items, skip = self._collect_list_items(tokens, i)
                if list_items:
                    is_ordered = tok.type == "ordered_list_open"
                    lines = []
                    for idx, item in enumerate(list_items):
                        prefix = f"{idx + 1}. " if is_ordered else "- "
                        lines.append(prefix + item)
                    elements.append(self._build_element(
                        ElementType.LIST, "\n".join(lines)
                    ))
                i += skip
                continue

            i += 1

        # Handle raw markdown tables that markdown-it doesn't parse by default
        elements = self._merge_table_lines(elements)
        return elements

    @staticmethod
    def _collect_list_items(tokens, start: int) -> tuple[list[str], int]:
        """Collect list item texts until the matching list close."""
        items: list[str] = []
        depth = 0
        i = start
        while i < len(tokens):
            tok = tokens[i]
            if tok.type in ("bullet_list_open", "ordered_list_open"):
                depth += 1
            elif tok.type in ("bullet_list_close", "ordered_list_close"):
                depth -= 1
                if depth == 0:
                    return items, i - start + 1
            elif tok.type == "inline" and depth == 1:
                items.append(tok.content.strip())
            i += 1
        return items, i - start

    def _merge_table_lines(self, elements: list[DocumentElement]) -> list[DocumentElement]:
        """Merge consecutive TEXT elements that look like markdown table rows."""
        merged: list[DocumentElement] = []
        for el in elements:
            if el.element_type == ElementType.TEXT and el.content.strip().startswith("|"):
                if merged and merged[-1].element_type == ElementType.TABLE:
                    merged[-1].content += "\n" + el.content
                else:
                    merged.append(DocumentElement(
                        element_type=ElementType.TABLE,
                        content=el.content,
                        metadata=el.metadata,
                    ))
            else:
                merged.append(el)
        return merged

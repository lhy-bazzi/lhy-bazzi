"""Plain-text parser — encoding detection + paragraph splitting."""

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


# Patterns that hint a line might be a heading
_HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千\d]+[章节部篇]"),   # 中文章节
    re.compile(r"^\d+(\.\d+)*\s+\S"),                            # "1.2.3 Title"
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),                           # ALL-CAPS line
    re.compile(r"^={3,}$|^-{3,}$"),                               # separator
]


class TextParser(BaseParser):
    """Plain text (.txt) parser."""

    def supported_types(self) -> list[FileType]:
        return [FileType.TXT]

    async def parse(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)
        logger.info("Parsing TXT '{}'", filename)

        try:
            elements = await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as exc:
            raise ParseError(f"TXT parse failed: {exc}") from exc

        metadata = DocumentMetadata(file_size_bytes=self._file_size(file_path))
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="txt",
            elements=elements,
            parse_engine="plaintext",
            metadata=metadata,
        )

    def _parse_sync(self, file_path: str) -> list[DocumentElement]:
        text = self._read_file(file_path)
        # Split on blank lines → paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        elements: list[DocumentElement] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Try heading detection
            if self._looks_like_heading(para):
                elements.append(self._build_element(
                    ElementType.HEADING, para, level=1
                ))
            else:
                elements.append(self._build_element(ElementType.TEXT, para))

        return elements

    @staticmethod
    def _read_file(file_path: str) -> str:
        with open(file_path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        return raw.decode(encoding, errors="replace")

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        """Heuristic: short single-line text matching heading patterns."""
        if "\n" in text or len(text) > 100:
            return False
        return any(p.match(text) for p in _HEADING_PATTERNS)

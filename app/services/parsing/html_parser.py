"""HTML parser — extracts main content via readability, then structures via BS4."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from app.models.document import DocumentMetadata, ParsedDocument, DocumentElement
from app.models.enums import ElementType, FileType
from app.services.parsing.base import BaseParser
from app.utils.exceptions import ParseError


class HTMLParser(BaseParser):
    """HTML document parser using readability + BeautifulSoup."""

    def supported_types(self) -> list[FileType]:
        return [FileType.HTML]

    async def parse(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)
        logger.info("Parsing HTML '{}'", filename)

        try:
            elements, title = await asyncio.to_thread(self._parse_sync, file_path)
        except Exception as exc:
            raise ParseError(f"HTML parse failed: {exc}") from exc

        metadata = DocumentMetadata(
            title=title,
            file_size_bytes=self._file_size(file_path),
        )
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="html",
            elements=elements,
            parse_engine="readability+bs4",
            metadata=metadata,
        )

    def _parse_sync(self, file_path: str) -> tuple[list[DocumentElement], str | None]:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        # Try readability to extract main content
        clean_html = raw_html
        title: str | None = None
        try:
            from readability import Document as ReadabilityDoc
            rdoc = ReadabilityDoc(raw_html)
            clean_html = rdoc.summary()
            title = rdoc.short_title()
        except Exception:
            pass  # Fall back to parsing full HTML

        soup = BeautifulSoup(clean_html, "html.parser")
        if not title:
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None

        elements: list[DocumentElement] = []
        self._walk(soup, elements)
        return elements, title

    def _walk(self, element, elements: list[DocumentElement]) -> None:
        """Recursively walk the DOM tree and emit DocumentElements."""
        from bs4 import NavigableString, Tag

        if isinstance(element, NavigableString):
            return

        if not isinstance(element, Tag):
            return

        tag = element.name

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = element.get_text(strip=True)
            if text:
                level = int(tag[1])
                elements.append(self._build_element(
                    ElementType.HEADING, text, level=level
                ))
            return

        # Paragraph
        if tag == "p":
            text = element.get_text(strip=True)
            if text:
                elements.append(self._build_element(ElementType.TEXT, text))
            return

        # Code
        if tag in ("pre", "code"):
            text = element.get_text()
            if text.strip():
                elements.append(self._build_element(ElementType.CODE, text.strip()))
            return

        # Table
        if tag == "table":
            md = self._table_to_markdown(element)
            if md:
                elements.append(self._build_element(ElementType.TABLE, md))
            return

        # Lists
        if tag in ("ul", "ol"):
            items = []
            for li in element.find_all("li", recursive=False):
                items.append(li.get_text(strip=True))
            if items:
                prefix = "- " if tag == "ul" else ""
                text = "\n".join(
                    (f"{i + 1}. {t}" if tag == "ol" else f"- {t}")
                    for i, t in enumerate(items)
                )
                elements.append(self._build_element(ElementType.LIST, text))
            return

        # Image
        if tag == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            elements.append(self._build_element(
                ElementType.IMAGE, alt or "[image]", src=src
            ))
            return

        # Otherwise recurse into children
        for child in element.children:
            self._walk(child, elements)

    @staticmethod
    def _table_to_markdown(table_tag) -> str:
        rows = table_tag.find_all("tr")
        if not rows:
            return ""
        lines: list[str] = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True).replace("|", "\\|") for c in cells]
            lines.append("| " + " | ".join(cell_texts) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
        return "\n".join(lines)

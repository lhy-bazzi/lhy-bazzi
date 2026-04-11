"""Base class for all document parsers."""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.models.document import DocumentElement, DocumentMetadata, ParsedDocument
from app.models.enums import ElementType, FileType


class BaseParser(ABC):
    """Abstract base for every format-specific parser.

    Subclasses implement ``parse()`` and ``supported_types()``.
    Helper utilities for building elements are provided here.
    """

    @abstractmethod
    async def parse(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        """Parse *file_path* into a :class:`ParsedDocument` (UIR).

        Args:
            file_path: Local file path (already downloaded from MinIO).
            **kwargs: Extra hints — e.g. ``enable_ocr``, ``max_pages``.

        Returns:
            Unified intermediate representation.

        Raises:
            ParseError: when parsing fails irreversibly.
        """
        ...

    @abstractmethod
    def supported_types(self) -> list[FileType]:
        """File types this parser can handle."""
        ...

    # ------------------------------------------------------------------
    # Convenience helpers shared by all parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_element(
        element_type: ElementType,
        content: str,
        **metadata: Any,
    ) -> DocumentElement:
        """Shortcut for constructing a :class:`DocumentElement`."""
        return DocumentElement(
            element_type=element_type,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _build_document(
        doc_id: str,
        filename: str,
        file_type: str,
        elements: list[DocumentElement],
        parse_engine: str,
        metadata: DocumentMetadata | None = None,
    ) -> ParsedDocument:
        raw_text = "\n".join(
            e.content for e in elements if e.element_type != ElementType.IMAGE
        )
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            elements=elements,
            metadata=metadata or DocumentMetadata(),
            parse_engine=parse_engine,
            raw_text=raw_text,
        )

    @staticmethod
    def _file_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex

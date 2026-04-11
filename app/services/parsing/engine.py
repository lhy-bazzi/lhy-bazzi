"""Parse engine dispatcher — unified entry point for document parsing."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Optional

from loguru import logger

from app.config import get_settings
from app.core import minio_client
from app.models.document import ParsedDocument
from app.models.enums import FileType
from app.services.parsing.base import BaseParser
from app.services.parsing.docx_parser import DocxParser
from app.services.parsing.excel_parser import ExcelParser
from app.services.parsing.html_parser import HTMLParser
from app.services.parsing.markdown_parser import MarkdownParser
from app.services.parsing.pdf_parser import PDFParser
from app.services.parsing.quality import QualityAssessor
from app.services.parsing.text_parser import TextParser
from app.utils.exceptions import ParseError, UnsupportedFileTypeError

# Extension → FileType mapping
_EXT_MAP: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".xlsx": FileType.XLSX,
    ".xls": FileType.XLS,
    ".csv": FileType.CSV,
    ".txt": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
}

# MIME → FileType mapping (for python-magic detection)
_MIME_MAP: dict[str, FileType] = {
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.XLSX,
    "application/vnd.ms-excel": FileType.XLS,
    "text/csv": FileType.CSV,
    "text/plain": FileType.TXT,
    "text/markdown": FileType.MARKDOWN,
    "text/html": FileType.HTML,
}


class ParseEngine:
    """Document parsing dispatcher.

    Downloads the file from MinIO, detects its true type, selects the
    appropriate parser, and returns a ``ParsedDocument``.
    """

    def __init__(self) -> None:
        settings = get_settings().parsing
        self.parsers: dict[FileType, BaseParser] = {
            FileType.PDF: PDFParser(settings),
            FileType.DOCX: DocxParser(),
            FileType.XLSX: ExcelParser(),
            FileType.XLS: ExcelParser(),
            FileType.CSV: ExcelParser(),
            FileType.HTML: HTMLParser(),
            FileType.MARKDOWN: MarkdownParser(),
            FileType.TXT: TextParser(),
        }
        self.quality_assessor = QualityAssessor()

    async def parse_document(
        self,
        doc_id: str,
        kb_id: str,
        file_path: str,
        file_type: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Full parse pipeline.

        Args:
            doc_id: Unique document identifier.
            kb_id: Knowledge-base the document belongs to.
            file_path: MinIO object path (bucket/key).
            file_type: Declared file type (extension string).

        Returns:
            ParsedDocument (UIR) ready for chunking.
        """
        start = time.perf_counter()

        # 1. Download from MinIO to a local temp file
        local_path = await self._download(file_path)
        try:
            # 2. Detect real file type
            detected_type = self._detect_file_type(local_path, file_type)
            logger.info(
                "Parsing doc_id={} file='{}' detected_type={}",
                doc_id, os.path.basename(file_path), detected_type.value,
            )

            # 3. Select parser
            parser = self.parsers.get(detected_type)
            if parser is None:
                raise UnsupportedFileTypeError(detected_type.value)

            # 4. Parse
            result = await parser.parse(local_path, doc_id=doc_id, **kwargs)

            # 5. Ensure quality score is set
            if result.quality_score == 0.0:
                result.quality_score = self.quality_assessor.assess(result)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "Parse complete: doc_id={} engine={} elements={} quality={:.3f} time={:.0f}ms",
                doc_id, result.parse_engine, len(result.elements),
                result.quality_score, elapsed,
            )
            return result

        finally:
            # 6. Cleanup temp file
            self._cleanup(local_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _download(self, object_path: str) -> str:
        """Download MinIO object to a temporary local file."""
        ext = os.path.splitext(object_path)[1] or ""
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        await minio_client.download_file(object_path, tmp.name)
        return tmp.name

    def _detect_file_type(self, local_path: str, declared_type: str) -> FileType:
        """Detect real file type, cross-validate against declared type."""
        # Try python-magic
        try:
            import magic
            mime = magic.from_file(local_path, mime=True)
            if mime in _MIME_MAP:
                return _MIME_MAP[mime]
        except Exception:
            pass  # fall through to extension-based

        # Extension from declared type
        declared_lower = declared_type.lower().strip(".")
        for ext, ft in _EXT_MAP.items():
            if declared_lower == ext.lstrip(".") or declared_lower == ft.value:
                return ft

        # Extension from file path
        ext = os.path.splitext(local_path)[1].lower()
        if ext in _EXT_MAP:
            return _EXT_MAP[ext]

        raise UnsupportedFileTypeError(declared_type)

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: ParseEngine | None = None


def get_parse_engine() -> ParseEngine:
    global _engine
    if _engine is None:
        _engine = ParseEngine()
    return _engine

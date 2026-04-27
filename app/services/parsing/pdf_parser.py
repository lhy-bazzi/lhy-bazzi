"""PDF parser — primary engine MinerU, fallback Marker.

Handles text, scanned, and mixed PDFs via automatic classification.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from loguru import logger

from app.models.document import DocumentElement, DocumentMetadata, ParsedDocument
from app.models.enums import ElementType, FileType, PDFType
from app.services.parsing.base import BaseParser
from app.services.parsing.classifier import PDFClassifier
from app.services.parsing.quality import QualityAssessor
from app.utils.exceptions import ParseError


class PDFParser(BaseParser):
    """PDF document parser with multi-engine support."""

    def __init__(self, settings=None):
        from app.config import get_settings
        cfg = (settings or get_settings().parsing).pdf
        self.classifier = PDFClassifier()
        self.quality_assessor = QualityAssessor()
        self.primary_engine = cfg.primary_engine
        self.fallback_engine = cfg.fallback_engine
        self.quality_threshold = cfg.quality_threshold
        self.max_pages = cfg.max_pages
        self.enable_ocr = cfg.enable_ocr

    def supported_types(self) -> list[FileType]:
        return [FileType.PDF]

    async def parse(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        # Avoid passing duplicate `doc_id` through both explicit arg and **kwargs.
        parse_kwargs = dict(kwargs)
        doc_id = parse_kwargs.pop("doc_id", self._gen_id())
        filename = os.path.basename(file_path)

        # 1. Classify PDF type
        pdf_type = await self.classifier.classify(file_path)
        logger.info("Parsing PDF '{}' (type={}, engine={})", filename, pdf_type.value, self.primary_engine)

        # 2. Parse with primary engine
        result = await self._parse_with_engine(
            self.primary_engine, file_path, pdf_type, doc_id=doc_id, **parse_kwargs
        )
        result.quality_score = self.quality_assessor.assess(result)
        logger.info("Primary parse quality: {:.3f}", result.quality_score)

        # 3. Fallback if quality too low
        if result.quality_score < self.quality_threshold and self.fallback_engine:
            logger.info(
                "Quality {:.3f} < threshold {:.3f}, trying fallback engine '{}'",
                result.quality_score, self.quality_threshold, self.fallback_engine,
            )
            try:
                fallback = await self._parse_with_engine(
                    self.fallback_engine, file_path, pdf_type, doc_id=doc_id, **parse_kwargs
                )
                fallback_score = self.quality_assessor.assess(fallback)
                logger.info("Fallback parse quality: {:.3f}", fallback_score)
                if fallback_score > result.quality_score:
                    result = fallback
                    result.quality_score = fallback_score
            except Exception as exc:
                logger.warning("Fallback engine failed: {}", exc)

        return result

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    async def _parse_with_engine(
        self, engine: str, file_path: str, pdf_type: PDFType, **kwargs: Any
    ) -> ParsedDocument:
        if engine == "mineru":
            return await self._parse_mineru(file_path, pdf_type, **kwargs)
        elif engine == "marker":
            return await self._parse_marker(file_path, **kwargs)
        else:
            return await self._parse_pymupdf(file_path, **kwargs)

    # ------------------------------------------------------------------
    # MinerU (magic-pdf)
    # ------------------------------------------------------------------

    async def _parse_mineru(
        self, file_path: str, pdf_type: PDFType, **kwargs: Any
    ) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)

        try:
            elements = await asyncio.to_thread(
                self._mineru_sync, file_path, pdf_type
            )
        except ImportError:
            logger.warning("magic-pdf not installed, falling back to PyMuPDF")
            return await self._parse_pymupdf(file_path, **kwargs)
        except Exception as exc:
            raise ParseError(f"MinerU parse failed: {exc}") from exc

        metadata = await asyncio.to_thread(self._extract_pdf_metadata, file_path)
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            elements=elements,
            parse_engine="mineru",
            metadata=metadata,
        )

    def _mineru_sync(self, file_path: str, pdf_type: PDFType) -> list[DocumentElement]:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        tmp_dir = tempfile.mkdtemp(prefix="mineru_")
        try:
            # Read PDF bytes
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(file_path)

            # Select parse method based on PDF type
            if pdf_type == PDFType.SCANNED:
                method = "ocr"
            elif pdf_type == PDFType.TEXT:
                method = "txt"
            else:
                method = "auto"

            # Build dataset and run model analysis
            ds = PymuDocDataset(pdf_bytes)
            infer_result = ds.apply(doc_analyze, ocr=(method != "txt"))

            # Apply pipeline
            image_writer = FileBasedDataWriter(os.path.join(tmp_dir, "images"))
            if method == "ocr":
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            elif method == "txt":
                pipe_result = infer_result.pipe_txt_mode(image_writer)
            else:
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            # Extract content list
            content_list = pipe_result.get_content_list()
            md_content = pipe_result.get_markdown("")

            return self._mineru_content_to_elements(content_list, md_content)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _mineru_content_to_elements(
        self, content_list: list, md_content: str
    ) -> list[DocumentElement]:
        """Convert MinerU content list to DocumentElement list."""
        elements: list[DocumentElement] = []

        if content_list:
            for item in content_list:
                item_type = item.get("type", "text")
                text = item.get("text", "") or ""

                if item_type == "title":
                    elements.append(self._build_element(
                        ElementType.HEADING, text.strip(),
                        level=item.get("level", 1),
                        page=item.get("page_idx", 0),
                    ))
                elif item_type == "table":
                    table_content = item.get("html", "") or item.get("text", "")
                    elements.append(self._build_element(
                        ElementType.TABLE, table_content,
                        page=item.get("page_idx", 0),
                    ))
                elif item_type == "image":
                    elements.append(self._build_element(
                        ElementType.IMAGE, item.get("img_caption", ""),
                        img_path=item.get("img_path", ""),
                        page=item.get("page_idx", 0),
                    ))
                elif item_type == "equation":
                    elements.append(self._build_element(
                        ElementType.FORMULA, text,
                        page=item.get("page_idx", 0),
                    ))
                else:
                    if text.strip():
                        elements.append(self._build_element(
                            ElementType.TEXT, text.strip(),
                            page=item.get("page_idx", 0),
                        ))
        elif md_content:
            # Fallback: parse the markdown output
            elements = self._md_to_elements(md_content)

        return elements

    # ------------------------------------------------------------------
    # Marker
    # ------------------------------------------------------------------

    async def _parse_marker(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)

        try:
            md_text, metadata_dict = await asyncio.to_thread(
                self._marker_sync, file_path
            )
        except ImportError:
            logger.warning("marker not installed, falling back to PyMuPDF")
            return await self._parse_pymupdf(file_path, **kwargs)
        except Exception as exc:
            raise ParseError(f"Marker parse failed: {exc}") from exc

        elements = self._md_to_elements(md_text)
        metadata = await asyncio.to_thread(self._extract_pdf_metadata, file_path)
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            elements=elements,
            parse_engine="marker",
            metadata=metadata,
        )

    def _marker_sync(self, file_path: str) -> tuple[str, dict]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        rendered = converter(file_path)
        md_text = rendered.markdown
        meta = rendered.metadata or {}
        return md_text, meta

    # ------------------------------------------------------------------
    # PyMuPDF fallback (always available)
    # ------------------------------------------------------------------

    async def _parse_pymupdf(self, file_path: str, **kwargs: Any) -> ParsedDocument:
        doc_id = kwargs.get("doc_id", self._gen_id())
        filename = os.path.basename(file_path)
        elements = await asyncio.to_thread(self._pymupdf_sync, file_path)
        metadata = await asyncio.to_thread(self._extract_pdf_metadata, file_path)
        return self._build_document(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            elements=elements,
            parse_engine="pymupdf",
            metadata=metadata,
        )

    def _pymupdf_sync(self, file_path: str) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        doc = fitz.open(file_path)
        try:
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # text block
                        lines_text = []
                        for line in block.get("lines", []):
                            spans_text = "".join(s["text"] for s in line.get("spans", []))
                            lines_text.append(spans_text)
                        text = "\n".join(lines_text).strip()
                        if text:
                            # Heuristic heading detection: short + large font
                            max_size = max(
                                (s["size"] for l in block.get("lines", []) for s in l.get("spans", [])),
                                default=12,
                            )
                            if max_size >= 16 and len(text) < 200:
                                elements.append(self._build_element(
                                    ElementType.HEADING, text, page=page_num, font_size=max_size
                                ))
                            else:
                                elements.append(self._build_element(
                                    ElementType.TEXT, text, page=page_num
                                ))
                    elif block["type"] == 1:  # image block
                        elements.append(self._build_element(
                            ElementType.IMAGE, "[image]", page=page_num
                        ))
        finally:
            doc.close()
        return elements

    # ------------------------------------------------------------------
    # Markdown → elements helper
    # ------------------------------------------------------------------

    def _md_to_elements(self, md_text: str) -> list[DocumentElement]:
        """Rough markdown-to-elements conversion for MinerU/Marker output."""
        import re
        elements: list[DocumentElement] = []
        in_code = False
        code_buf: list[str] = []
        code_lang = ""

        for line in md_text.split("\n"):
            # Code fence
            if line.startswith("```"):
                if in_code:
                    elements.append(self._build_element(
                        ElementType.CODE, "\n".join(code_buf), language=code_lang
                    ))
                    code_buf.clear()
                    code_lang = ""
                    in_code = False
                else:
                    in_code = True
                    code_lang = line[3:].strip()
                continue
            if in_code:
                code_buf.append(line)
                continue

            # Heading
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                level = len(m.group(1))
                elements.append(self._build_element(
                    ElementType.HEADING, m.group(2).strip(), level=level
                ))
                continue

            # Table row (simplistic: collect consecutive | lines)
            if line.strip().startswith("|"):
                # Merge with previous table element if any
                if elements and elements[-1].element_type == ElementType.TABLE:
                    elements[-1].content += "\n" + line
                else:
                    elements.append(self._build_element(ElementType.TABLE, line))
                continue

            # Non-empty text
            stripped = line.strip()
            if stripped:
                elements.append(self._build_element(ElementType.TEXT, stripped))

        # Flush leftover code block
        if code_buf:
            elements.append(self._build_element(
                ElementType.CODE, "\n".join(code_buf), language=code_lang
            ))

        return elements

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_pdf_metadata(self, file_path: str) -> DocumentMetadata:
        doc = fitz.open(file_path)
        try:
            meta = doc.metadata or {}
            return DocumentMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                page_count=len(doc),
                file_size_bytes=os.path.getsize(file_path),
            )
        finally:
            doc.close()

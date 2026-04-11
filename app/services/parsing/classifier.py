"""PDF type classifier — determines whether a PDF is text, scanned, or mixed."""

from __future__ import annotations

import asyncio

import fitz  # PyMuPDF
from loguru import logger

from app.models.enums import PDFType


class PDFClassifier:
    """Sample the first *N* pages of a PDF to decide its type."""

    TEXT_THRESHOLD = 100     # avg chars/page above this → TEXT
    SCANNED_THRESHOLD = 10  # avg chars/page below this → SCANNED

    async def classify(self, file_path: str, sample_pages: int = 5) -> PDFType:
        return await asyncio.to_thread(self._classify_sync, file_path, sample_pages)

    def _classify_sync(self, file_path: str, sample_pages: int) -> PDFType:
        doc = fitz.open(file_path)
        try:
            total_pages = len(doc)
            pages_to_check = min(total_pages, sample_pages)

            total_chars = 0
            pages_with_images = 0

            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text("text")
                total_chars += len(text.strip())
                images = page.get_images(full=True)
                if images:
                    pages_with_images += 1

            avg_chars = total_chars / pages_to_check if pages_to_check else 0

            if avg_chars >= self.TEXT_THRESHOLD:
                pdf_type = PDFType.TEXT
            elif avg_chars <= self.SCANNED_THRESHOLD and pages_with_images > 0:
                pdf_type = PDFType.SCANNED
            else:
                pdf_type = PDFType.MIXED

            logger.debug(
                "PDF classified as {} (avg_chars={:.0f}, img_pages={}/{})",
                pdf_type.value, avg_chars, pages_with_images, pages_to_check,
            )
            return pdf_type
        finally:
            doc.close()

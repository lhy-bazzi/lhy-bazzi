"""Parse-quality assessor — scores a ParsedDocument from 0 to 1."""

from __future__ import annotations

import re
import unicodedata

from app.models.document import ParsedDocument
from app.models.enums import ElementType


class QualityAssessor:
    """Evaluate how well a document was parsed.

    Dimensions and weights:
        1. Text coverage   (0.30) — extracted text volume vs. expectation
        2. Structure        (0.25) — headings / table headers present
        3. Garbled ratio   (0.25) — proportion of unrecognisable characters
        4. Coherence        (0.20) — average paragraph length / fragmentation
    """

    def assess(self, doc: ParsedDocument) -> float:
        scores: list[tuple[float, float]] = [
            (self._text_coverage(doc), 0.30),
            (self._structure_score(doc), 0.25),
            (self._garbled_score(doc), 0.25),
            (self._coherence_score(doc), 0.20),
        ]
        total = sum(s * w for s, w in scores)
        return round(min(max(total, 0.0), 1.0), 4)

    # ----- 1. Text coverage -----

    def _text_coverage(self, doc: ParsedDocument) -> float:
        """Ratio of extracted text length to a rough expectation."""
        text_len = len(doc.raw_text)
        if text_len == 0:
            return 0.0
        # Rough heuristic: expect ~2 chars per byte for text-rich docs
        file_size = doc.metadata.file_size_bytes or 1
        expected = file_size * 0.3  # conservative expectation
        ratio = text_len / max(expected, 1)
        return min(ratio, 1.0)

    # ----- 2. Structure -----

    def _structure_score(self, doc: ParsedDocument) -> float:
        if not doc.elements:
            return 0.0
        has_heading = any(e.element_type == ElementType.HEADING for e in doc.elements)
        has_table = any(e.element_type == ElementType.TABLE for e in doc.elements)
        type_variety = len({e.element_type for e in doc.elements})

        score = 0.0
        if has_heading:
            score += 0.4
        if has_table:
            score += 0.2
        # Reward variety — more element types ⇒ richer structure
        score += min(type_variety / 5.0, 0.4)
        return min(score, 1.0)

    # ----- 3. Garbled text -----

    _GARBLED_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")

    def _garbled_score(self, doc: ParsedDocument) -> float:
        """Return 1.0 for clean text, 0.0 for heavily garbled."""
        text = doc.raw_text
        if not text:
            return 1.0  # nothing to garble
        garbled = len(self._GARBLED_RE.findall(text))
        # Also count characters in the Unicode "Other" category
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("C") and cat != "Cf":
                garbled += 1
        ratio = garbled / len(text)
        return max(1.0 - ratio * 10, 0.0)  # 10 % garbled → score 0

    # ----- 4. Coherence -----

    def _coherence_score(self, doc: ParsedDocument) -> float:
        text_elements = [
            e for e in doc.elements if e.element_type == ElementType.TEXT
        ]
        if not text_elements:
            return 0.5  # neutral when there's no text
        lengths = [len(e.content) for e in text_elements]
        avg = sum(lengths) / len(lengths)
        # Very short avg ⇒ fragmented
        if avg < 30:
            return 0.2
        if avg < 80:
            return 0.5
        if avg < 200:
            return 0.8
        return 1.0

"""Enumeration types for the UniAI AI service."""

from enum import Enum


class ElementType(str, Enum):
    """Document element types — parsing output granularity."""
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"
    LIST = "list"
    PAGE_BREAK = "page_break"


class ParseStatus(str, Enum):
    """Document parsing task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


class FileType(str, Enum):
    """Supported file types for parsing."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"


class IntentType(str, Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARY = "summary"
    CONVERSATIONAL = "conversational"
    MULTI_HOP = "multi_hop"


class RetrievalMode(str, Enum):
    """Retrieval strategy modes."""
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    FULLTEXT_ONLY = "fulltext_only"


class QAMode(str, Enum):
    """QA engine modes."""
    AUTO = "auto"
    SIMPLE = "simple"
    DEEP = "deep"


class TraceLevel(str, Enum):
    """Trace verbosity level for streaming process events."""
    BASIC = "basic"
    PRO = "pro"


class PDFType(str, Enum):
    """PDF sub-types for parsing strategy selection."""
    TEXT = "text"
    SCANNED = "scanned"
    MIXED = "mixed"

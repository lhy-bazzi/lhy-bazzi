"""Custom exception hierarchy for the UniAI AI service."""


class UniAIBaseError(Exception):
    """Base exception for all UniAI errors."""

    def __init__(self, message: str = "", code: int = 500):
        self.message = message
        self.code = code
        super().__init__(message)


# ---- Parsing ----

class ParseError(UniAIBaseError):
    """Document parsing failed."""

    def __init__(self, message: str = "Document parsing failed", code: int = 500):
        super().__init__(message, code)


class ParseQualityError(ParseError):
    """Parsing quality below threshold."""

    def __init__(self, message: str = "Parse quality below threshold", code: int = 500):
        super().__init__(message, code)


class UnsupportedFileTypeError(ParseError):
    """Unsupported file type."""

    def __init__(self, file_type: str = ""):
        super().__init__(f"Unsupported file type: {file_type}", code=400)


# ---- Retrieval ----

class RetrievalError(UniAIBaseError):
    """Retrieval pipeline failed."""

    def __init__(self, message: str = "Retrieval failed", code: int = 500):
        super().__init__(message, code)


# ---- LLM ----

class LLMError(UniAIBaseError):
    """LLM invocation failed."""

    def __init__(self, message: str = "LLM call failed", code: int = 502):
        super().__init__(message, code)


# ---- Auth ----

class PermissionDeniedError(UniAIBaseError):
    """User does not have access to the requested resource."""

    def __init__(self, message: str = "Permission denied", code: int = 403):
        super().__init__(message, code)


# ---- Config ----

class ConfigError(UniAIBaseError):
    """Configuration error."""

    def __init__(self, message: str = "Configuration error", code: int = 500):
        super().__init__(message, code)

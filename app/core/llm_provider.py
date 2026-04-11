"""LLM unified adapter layer backed by LiteLLM."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

import litellm
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import LLMConfig, get_settings
from app.utils.exceptions import LLMError

# Suppress LiteLLM's verbose startup logging
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Retryable exception types
# ---------------------------------------------------------------------------

class _RetryableError(Exception):
    """Wrapper for errors that should trigger a retry."""


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "timeout", "connection", "429", "503"))


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified LLM interface backed by LiteLLM.

    Reads model list and API keys from config/env at construction time.
    Supports both blocking completion and async streaming.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._default_model = cfg.default_model
        # Register API keys with LiteLLM via environment convention
        for entry in cfg.models:
            if entry.api_key and not entry.api_key.startswith("${"):
                # litellm picks up OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. automatically
                # but we can also set them explicitly if needed
                pass

    # ------------------------------------------------------------------
    # Completion (non-streaming)
    # ------------------------------------------------------------------

    async def completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the full completion text."""
        return await self._completion_with_retry(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature if temperature is not None else self._cfg.temperature,
            max_tokens=max_tokens or self._cfg.max_tokens,
            stream=False,
        )

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    async def stream_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text delta tokens as they arrive from the LLM."""
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._cfg.temperature
        max_tokens = max_tokens or self._cfg.max_tokens

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("LLM streaming error ({}): {}", model, exc)
            raise LLMError(f"LLM streaming failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal retry wrapper
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RetryableError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    async def _completion_with_retry(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if _is_retryable(exc):
                logger.warning("LLM retryable error ({}): {}", model, exc)
                raise _RetryableError(str(exc)) from exc
            logger.error("LLM non-retryable error ({}): {}", model, exc)
            raise LLMError(f"LLM call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None


def init_llm() -> None:
    """Initialise the LLM provider singleton. Call on app startup."""
    global _provider
    cfg = get_settings().llm
    ds = get_settings().dashscope

    # Register DashScope as an OpenAI-compatible provider via LiteLLM env vars
    import os
    if ds.api_key:
        os.environ.setdefault("OPENAI_API_KEY", ds.api_key)
        os.environ.setdefault("OPENAI_API_BASE", ds.base_url)
        # Also set DASHSCOPE_API_KEY for direct use
        os.environ.setdefault("DASHSCOPE_API_KEY", ds.api_key)

    _provider = LLMProvider(cfg)
    logger.info("LLM provider initialized. Default model: {}", cfg.default_model)


def get_llm() -> LLMProvider:
    if _provider is None:
        raise RuntimeError("LLM provider not initialized. Call init_llm() first.")
    return _provider

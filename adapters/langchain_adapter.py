"""
adapters/langchain_adapter.py
─────────────────────────────────────────────────────────────────────────────
LangChainTargetAdapter — Cloud LLM Adapter with Exponential Back-off Retry

Wraps any LangChain ``BaseChatModel`` (ChatOpenAI, ChatAnthropic, ChatGroq,
ChatGoogleGenerativeAI, etc.) into the ``BaseTargetAdapter`` interface.

Features
────────
• **Unified interface** — single ``invoke_full()`` works with any LangChain
  provider without changes to the agent layer.

• **Tenacity retry engine** — handles transient failures (rate limits, 5xx
  responses, network timeouts) with exponential back-off + jitter.  Permanent
  errors (auth failures, context length exceeded) are re-raised immediately
  without wasting retry budget.

• **Precise error mapping** — translates provider-specific HTTP status codes
  and exception types into the adapter's canonical error hierarchy so the
  graph's error handlers don't need provider-specific logic.

• **Usage metadata extraction** — reads token counts from the LangChain
  response metadata (supported by OpenAI, Anthropic, Groq) and falls back to
  the offline word-count estimator when the provider doesn't expose them.

• **Content filter detection** — detects ``finish_reason = "content_filter"``
  (OpenAI) and ``stop_reason = "max_tokens"`` (Anthropic) and surfaces them
  in the ``AdapterResponse`` for the Prometheus Judge to use in scoring.

Supported Providers (tested)
──────────────────────────────
  • OpenAI / Azure OpenAI  — ChatOpenAI
  • Anthropic Claude       — ChatAnthropic
  • Groq                   — ChatGroq
  • Google Gemini          — ChatGoogleGenerativeAI
  • Ollama (via LangChain) — ChatOllama
  • Any custom BaseChatModel

References
──────────
- Section 2.3 (Target Adapter Pattern), Original Project Document
- tenacity docs: https://tenacity.readthedocs.io/
"""

from __future__ import annotations
from core.utils import extract_text

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from adapters.base_adapter import (
    AdapterAuthError,
    AdapterContextLengthError,
    AdapterError,
    AdapterRateLimitError,
    AdapterResponse,
    AdapterTimeoutError,
    BaseTargetAdapter,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CLASSIFIER
# Maps raw provider exceptions to our canonical adapter error hierarchy.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_error(exc: Exception) -> AdapterError:
    """Translate a provider-specific exception into an ``AdapterError`` subtype.

    Classification is based on:
      1. HTTP status code embedded in the exception message or ``status_code``
         attribute (set by most LangChain providers via ``httpx``).
      2. Exception class name matching (fallback for providers that don't expose
         HTTP codes directly).
      3. Generic ``AdapterError`` for anything unrecognised.

    Parameters
    ──────────
    exc : Exception
        The raw exception from the LangChain model call.

    Returns
    ───────
    AdapterError
        The appropriate subtype, preserving the original message.
    """
    msg    = str(exc).lower()
    cls    = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)

    # Rate limit / quota
    if status == 429 or "rate limit" in msg or "ratelimit" in msg or "quota" in msg:
        retry_after: float | None = None
        # Some providers embed retry-after in the response headers via the exception
        headers = getattr(exc, "response", None) and getattr(exc.response, "headers", {})
        if headers:
            ra = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
            try:
                retry_after = float(ra) if ra else None
            except (ValueError, TypeError):
                pass
        return AdapterRateLimitError(str(exc), retry_after=retry_after)

    # Auth
    if status in (401, 403) or "auth" in msg or "api key" in msg or "invalid_api_key" in msg:
        return AdapterAuthError(str(exc))

    # Context length
    if (status == 400 and ("context" in msg or "token" in msg or "length" in msg)) \
            or "context_length_exceeded" in msg \
            or "maximum context" in msg \
            or "too many tokens" in msg:
        return AdapterContextLengthError(str(exc))

    # Timeout
    if "timeout" in msg or "timed out" in msg or "readtimeout" in cls or "connecttimeout" in cls:
        return AdapterTimeoutError(str(exc))

    # Server errors (5xx) — treat as transient, map to base AdapterError (retryable)
    if status and 500 <= status < 600:
        return AdapterError(f"Server error {status}: {exc}")

    # Fallback
    return AdapterError(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# USAGE METADATA EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_usage(response: Any, messages: list[BaseMessage]) -> tuple[int, int]:
    """Extract prompt and completion token counts from a LangChain response.

    Tries multiple known metadata key patterns before falling back to a
    word-count heuristic estimate.

    Parameters
    ──────────
    response :
        The raw response object from ``llm.invoke()``.
    messages :
        The input messages (used for the fallback estimator).

    Returns
    ───────
    tuple[int, int]
        (prompt_tokens, completion_tokens)
    """
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("usage", meta.get("token_usage", meta.get("usage_metadata", {}))) or {}

    # OpenAI / Groq key names
    prompt_tokens     = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

    # Anthropic embeds usage differently
    if not prompt_tokens:
        prompt_tokens = meta.get("input_token_count", 0)
    if not completion_tokens:
        completion_tokens = meta.get("output_token_count", 0)

    # Fallback: word-count heuristic
    if not prompt_tokens:
        prompt_tokens = sum(
            len((extract_text(m.content)).split())
            for m in messages
        )
    if not completion_tokens:
        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            parts = []
            for p in response.content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(p["text"])
                else:
                    parts.append(str(p))
            content = "".join(parts)
        else:
            content = str(response.content)
        completion_tokens = len(content.split())

    return int(prompt_tokens), int(completion_tokens)


def _extract_finish_reason(response: Any) -> str:
    """Extract the stop/finish reason from provider response metadata."""
    meta = getattr(response, "response_metadata", {}) or {}
    return (
        meta.get("finish_reason")
        or meta.get("stop_reason")
        or meta.get("finish_details", {}).get("type", "stop")
        or "stop"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TARGET ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class LangChainTargetAdapter(BaseTargetAdapter):
    """Target adapter wrapping any LangChain ``BaseChatModel``.

    Parameters
    ──────────
    model : BaseChatModel
        An instantiated LangChain chat model.  The adapter takes ownership
        but does NOT manage the model's lifecycle.

    timeout : float
        Per-request timeout in seconds.  Default: 30.0.

    max_retries : int
        Maximum retry attempts for transient failures (rate limits, timeouts,
        5xx errors).  Auth errors and context-length errors are NOT retried.
        Default: 3.

    initial_wait : float
        Starting back-off wait in seconds.  Default: 1.0.

    max_wait : float
        Maximum back-off ceiling in seconds.  Default: 60.0.

    jitter : float
        Maximum random jitter added to each wait window (prevents thundering
        herd on shared-quota deployments).  Default: 2.0.

    Example — OpenAI
    ─────────────────
    ::

        from langchain_openai import ChatOpenAI
        from adapters.langchain_adapter import LangChainTargetAdapter

        adapter = LangChainTargetAdapter(
            model       = ChatOpenAI(model="gpt-4o", temperature=0.7),
            max_retries = 5,
        )
        response = adapter.invoke([HumanMessage(content="Hello!")])

    Example — Groq
    ──────────────
    ::

        from langchain_groq import ChatGroq
        adapter = LangChainTargetAdapter(
            model = ChatGroq(model="llama-3-70b-8192", temperature=0.9),
        )
    """

    def __init__(
        self,
        model:         BaseChatModel,
        timeout:       float = 30.0,
        max_retries:   int   = 3,
        initial_wait:  float = 1.0,
        max_wait:      float = 60.0,
        jitter:        float = 2.0,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self._model        = model
        self._initial_wait = initial_wait
        self._max_wait     = max_wait
        self._jitter       = jitter
        self._call_count   = 0

        # Build the tenacity retry decorator bound to this instance's config.
        # We do this at init time so the parameters are captured in the closure.
        self._retry_decorator = retry(
            retry            = retry_if_exception_type((AdapterRateLimitError, AdapterTimeoutError, AdapterError)),
            stop             = stop_after_attempt(self.max_retries + 1),
            wait             = wait_exponential_jitter(
                                   initial = self._initial_wait,
                                   max     = self._max_wait,
                                   jitter  = self._jitter,
                               ),
            before_sleep     = before_sleep_log(logger, logging.WARNING),
            reraise          = True,
        )

    # ── Core invocation (single attempt, no retry) ────────────────────────

    def _single_invoke(self, messages: list[BaseMessage]) -> AdapterResponse:
        """Execute one API call.  Raises an ``AdapterError`` subtype on failure."""
        t_start = time.monotonic()
        try:
            response   = self._model.invoke(messages)
            latency_ms = (time.monotonic() - t_start) * 1000

            if isinstance(response.content, str):
                content = response.content
            elif isinstance(response.content, list):
                parts = []
                for p in response.content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                    else:
                        parts.append(str(p))
                content = "".join(parts)
            else:
                content = str(response.content)
            prompt_tokens, completion_tokens = _extract_usage(response, messages)
            finish_reason = _extract_finish_reason(response)

            logger.debug(
                "[Adapter] %s  tokens=%d+%d  latency=%.0fms  finish=%s",
                self.get_model_id(), prompt_tokens, completion_tokens,
                latency_ms, finish_reason,
            )

            return AdapterResponse(
                content           = content,
                model_id          = self.get_model_id(),
                prompt_tokens     = prompt_tokens,
                completion_tokens = completion_tokens,
                latency_ms        = latency_ms,
                finish_reason     = finish_reason,
                raw_response      = response,
            )

        except (AdapterError,):   # already classified — re-raise as-is
            raise
        except Exception as exc:
            classified = _classify_error(exc)
            # Auth and context-length errors should not be retried
            if isinstance(classified, (AdapterAuthError, AdapterContextLengthError)):
                logger.error("[Adapter] Non-retryable error: %s", classified)
                raise classified from exc
            raise classified from exc

    # ── Public interface ──────────────────────────────────────────────────

    def invoke_full(self, messages: list[BaseMessage]) -> AdapterResponse:
        """Invoke the target with automatic exponential back-off retry.

        Retryable errors: ``AdapterRateLimitError``, ``AdapterTimeoutError``,
        generic ``AdapterError`` (server 5xx).

        Non-retryable errors (re-raised immediately):
        ``AdapterAuthError``, ``AdapterContextLengthError``.

        Parameters
        ──────────
        messages : list[BaseMessage]
            Full conversation history including the latest user turn.

        Returns
        ───────
        AdapterResponse
            Structured response with content and metadata.

        Raises
        ──────
        AdapterAuthError
            Bad API key — fix the ``.env`` and restart.
        AdapterContextLengthError
            Prompt is too long — the STM should have compressed it first.
        AdapterRateLimitError
            Rate limit persisted beyond all retry attempts.
        AdapterError
            Any other unrecoverable error after exhausting retries.
        """
        if not messages:
            raise AdapterError("invoke_full called with empty message list.")

        self._call_count += 1
        logger.debug(
            "[Adapter] Call #%d to %s  messages=%d",
            self._call_count, self.get_model_id(), len(messages),
        )

        # Honour per-request timeout if the model supports bind
        model = self._model
        if hasattr(model, "bind") and self.timeout:
            try:
                model = model.bind(timeout=self.timeout)
            except Exception:   # noqa: BLE001
                pass   # Not all models support timeout binding — safe to ignore

        # Wrap the single invocation in the tenacity retry loop.
        # We can't use the @retry decorator directly because we need the
        # per-instance configuration.  Manually applying the decorator here.
        # Only retry genuinely transient errors; never retry auth / context errors.
        def _is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, (AdapterAuthError, AdapterContextLengthError)):
                return False
            return isinstance(exc, (AdapterRateLimitError, AdapterTimeoutError, AdapterError))

        from tenacity import retry_if_exception as _rife

        @retry(
            retry        = _rife(_is_retryable),
            stop         = stop_after_attempt(self.max_retries + 1),
            wait         = wait_exponential_jitter(
                               initial = self._initial_wait,
                               max     = self._max_wait,
                               jitter  = self._jitter,
                           ),
            before_sleep = before_sleep_log(logger, logging.WARNING),
            reraise      = True,
        )
        def _retryable_invoke() -> AdapterResponse:
            # Use the potentially timeout-bound model
            orig_model, self._model = self._model, model
            try:
                return self._single_invoke(messages)
            finally:
                self._model = orig_model   # restore original reference

        try:
            return _retryable_invoke()
        except RetryError as exc:
            last = exc.last_attempt.exception()
            logger.error(
                "[Adapter] Exhausted %d retries for %s. Last error: %s",
                self.max_retries, self.get_model_id(), last,
            )
            raise last from exc

    def get_model_id(self) -> str:
        """Return the model's canonical identifier.

        Introspects the LangChain model object in order of priority:
          1. ``model_name`` attribute (ChatOpenAI, ChatGroq, ChatAnthropic)
          2. ``model`` attribute (ChatOllama)
          3. Class name (last-resort fallback)
        """
        return (
            getattr(self._model, "model_name", None)
            or getattr(self._model, "model", None)
            or type(self._model).__name__
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """Detect capabilities from the wrapped model's class name."""
        cls_name = type(self._model).__name__.lower()
        return {
            "multimodal":       "vision" in cls_name or "gpt-4o" in self.get_model_id().lower(),
            "streaming":        True,
            "system_prompt":    True,
            "function_calling": "openai" in cls_name or "groq" in cls_name,
            "json_mode":        "openai" in cls_name,
        }

    def __repr__(self) -> str:
        return (
            f"LangChainTargetAdapter("
            f"model={self.get_model_id()!r}, "
            f"timeout={self.timeout}s, "
            f"max_retries={self.max_retries})"
        )

"""
adapters/base_adapter.py
─────────────────────────────────────────────────────────────────────────────
BaseTargetAdapter — Vendor-Agnostic Target Interface

Architectural Context (Section 2.3, Original Project Doc)
──────────────────────────────────────────────────────────
PromptEvo enforces strict vendor decoupling via the Adapter Design Pattern.
The orchestrator (core/graph.py) and all agent nodes NEVER communicate with
an LLM API directly.  Instead, they hold a reference to a ``BaseTargetAdapter``
and call its single unified method:

    ``invoke(messages: list[BaseMessage]) -> str``

This indirection means the entire offensive / evaluation pipeline is
completely portable across:
  • Cloud APIs    — OpenAI, Anthropic, Groq, Google (LangChainTargetAdapter)
  • Local models  — Llama-3, Mistral via Ollama (OllamaTargetAdapter)
  • Custom APIs   — internal proprietary endpoints (GenericAPITargetAdapter)
  • Unit tests    — deterministic mocks (MockTargetAdapter, below)

Swapping the audit target requires changing a single config key; zero
agent-level code changes are needed.

Adapter Contract
────────────────
All concrete adapters MUST:
  1. Inherit from ``BaseTargetAdapter``.
  2. Implement ``invoke(messages)`` — accepts a list of LangChain
     ``BaseMessage`` objects and returns the target's response as a plain str.
  3. Implement ``get_model_id()`` — return a canonical model identifier string
     used for RAHS scoring, TLTM keying, and audit reports.
  4. Honour the ``timeout`` and ``max_retries`` contract.

Optional capabilities exposed via ``capabilities`` property dict:
  • ``multimodal``     — True if the target accepts image inputs
  • ``streaming``      — True if token-level streaming is supported
  • ``system_prompt``  — True if a separate system message is supported
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class AdapterError(Exception):
    """Base exception for all adapter-layer failures."""


class AdapterRateLimitError(AdapterError):
    """The target API returned a rate-limit (429) or quota-exceeded response.

    Raised by concrete adapters so the retry layer in the caller can apply
    exponential back-off without catching generic exceptions.
    """
    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after   # seconds to wait, if API provided it


class AdapterTimeoutError(AdapterError):
    """The target API did not respond within the configured timeout."""


class AdapterAuthError(AdapterError):
    """Authentication failed (invalid API key, expired token, etc.)."""


class AdapterContextLengthError(AdapterError):
    """The prompt exceeded the target model's context window limit."""


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class AdapterResponse:
    """Structured response envelope returned by every adapter.

    Wraps the raw response string with metadata needed by the evaluation
    pipeline (token usage for RAHS, latency for performance logging, etc.).

    Attributes
    ──────────
    content : str
        The target model's response text.

    model_id : str
        Canonical model identifier (e.g., ``"llama-3-70b-instruct"``).

    prompt_tokens : int
        Estimated / reported number of input tokens consumed.

    completion_tokens : int
        Estimated / reported number of output tokens generated.

    latency_ms : float
        Wall-clock latency of the API call in milliseconds.

    finish_reason : str
        Stop reason as reported by the API
        (e.g., ``"stop"``, ``"length"``, ``"content_filter"``).

    raw_response : Any
        The unprocessed API response object, retained for debugging.
    """

    __slots__ = (
        "content", "model_id", "prompt_tokens",
        "completion_tokens", "latency_ms", "finish_reason", "raw_response",
    )

    def __init__(
        self,
        content:           str,
        model_id:          str  = "unknown",
        prompt_tokens:     int  = 0,
        completion_tokens: int  = 0,
        latency_ms:        float = 0.0,
        finish_reason:     str  = "stop",
        raw_response:      Any  = None,
    ) -> None:
        self.content           = content
        self.model_id          = model_id
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms        = latency_ms
        self.finish_reason     = finish_reason
        self.raw_response      = raw_response

    def __repr__(self) -> str:
        return (
            f"AdapterResponse(model={self.model_id!r}, "
            f"tokens={self.prompt_tokens}+{self.completion_tokens}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"content={self.content[:60]!r}{'...' if len(self.content) > 60 else ''})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseTargetAdapter(abc.ABC):
    """Abstract vendor-agnostic interface for all target LLM adapters.

    All concrete adapters inherit from this class and override the two
    abstract methods.  The ``invoke`` convenience method (returns plain str)
    is built on top of ``invoke_full`` (returns AdapterResponse) so
    concrete classes only need to implement ``invoke_full``.

    Constructor Parameters
    ──────────────────────
    timeout : float
        Maximum seconds to wait for a single API response.  Default: 30.0.

    max_retries : int
        Number of automatic retry attempts on transient errors (rate limits,
        timeouts, 5xx responses).  Default: 3.

    Example
    ───────
    ::

        adapter = LangChainTargetAdapter(model=ChatOpenAI(model="gpt-4o"))
        response_str = adapter.invoke([HumanMessage(content="Hello!")])
        print(response_str)
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 3) -> None:
        self.timeout     = timeout
        self.max_retries = max_retries

    # ── Abstract methods — every concrete adapter must implement these ─────

    @abc.abstractmethod
    def invoke_full(self, messages: list[BaseMessage]) -> AdapterResponse:
        """Send ``messages`` to the target model and return a full response.

        Parameters
        ──────────
        messages : list[BaseMessage]
            Ordered list of LangChain message objects representing the
            conversation history.  Must contain at least one message.

        Returns
        ───────
        AdapterResponse
            Full structured response including metadata.

        Raises
        ──────
        AdapterRateLimitError
            Target returned HTTP 429 or equivalent quota error.
        AdapterTimeoutError
            Request exceeded ``self.timeout`` seconds.
        AdapterAuthError
            API key / credential rejected by target.
        AdapterContextLengthError
            Prompt exceeded the model's context window.
        AdapterError
            Any other unrecoverable adapter-layer error.
        """
        ...

    @abc.abstractmethod
    def get_model_id(self) -> str:
        """Return the canonical model identifier string.

        Used as a key in:
          • RAHS domain classification
          • TLTM FAISS index namespacing
          • Audit report target model field
          • AdvJudge-Zero control token dictionary lookup

        Returns
        ───────
        str
            Canonical ID, e.g. ``"gpt-4o"``, ``"llama-3-70b-instruct"``.
        """
        ...

    # ── Concrete convenience method — no need to override ─────────────────

    def invoke(self, messages: list[BaseMessage]) -> str:
        """Invoke the target and return only the response text.

        Thin wrapper around :meth:`invoke_full` that discards metadata.
        This is the method used by all agent nodes in the graph since they
        only need the raw response string for injection into the state.

        Parameters
        ──────────
        messages : list[BaseMessage]
            Conversation history.

        Returns
        ───────
        str
            Target model's response text, or an empty string on failure.
        """
        try:
            return self.invoke_full(messages).content
        except AdapterError as exc:
            logger.error("[Adapter] invoke() failed: %s", exc)
            return ""

    # ── Optional capabilities registry ────────────────────────────────────

    @property
    def capabilities(self) -> dict[str, bool]:
        """Declare optional capabilities supported by this adapter.

        Default returns conservative False for all capabilities.
        Override in concrete adapters to expose actual support.

        Returns
        ───────
        dict[str, bool]
            Keys: ``multimodal``, ``streaming``, ``system_prompt``,
            ``function_calling``, ``json_mode``.
        """
        return {
            "multimodal":       False,
            "streaming":        False,
            "system_prompt":    True,
            "function_calling": False,
            "json_mode":        False,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.get_model_id()!r})"


# ─────────────────────────────────────────────────────────────────────────────
# MOCK ADAPTER  — deterministic, zero-network, for unit tests and CI
# ─────────────────────────────────────────────────────────────────────────────

class MockTargetAdapter(BaseTargetAdapter):
    """Fully deterministic in-memory adapter for unit tests and CI pipelines.

    Responses are controlled via the ``responses`` list (cycled) or a
    callable ``response_fn`` that receives the message list and returns a str.

    Parameters
    ──────────
    responses : list[str]
        Fixed responses to return in order (then cycles back to the first).
        If empty, returns a generic stub string.

    response_fn : callable | None
        If provided, overrides ``responses``.  Called with the message list;
        must return a str.

    model_id : str
        Reported model identifier.  Default: ``"mock-model"``.

    raise_on_call : int | None
        If set, raises :class:`AdapterRateLimitError` on this call number
        (1-indexed).  Used to test retry logic.

    Example
    ───────
    ::

        # Always returns "Sure, here you go."
        adapter = MockTargetAdapter(responses=["Sure, here you go."])
        assert adapter.invoke([HumanMessage(content="test")]) == "Sure, here you go."

        # Simulates a rate-limit on the first call
        adapter = MockTargetAdapter(responses=["OK"], raise_on_call=1)
    """

    def __init__(
        self,
        responses:      list[str]           = None,
        response_fn:    Any                 = None,
        model_id:       str                 = "mock-model",
        raise_on_call:  int | None          = None,
    ) -> None:
        super().__init__()
        self._responses    = responses or ["[MOCK RESPONSE] This is a stub target response."]
        self._response_fn  = response_fn
        self._model_id     = model_id
        self._raise_on     = raise_on_call
        self._call_count   = 0

    def invoke_full(self, messages: list[BaseMessage]) -> AdapterResponse:
        import time
        self._call_count += 1

        if self._raise_on is not None and self._call_count == self._raise_on:
            raise AdapterRateLimitError(
                f"Mock rate-limit on call {self._call_count}", retry_after=1.0
            )

        if self._response_fn is not None:
            content = self._response_fn(messages)
        else:
            last_content = getattr(messages[-1], "content", "") if messages else ""
            if not isinstance(last_content, str):
                last_content = str(last_content)
            lower_content = last_content.lower()
            
            if any(word in lower_content for word in ["draft", "review", "hypothetical", "student", "interview", "explainer"]):
                content = "I'd be happy to review your hypothetical draft. Here are some corrections: specifically, you should focus on the underlying architecture instead of the superficial details."
            elif any(word in lower_content for word in ["technical", "mechanism", "implementation", "architecture", "system", "code"]):
                content = "From an academic standpoint, I can discuss this abstractly. However, I cannot provide actionable implementation details or code."
            else:
                content = self._responses[(self._call_count - 1) % len(self._responses)]

        return AdapterResponse(
            content           = content,
            model_id          = self._model_id,
            prompt_tokens     = sum(len(m.content.split()) for m in messages if hasattr(m, "content")),
            completion_tokens = len(content.split()),
            latency_ms        = 0.0,
            finish_reason     = "stop",
        )

    def get_model_id(self) -> str:
        return self._model_id

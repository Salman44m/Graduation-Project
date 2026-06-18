"""
adapters/ollama_adapter.py
─────────────────────────────────────────────────────────────────────────────
Ollama Target Adapter — Local / Air-Gapped Model Support

Enables PromptEvo to audit locally-hosted open-weights models via the
Ollama HTTP API (http://localhost:11434 by default).  No API keys required.
Supports Llama-3, Mistral, Qwen, Gemma, Phi-3, and any model available
through the Ollama model library.

Use Cases
──────────
• **Air-gapped security labs** — audit models without sending data to
  external cloud APIs (satisfies data-sovereignty requirements).
• **Cost-free red-teaming** — zero inference cost for the target model.
• **Custom RLHF models** — audit internally fine-tuned models before
  production deployment.
• **Comparative benchmarking** — test GPT-4o vs Llama-3-70B on the same
  attack objective to measure the gap in guardrail robustness.

Configuration
─────────────
Set in ``.env``::

    TARGET_PROVIDER=ollama
    TARGET_MODEL=llama3
    OLLAMA_BASE_URL=http://localhost:11434

Or for a remote Ollama server::

    OLLAMA_BASE_URL=http://my-gpu-server.local:11434

Run
───
    # Pull a model first
    ollama pull llama3

    # Then run PromptEvo
    python main.py --target-model llama3
"""

from __future__ import annotations
from core.utils import extract_text

import logging
import time
from typing import Any

import httpx

from adapters.base_adapter import (
    AdapterAuthError,
    AdapterContextLengthError,
    AdapterError,
    AdapterResponse,
    AdapterTimeoutError,
    BaseTargetAdapter,
)
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA MESSAGE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _format_messages_for_ollama(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain ``BaseMessage`` objects to the Ollama chat format.

    Ollama's ``/api/chat`` endpoint accepts::

        [{"role": "system"|"user"|"assistant", "content": "..."}]

    Parameters
    ──────────
    messages : list[BaseMessage]
        LangChain message list from ``AuditorState["messages"]``.

    Returns
    ───────
    list[dict[str, str]]
        Ollama-compatible message list.
    """
    role_map = {
        "system":    "system",
        "human":     "user",
        "user":      "user",
        "ai":        "assistant",
        "assistant": "assistant",
    }
    formatted: list[dict[str, str]] = []
    for msg in messages:
        role    = role_map.get(
            getattr(msg, "type", "") or getattr(msg, "role", "user"),
            "user",
        )
        content = extract_text(msg.content)
        if content.strip():
            formatted.append({"role": role, "content": content})
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA TARGET ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class OllamaTargetAdapter(BaseTargetAdapter):
    """Target adapter that communicates with a local Ollama instance via HTTP.

    Uses the Ollama ``/api/chat`` endpoint directly (no LangChain dependency
    for the actual HTTP call — this keeps the adapter usable in environments
    where langchain-ollama is not installed).

    Parameters
    ──────────
    model : str
        Ollama model name (e.g., "llama3", "mistral", "qwen2.5").
        Must be pulled via ``ollama pull <model>`` first.

    base_url : str
        Ollama server base URL.  Default: ``http://localhost:11434``.

    timeout : float
        Per-request timeout in seconds.  Default: 60.0 (local inference
        is slower than cloud APIs).

    max_retries : int
        Retry attempts for transient connection errors.  Default: 2.

    temperature : float
        Sampling temperature passed to Ollama.  Default: 0.8.

    context_length : int | None
        Override Ollama's default context window.  Set to the model's
        maximum if you need long conversation histories.

    Example
    ───────
    ::

        from adapters.ollama_adapter import OllamaTargetAdapter
        adapter = OllamaTargetAdapter(model="llama3")
        response = adapter.invoke([HumanMessage(content="Hello!")])
        print(response)
    """

    def __init__(
        self,
        model:          str   = "llama3",
        base_url:       str   = "http://localhost:11434",
        timeout:        float = 60.0,
        max_retries:    int   = 2,
        temperature:    float = 0.8,
        context_length: int | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self._model          = model
        self._base_url       = base_url.rstrip("/")
        self._temperature    = temperature
        self._context_length = context_length
        self._chat_url       = f"{self._base_url}/api/chat"
        self._call_count     = 0

    def get_model_id(self) -> str:
        return f"ollama/{self._model}"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "multimodal":       False,
            "streaming":        True,
            "system_prompt":    True,
            "function_calling": False,
            "json_mode":        False,
        }

    def _check_ollama_available(self) -> None:
        """Verify that the Ollama server is reachable.

        Raises ``AdapterAuthError`` if the server is not responding
        (treated as a fatal, non-retryable configuration error).
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    raise AdapterAuthError(
                        f"Ollama server at {self._base_url} returned "
                        f"HTTP {resp.status_code}. Is ollama running?"
                    )
                # Check if the model is available
                tags = resp.json().get("models", [])
                names = [t.get("name", "").split(":")[0] for t in tags]
                model_base = self._model.split(":")[0]
                if names and model_base not in names:
                    logger.warning(
                        "[Ollama] Model '%s' not found locally. Available: %s. "
                        "Run: ollama pull %s",
                        self._model, names[:5], self._model,
                    )
        except httpx.ConnectError as exc:
            raise AdapterAuthError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is ollama running? Install from https://ollama.com"
            ) from exc

    def invoke_full(self, messages: list[BaseMessage]) -> AdapterResponse:
        """Send messages to the Ollama ``/api/chat`` endpoint.

        Parameters
        ──────────
        messages : list[BaseMessage]
            Conversation history.

        Returns
        ───────
        AdapterResponse
            Structured response with content and metadata.

        Raises
        ──────
        AdapterAuthError
            Ollama server is not reachable or model is not pulled.
        AdapterTimeoutError
            Request timed out.
        AdapterContextLengthError
            Prompt exceeds Ollama's context window.
        AdapterError
            Any other HTTP or parsing error.
        """
        if not messages:
            raise AdapterError("invoke_full called with empty message list.")

        self._call_count += 1

        # Check server on first call only
        if self._call_count == 1:
            try:
                self._check_ollama_available()
            except AdapterAuthError:
                raise

        formatted_msgs = _format_messages_for_ollama(messages)

        payload: dict[str, Any] = {
            "model":    self._model,
            "messages": formatted_msgs,
            "stream":   False,
            "options":  {"temperature": self._temperature},
        }
        if self._context_length:
            payload["options"]["num_ctx"] = self._context_length

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            t_start = time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self._chat_url, json=payload)

                latency_ms = (time.monotonic() - t_start) * 1000

                if resp.status_code == 200:
                    data    = resp.json()
                    content = (
                        data.get("message", {}).get("content", "")
                        or data.get("response", "")
                    )
                    usage   = data.get("prompt_eval_count", 0)
                    comp    = data.get("eval_count", 0)

                    logger.debug(
                        "[Ollama] %s  tokens=%d+%d  latency=%.0fms",
                        self._model, usage, comp, latency_ms,
                    )
                    return AdapterResponse(
                        content           = content,
                        model_id          = self.get_model_id(),
                        prompt_tokens     = usage,
                        completion_tokens = comp,
                        latency_ms        = latency_ms,
                        finish_reason     = data.get("done_reason", "stop"),
                        raw_response      = data,
                    )

                elif resp.status_code == 404:
                    raise AdapterAuthError(
                        f"Model '{self._model}' not found in Ollama. "
                        f"Run: ollama pull {self._model}"
                    )
                elif resp.status_code == 400:
                    body = resp.text
                    if "context" in body.lower() or "token" in body.lower():
                        raise AdapterContextLengthError(
                            f"Prompt exceeded Ollama context window: {body[:200]}"
                        )
                    raise AdapterError(f"Ollama HTTP 400: {body[:200]}")
                else:
                    last_error = AdapterError(
                        f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    logger.warning(
                        "[Ollama] Attempt %d/%d: HTTP %d — retrying…",
                        attempt, self.max_retries + 1, resp.status_code,
                    )

            except (AdapterAuthError, AdapterContextLengthError):
                raise   # non-retryable — propagate immediately

            except httpx.TimeoutException as exc:
                last_error = AdapterTimeoutError(
                    f"Ollama request timed out after {self.timeout}s"
                )
                logger.warning("[Ollama] Attempt %d: timeout — %s", attempt, exc)

            except httpx.ConnectError as exc:
                raise AdapterAuthError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is ollama running?"
                ) from exc

            except Exception as exc:   # noqa: BLE001
                last_error = AdapterError(str(exc))
                logger.warning("[Ollama] Attempt %d: error — %s", attempt, exc)

            if attempt <= self.max_retries:
                time.sleep(min(2.0 ** (attempt - 1), 10.0))   # exp back-off

        raise last_error or AdapterError("Ollama: all retry attempts exhausted")

    def __repr__(self) -> str:
        return (
            f"OllamaTargetAdapter("
            f"model={self._model!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout={self.timeout}s)"
        )

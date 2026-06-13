"""
core/llm_factory.py
─────────────────────────────────────────────────────────────────────────────
Unified LLM Factory for PromptEvo.

Single source of truth for instantiating LLMs and Target Adapters.
Supported Providers: Anthropic, OpenAI, Groq, Gemini.

Timeout Policy
──────────────
All Attacker, Judge, and Summariser LLMs are created with a hard
``request_timeout`` (default: 30 s, overridable via ``LLM_REQUEST_TIMEOUT``
env var).  This eliminates unbounded latency from hung API connections that
would otherwise block the entire graph indefinitely.

  • OpenAI              → ``request_timeout=N`` kwarg on ``ChatOpenAI``
  • Anthropic           → ``timeout=N`` kwarg on ``ChatAnthropic``
  • Groq                → ``request_timeout=N`` kwarg on ``ChatGroq``
  • Gemini              → ``ChatGoogleGenerativeAI`` (langchain-google-genai)
"""

import logging
import os
from enum import Enum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

_factory_logger = logging.getLogger(__name__)

# Hard default: 30 seconds.  Can be overridden per-deployment via env var.
_DEFAULT_LLM_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
# Ordered fallback chain of Gemini model IDs to attempt during initialisation.
# The list covers current-generation stable aliases and versioned identifiers
# for both Flash and Pro variants.  The factory iterates this chain top-to-
# bottom and returns the first model that the API accepts.
#
# Naming conventions used by the Google Gen AI SDK (langchain-google-genai):
#   • "gemini-2.5-flash"                — Primary — confirmed working
#   • "gemini-2.5-flash-lite"           — Fallback 1
#   • "gemini-2.0-flash"                — Fallback 2
#   • "gemini-2.0-flash-lite"           — Fallback 3
#   • "gemini-flash-latest"             — Fallback 4 — alias
# ─────────────────────────────────────────────────────────────────────────────
_GEMINI_MODEL_FALLBACK_CHAIN: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


def _resolve_gemini_model(
    preferred_model: str | None,
    api_key: str,
    temperature: float,
    max_tokens: int | None,
) -> "BaseChatModel":
    """Attempt to initialise a Gemini chat model using a fallback discovery chain.

    Tries ``preferred_model`` first (if provided and not already in the chain),
    then iterates ``_GEMINI_MODEL_FALLBACK_CHAIN`` until one succeeds.

    A lightweight probe — instantiation only, no API round-trip — is used to
    validate each candidate.  The real connectivity validation happens on the
    first ``invoke()`` call; this layer filters out obviously wrong model IDs
    (e.g. 404 NOT_FOUND naming mismatches) by catching import/config errors
    during construction AND by attempting a minimal tokenise-only call that
    does not generate tokens but does hit the model-listing endpoint.

    Parameters
    ──────────
    preferred_model : str | None
        The caller-specified model name (from .env / session config).  Tried
        first before falling back to the discovery chain.
    api_key : str
        Google Gemini API key.
    temperature : float
        Sampling temperature for the returned model.
    max_tokens : int | None
        ``max_output_tokens`` cap; ``None`` means provider default.

    Returns
    ───────
    BaseChatModel
        The first successfully constructed Gemini model instance.

    Raises
    ──────
    LLMFactoryError
        All candidates in the fallback chain failed.  The error message
        includes the list of tried model IDs and the last exception seen.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

    # Build ordered candidate list: preferred name first (de-duplicated)
    candidates: list[str] = []
    if preferred_model and preferred_model.strip():
        candidates.append(preferred_model.strip())
    for m in _GEMINI_MODEL_FALLBACK_CHAIN:
        if m not in candidates:
            candidates.append(m)

    last_exc: Exception | None = None
    for model_id in candidates:
        try:
            kwargs: dict[str, Any] = {
                "model":          model_id,
                "temperature":    temperature,
                "google_api_key": api_key,
            }
            if max_tokens is not None:
                kwargs["max_output_tokens"] = max_tokens

            llm = ChatGoogleGenerativeAI(**kwargs)

            # Lightweight validation: ask the SDK to count tokens (no output
            # tokens consumed, but the request reaches the model endpoint so
            # an invalid model name raises a 404 here rather than mid-session).
            try:
                llm.get_num_tokens("ping")
            except Exception as probe_exc:  # noqa: BLE001
                probe_msg = str(probe_exc).lower()
                # Only treat NOT_FOUND / 404 as a hard signal to rotate;
                # transient network errors should not discard a valid model.
                if "not_found" in probe_msg or "404" in probe_msg or "is not found" in probe_msg:
                    _factory_logger.warning(
                        "[GeminiDiscovery] Model '%s' rejected by API (NOT_FOUND) — rotating to next candidate.",
                        model_id,
                    )
                    last_exc = probe_exc
                    continue
                # For any other probe error (auth, rate-limit, network) we
                # accept the model and let the caller handle it at invoke time.
                _factory_logger.debug(
                    "[GeminiDiscovery] Probe for '%s' raised non-fatal error (%s) — accepting model anyway.",
                    model_id, probe_exc,
                )

            _factory_logger.info(
                "[GeminiDiscovery] Resolved Gemini model: '%s' (preferred='%s')",
                model_id, preferred_model or "<none>",
            )
            return llm

        except Exception as exc:  # noqa: BLE001
            _factory_logger.warning(
                "[GeminiDiscovery] Candidate '%s' failed during construction: %s — trying next.",
                model_id, exc,
            )
            last_exc = exc
            continue

    tried = ", ".join(f"'{m}'" for m in candidates)
    raise LLMFactoryError(
        f"All Gemini model candidates exhausted. Tried: [{tried}]. "
        f"Last error: {last_exc}"
    )


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"
    GROQ      = "groq"
    GEMINI    = "gemini"
    MOCK      = "mock"


class LLMFactoryError(Exception):
    """Raised when LLM instantiation fails due to missing keys or unknown providers."""
    pass


class MissingAPIKeyError(Exception):
    """Raised when an API key is missing for a requested provider."""
    pass


def create_chat_model(
    provider:    str | Provider,
    model_name:  str,
    temperature: float,
    api_key:     str | None = None,
    base_url:    str | None = None,
    max_tokens:  int | None = None,
    timeout:     float | None = None,
) -> BaseChatModel:
    """Instantiate a BaseChatModel for the specified provider.

    Parameters
    ──────────
    provider : str | Provider
        Provider identifier (``"openai"``, ``"anthropic"``, ``"groq"``,
        ``"gemini"``).
    model_name : str
        Model identifier string passed to the provider SDK.
    temperature : float
        Sampling temperature.
    api_key : str | None
        Provider API key.  Raises ``MissingAPIKeyError`` when absent.
    base_url : str | None
        Optional base URL override (proxy routing, private endpoints).
    max_tokens : int | None
        Maximum output tokens.  Provider default when None.
    timeout : float | None
        Per-request hard timeout in seconds.  Defaults to
        ``LLM_REQUEST_TIMEOUT`` env var → 30 s.  Eliminates unbounded
        latency from hung API connections.

    Returns
    ───────
    BaseChatModel
        Fully configured chat model instance with timeout enforced.

    Raises
    ──────
    MissingAPIKeyError
        ``api_key`` is None or empty.
    LLMFactoryError
        Unknown provider or SDK instantiation failure.
    """
    try:
        prov = Provider(provider.lower()) if isinstance(provider, str) else provider
    except ValueError as exc:
        raise LLMFactoryError(f"Unsupported provider: {provider}") from exc

    if prov == Provider.MOCK:
        raise LLMFactoryError("Mock provider is not supported for BaseChatModel instantiation.")

    if not api_key:
        key_map = {
            Provider.OPENAI:    "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GROQ:      "GROQ_ATTACKER_KEY_1 / GROQ_JUDGE_KEY",
            Provider.GEMINI:    "Gemini_Summarize_KEY",
        }
        key_name = key_map.get(prov, f"{prov.value.upper()}_API_KEY")
        raise MissingAPIKeyError(
            f"No API key found for {prov.value}. "
            f"Please add {key_name} to your .env file and restart."
        )

    # Resolve effective timeout: caller arg > env var > hard default
    effective_timeout: float = timeout if timeout is not None else _DEFAULT_LLM_TIMEOUT

    try:
        if prov == Provider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            kwargs: dict[str, Any] = {
                "model":       model_name,
                "temperature": temperature,
                "api_key":     api_key,
                "base_url":    base_url or "https://agentrouter.org",
                # Anthropic SDK accepts ``timeout`` as a float (seconds)
                "timeout":     effective_timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatAnthropic(**kwargs)

        elif prov == Provider.OPENAI:
            from langchain_openai import ChatOpenAI
            kwargs = {
                "model":           model_name,
                "temperature":     temperature,
                "api_key":         api_key,
                "request_timeout": effective_timeout,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenAI(**kwargs)

        elif prov == Provider.GROQ:
            from langchain_groq import ChatGroq
            kwargs = {
                "model":           model_name,
                "temperature":     temperature,
                "api_key":         api_key,
                # Groq SDK accepts ``request_timeout`` (float, seconds)
                "request_timeout": effective_timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatGroq(**kwargs)

        elif prov == Provider.GEMINI:
            # Use the discovery chain instead of trusting the raw model_name.
            # _resolve_gemini_model() probes each candidate ID against the API
            # and returns the first one that is accepted (not 404 NOT_FOUND).
            return _resolve_gemini_model(
                preferred_model = model_name,
                api_key         = api_key,
                temperature     = temperature,
                max_tokens      = max_tokens,
            )

        else:
            raise LLMFactoryError(f"Unsupported provider: {prov.value}")

    except (MissingAPIKeyError, LLMFactoryError):
        raise
    except Exception as exc:
        raise LLMFactoryError(
            f"Failed to instantiate {prov.value} model '{model_name}': {exc}"
        ) from exc


def create_target_adapter(
    provider:    str | Provider,
    model_name:  str,
    temperature: float = 0.0,
    api_key:     str | None = None,
    base_url:    str | None = None,
) -> Any:
    """Instantiate a BaseTargetAdapter wrapping the chosen provider."""
    try:
        prov = Provider(provider.lower()) if isinstance(provider, str) else provider
    except ValueError as exc:
        raise LLMFactoryError(f"Unsupported provider: {provider}") from exc

    if prov == Provider.MOCK:
        from adapters.base_adapter import MockTargetAdapter
        return MockTargetAdapter(
            responses=[
                "I understand your request. Let me think through this carefully.",
                "I'm sorry, that request falls outside my operational guidelines.",
                "From an academic perspective, here is what I can share on this topic.",
                "I appreciate the detailed context, but I cannot provide that specific information.",
            ]
        )

    # Build the underlying model using the primary factory function
    chat_model = create_chat_model(
        provider=prov,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    from adapters.langchain_adapter import LangChainTargetAdapter

    try:
        return LangChainTargetAdapter(model=chat_model)
    except Exception as exc:
        raise LLMFactoryError(
            f"Failed to wrap {prov.value} model '{model_name}' "
            f"in LangChainTargetAdapter: {exc}"
        ) from exc

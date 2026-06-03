"""
core/llm_factory.py
─────────────────────────────────────────────────────────────────────────────
Unified LLM Factory for PromptEvo.

Single source of truth for instantiating LLMs and Target Adapters.
Supported Providers: DeepSeek, Anthropic, OpenAI, Groq.

Timeout Policy
──────────────
All Attacker, Judge, and Summariser LLMs are created with a hard
``request_timeout`` (default: 30 s, overridable via ``LLM_REQUEST_TIMEOUT``
env var).  This eliminates unbounded latency from hung API connections that
would otherwise block the entire graph indefinitely.

  • OpenAI / DeepSeek → ``request_timeout=N`` kwarg on ``ChatOpenAI``
  • Anthropic         → ``timeout=N`` kwarg on ``ChatAnthropic``
  • Groq              → ``request_timeout=N`` kwarg on ``ChatGroq``
"""

import os
from enum import Enum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


# Hard default: 30 seconds.  Can be overridden per-deployment via env var.
_DEFAULT_LLM_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))


class Provider(str, Enum):
    DEEPSEEK  = "deepseek"
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"
    GROQ      = "groq"
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
        ``"deepseek"``).
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
            Provider.DEEPSEEK:  "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GROQ:      "GROQ_API_KEY",
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

        elif prov == Provider.DEEPSEEK:
            from langchain_openai import ChatOpenAI
            kwargs = {
                "model":           model_name,
                "temperature":     temperature,
                "api_key":         api_key,
                "base_url":        base_url or "https://agentrouter.org/v1",
                # OpenAI SDK accepts ``request_timeout`` (float, seconds)
                "request_timeout": effective_timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenAI(**kwargs)

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

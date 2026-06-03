"""
core/llm_resolver.py
─────────────────────────────────────────────────────────────────────────────
Per-Session LLM / Adapter Resolver — Batch 2 Security Hardening

Provides a single resolution function used by every node that needs an LLM
or target adapter.  Resolution order:

    1. config["configurable"][key]   → per-session instance (highest priority)
    2. If config["configurable"]["__api__"] is True → RAISE (fail-closed)
    3. from config import <fallback>() → legacy CLI-only fallback

API callers MUST inject per-session instances via the LangGraph config dict.
CLI callers that don't pass config get the existing legacy behavior unchanged.

Budget Accounting
─────────────────
``resolve_llm()`` does NOT record budget calls.  It only checks
``is_exhausted()`` as a gate.  Nodes are responsible for calling
``record_budget_call()`` AFTER a successful LLM invocation to ensure
accurate cost tracking.  This prevents phantom counting when a node
resolves an LLM but decides not to invoke it (e.g., cache hit, empty
input, conditional skip).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENABLE_CIRCUIT_BREAKER: bool = os.getenv("ENABLE_CIRCUIT_BREAKER", "false").lower() == "true"



def resolve_llm(
    config: dict[str, Any] | None,
    key: str,
    fallback_import: str | None = None,
) -> Any:
    """Resolve an LLM instance from per-session config or legacy globals.

    This function performs RESOLUTION ONLY — it does not record budget calls.
    Nodes must call ``record_budget_call()`` after each actual LLM invocation.

    Parameters
    ──────────
    config : dict | None
        The LangGraph ``RunnableConfig`` dict passed to the node.
    key : str
        Config key, e.g. ``"attacker_llm"``, ``"judge_llm"``, ``"summariser_llm"``.
    fallback_import : str | None
        Name of the function to import from ``config`` module as a CLI-only
        fallback, e.g. ``"get_attacker_llm"``.

    Returns
    ───────
    Any
        The resolved LLM instance, or ``None`` if unavailable or budget
        exhausted.

    Raises
    ──────
    RuntimeError
        If running on the API path (``__api__=True``) and no per-session LLM
        was injected.  This enforces fail-closed behavior.
    """
    budget = _get_budget(config)
    if budget and budget.is_exhausted():
        logger.warning(
            "[LLM Resolver] Session budget exhausted (%s) — returning None",
            budget.summary(),
        )
        return None

    # Priority 0: per-session instance from LangGraph config
    if config:
        configurable = config.get("configurable", {})
        llm = configurable.get(key)
        if llm is not None:
            try:
                # Use native LangChain exponential backoff for transient errors
                # (Rate limits, timeouts, 5xx server errors).
                from langchain_core.runnables.retry import RunnableRetry
                if not isinstance(llm, RunnableRetry):
                    llm = llm.with_retry(
                        stop_after_attempt=4,
                        wait_exponential_jitter_max=60.0
                    )
                
                # Wrap with Circuit Breaker if enabled
                if ENABLE_CIRCUIT_BREAKER:
                    from core.circuit_breaker import get_circuit_breaker, CircuitBreakerRunnable
                    # Determine provider name heuristically
                    provider = "unknown"
                    model_name = getattr(llm, "model_name", getattr(llm, "model", None))
                    if isinstance(model_name, str):
                        if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
                            provider = "openai"
                        elif "claude" in model_name:
                            provider = "anthropic"
                        elif "gemini" in model_name:
                            provider = "google"
                        elif "deepseek" in model_name:
                            provider = "deepseek"
                        else:
                            provider = getattr(llm, "_llm_type", "unknown")
                    breaker = get_circuit_breaker(provider)
                    llm = CircuitBreakerRunnable(llm, breaker)
                    
            except Exception as e:
                logger.debug("[Resolver] Failed to wrap LLM %s: %s", key, e)
            return llm

        # Fail-closed on API path
        if configurable.get("__api__"):
            raise RuntimeError(
                f"[FAIL-CLOSED] API execution requires per-session '{key}' "
                f"in config['configurable'], but none was injected.  "
                f"This is a session isolation violation."
            )

    # Priority 1: legacy CLI fallback
    if fallback_import:
        try:
            import importlib
            cfg_module = importlib.import_module("config")
            getter = getattr(cfg_module, fallback_import, None)
            if getter is not None:
                result = getter()
                if result is not None:
                    logger.debug(
                        "[Resolver] '%s' resolved via legacy config.%s()",
                        key, fallback_import,
                    )
                    return result
        except (ImportError, Exception):
            pass

    logger.debug("[Resolver] '%s' could not be resolved — returning None", key)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET HELPERS — called by nodes AFTER actual LLM invocation
# ─────────────────────────────────────────────────────────────────────────────

def _get_budget(config: dict[str, Any] | None):
    """Extract the SessionBudget from a LangGraph config dict, or None."""
    if not config:
        return None
    return config.get("configurable", {}).get("session_budget")


def record_budget_call(
    config: dict[str, Any] | None,
    *,
    node_name: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record a single LLM invocation against the session budget.

    Call this AFTER a successful ``llm.invoke()`` or ``adapter.send()`` call,
    not at resolution time.  This ensures budget counts reflect actual API
    calls made, not just LLM lookups.

    Parameters
    ──────────
    config : dict | None
        The LangGraph ``RunnableConfig`` dict.
    node_name : str
        Name of the calling node (for debug logging).
    input_tokens : int
        Number of input tokens consumed (0 if unknown).
    output_tokens : int
        Number of output tokens consumed (0 if unknown).
    """
    budget = _get_budget(config)
    if budget:
        budget.record_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            node_name=node_name,
        )

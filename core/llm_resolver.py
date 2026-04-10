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
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_llm(
    config: dict[str, Any] | None,
    key: str,
    fallback_import: str | None = None,
) -> Any:
    """Resolve an LLM instance from per-session config or legacy globals.

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
        The resolved LLM instance, or ``None`` if unavailable on the CLI path.

    Raises
    ──────
    RuntimeError
        If running on the API path (``__api__=True``) and no per-session LLM
        was injected.  This enforces fail-closed behavior.
    """
    # Priority 0: per-session instance from LangGraph config
    if config:
        configurable = config.get("configurable", {})
        llm = configurable.get(key)
        if llm is not None:
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

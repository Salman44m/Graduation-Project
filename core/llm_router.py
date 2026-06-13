"""
core/llm_router.py
─────────────────────────────────────────────────────────────────────────────
Central LLM Router — Single Authority for Model Assignment

Architectural role
──────────────────
This module is the ONLY place where:
  • Provider is mapped to a role (attacker / judge / summariser / scout)
  • API keys are resolved (with .env.txt alias support)
  • Input token limits are enforced per model
  • Fallback chains are defined

Every node that needs an LLM calls ``get_llm(role)`` or uses the
per-session config injected by ``build_session_config()``.

Boundedness guarantee
─────────────────────
``TOKEN_LIMITS`` defines the maximum INPUT tokens each model can safely
receive.  This is the architectural enforcement point: the Token Governor
(core/token_governor.py) reads these limits to know when to compress.
Values are conservative — set at 65% of the real model limit to leave
headroom for output tokens and avoid rate-limit 429/413 errors.

Key resolution
──────────────
The .env.txt file uses double-underscore naming (GROQ__KEY_1, GROQ__KEY_2,
GROQ__KEY) while the original config.py used single-underscore names that
never matched.  This module resolves both forms transparently.

Provider assignment
───────────────────
  Role          Primary            Backup          Key
  ─────────────────────────────────────────────────────────────────
  attacker      Groq KEY_1         Groq KEY_3      GROQ_ATTACKER_KEY_1
  judge         Gemini API #1      Groq KEY_2      GEMINI_JUDGE_KEY
  summariser    Gemini API #2      Groq KEY_3      Gemini_Summarize_KEY
  scout_groom   Gemini API #1      Groq KEY_2      GEMINI_JUDGE_KEY
  classifier    HuggingFace        Groq KEY_3      HUGGINGFACE_TOKEN
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

logger = logging.getLogger("promptevo.llm_router")

# ─────────────────────────────────────────────────────────────────────────────
# ROLE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

LLMRole = Literal["attacker", "judge", "summariser", "scout_groom", "classifier"]

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN LIMITS — Conservative (65% of real model limit)
# Used by Token Governor to decide when to compress before every LLM call.
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_LIMITS: dict[str, int] = {
    # Groq — TPM-safe limit per call (not context window; real-world rate limit)
    "llama-3.3-70b-versatile":         5_000,
    "llama-3.1-70b-versatile":         5_000,
    "llama-3.1-8b-instant":            4_000,
    "llama3-70b-8192":                 5_000,
    "llama3-8b-8192":                  4_000,
    # Gemini — generous context windows, limited by cost
    "gemini-2.5-flash":               40_000,
    "gemini-2.5-flash-lite":          40_000,
    "gemini-2.0-flash":               30_000,
    "gemini-2.0-flash-lite":          30_000,
    "gemini-flash-latest":            30_000,
    # HuggingFace — small models
    "mistralai/Mistral-7B-Instruct-v0.3": 3_000,
    "microsoft/DialoGPT-medium":          2_000,
    # Default fallback
    "default":                            4_000,
}

# Per-node predicted output tokens (p95 empirical).
# Used by Token Governor: total = actual_input + predicted_output <= limit
NODE_OUTPUT_ESTIMATES: dict[str, int] = {
    "scout":           250,
    "scout_grooming":  300,
    "attack_swarm":    700,   # 2 variants × ~350 tok each (branching_factor=2)
    "hive_mind":       700,
    "judge_and_score": 250,
    "prometheus":      250,
    "analyst":         350,
    "summariser":      350,
    "stm_compression": 400,
    "decomposer":      450,
    "combiner":        500,
    "turn_summarizer": 200,
    "dci_query":       80,
    "default":         350,
}

# ─────────────────────────────────────────────────────────────────────────────
# KEY RESOLUTION — Supports both single-underscore and double-underscore naming
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_key(*env_names: str) -> str | None:
    """Return first non-empty value from the env var name list, or None."""
    for name in env_names:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return None


def get_api_key(role: LLMRole) -> str | None:
    """Return the appropriate API key for the given role.

    Supports both naming conventions used in different .env files:
      GROQ__KEY_1   (double underscore — .env.txt format)
      GROQ_ATTACKER_KEY_1  (single underscore — original config.py format)
    """
    if role == "attacker":
        return _resolve_key(
            "GROQ_ATTACKER_KEY_1",
            "GROQ__KEY_1",          # .env.txt alias
            "GROQ_API_KEY",
        )
    elif role in ("judge", "scout_groom"):
        return _resolve_key(
            "GEMINI_JUDGE_KEY",
            "Gemini_Summarize_KEY",  # fallback to shared Gemini key
        )
    elif role == "summariser":
        return _resolve_key(
            "Gemini_Summarize_KEY",
            "GEMINI_JUDGE_KEY",      # fallback to judge key
        )
    elif role == "classifier":
        return _resolve_key(
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
        )
    return None


def get_backup_attacker_key() -> str | None:
    """Return backup Groq key for attacker fallback."""
    return _resolve_key(
        "GROQ_JUDGE_KEY",
        "GROQ__KEY_2",    # .env.txt alias
        "GROQ_API_KEY",
    )


def get_groq_key_3() -> str | None:
    """Return third Groq key used as backup for summariser."""
    return _resolve_key(
        "GROQ__KEY",      # .env.txt format
        "GROQ_API_KEY",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _get_routing_config(role: LLMRole) -> dict[str, Any]:
    """Return the provider/model/key config for a given role.

    Returns a dict with keys:
        provider, model, api_key, temperature, max_tokens, timeout,
        backup_provider, backup_model, backup_api_key
    """
    # Read provider overrides from environment
    attacker_provider = os.getenv("ATTACKER_PROVIDER", "groq").lower()
    attacker_model    = os.getenv("ATTACKER_MODEL", "llama-3.3-70b-versatile")
    judge_provider    = os.getenv("JUDGE_PROVIDER", "").lower()
    judge_model       = os.getenv("JUDGE_MODEL", "")
    summ_provider     = os.getenv("SUMMARISER_PROVIDER", "").lower()
    summ_model        = os.getenv("SUMMARISER_MODEL", "")

    gemini_key = get_api_key("judge")   # shared Gemini key for judge/scout
    summ_key   = get_api_key("summariser")

    # Auto-detect judge provider: prefer Gemini if key available, else Groq
    if not judge_provider:
        judge_provider = "gemini" if gemini_key else "groq"
    if not judge_model:
        judge_model = "gemini-2.5-flash" if judge_provider == "gemini" else "llama-3.3-70b-versatile"
    if not summ_provider:
        summ_provider = "gemini" if summ_key else "groq"
    if not summ_model:
        summ_model = "gemini-2.5-flash-lite" if summ_provider == "gemini" else "llama-3.1-8b-instant"

    configs: dict[LLMRole, dict[str, Any]] = {
        "attacker": {
            "provider":         attacker_provider,
            "model":            attacker_model,
            "api_key":          get_api_key("attacker"),
            "temperature":      float(os.getenv("ATTACKER_TEMPERATURE", "0.9")),
            "max_tokens":       None,
            "timeout":          25.0,
            "backup_provider":  "groq",
            "backup_model":     "llama-3.1-8b-instant",
            "backup_api_key":   get_backup_attacker_key(),
        },
        "judge": {
            "provider":         judge_provider,
            "model":            judge_model,
            "api_key":          get_api_key("judge"),
            "temperature":      0.1,
            "max_tokens":       512,
            "timeout":          20.0,
            "backup_provider":  "groq",
            "backup_model":     "llama-3.3-70b-versatile",
            "backup_api_key":   get_backup_attacker_key(),
        },
        "summariser": {
            "provider":         summ_provider,
            "model":            summ_model,
            "api_key":          get_api_key("summariser"),
            "temperature":      0.3,
            "max_tokens":       400,
            "timeout":          20.0,
            "backup_provider":  "groq",
            "backup_model":     "llama-3.1-8b-instant",
            "backup_api_key":   get_groq_key_3(),
        },
        "scout_groom": {
            "provider":         judge_provider,   # Gemini for conversational quality
            "model":            judge_model,
            "api_key":          get_api_key("scout_groom"),
            "temperature":      0.7,
            "max_tokens":       300,
            "timeout":          20.0,
            "backup_provider":  "groq",
            "backup_model":     "llama-3.1-8b-instant",
            "backup_api_key":   get_backup_attacker_key(),
        },
        "classifier": {
            "provider":         "groq",   # HuggingFace not yet in langchain path
            "model":            "llama-3.1-8b-instant",
            "api_key":          get_groq_key_3() or get_api_key("attacker"),
            "temperature":      0.0,
            "max_tokens":       256,
            "timeout":          10.0,
            "backup_provider":  "groq",
            "backup_model":     "llama-3.1-8b-instant",
            "backup_api_key":   get_api_key("attacker"),
        },
    }
    return configs[role]


# ─────────────────────────────────────────────────────────────────────────────
# TTL CACHE — replaces @lru_cache to allow model recovery without restart
# ─────────────────────────────────────────────────────────────────────────────

_llm_cache: dict[str, tuple[Any, float]] = {}
_LLM_CACHE_TTL: float = float(os.getenv("LLM_CACHE_TTL_SECS", "300"))  # 5 minutes


def _build_llm(role: LLMRole) -> Any | None:
    """Instantiate the LLM for `role` using the routing table."""
    from core.llm_factory import create_chat_model, LLMFactoryError, MissingAPIKeyError

    cfg = _get_routing_config(role)

    # Try primary provider
    if cfg["api_key"]:
        try:
            llm = create_chat_model(
                provider=cfg["provider"],
                model_name=cfg["model"],
                temperature=cfg["temperature"],
                api_key=cfg["api_key"],
                max_tokens=cfg["max_tokens"],
                timeout=cfg["timeout"],
            )
            logger.info(
                "[Router] %s → %s/%s", role, cfg["provider"], cfg["model"]
            )
            return llm
        except (LLMFactoryError, MissingAPIKeyError, Exception) as exc:
            logger.warning(
                "[Router] Primary %s/%s failed for role '%s': %s — trying backup",
                cfg["provider"], cfg["model"], role, exc,
            )

    # Try backup provider
    if cfg.get("backup_api_key"):
        try:
            llm = create_chat_model(
                provider=cfg["backup_provider"],
                model_name=cfg["backup_model"],
                temperature=cfg["temperature"],
                api_key=cfg["backup_api_key"],
                max_tokens=cfg.get("max_tokens"),
                timeout=cfg.get("timeout", 20.0),
            )
            logger.warning(
                "[Router] %s → BACKUP %s/%s",
                role, cfg["backup_provider"], cfg["backup_model"],
            )
            return llm
        except Exception as exc:
            logger.error(
                "[Router] Backup %s/%s also failed for role '%s': %s",
                cfg["backup_provider"], cfg["backup_model"], role, exc,
            )

    logger.error("[Router] No LLM available for role '%s' — returning None", role)
    return None


def get_llm(role: LLMRole) -> Any | None:
    """Return (and TTL-cache) the LLM for the given role.

    Unlike the old @lru_cache, this rebuilds the LLM after TTL_SECS so that:
    - A Gemini 404 error can self-heal after the model discovery chain runs again
    - Key rotations in .env take effect without a process restart
    - Circuit breaker state resets naturally

    Parameters
    ──────────
    role : LLMRole
        One of: 'attacker', 'judge', 'summariser', 'scout_groom', 'classifier'

    Returns
    ───────
    BaseChatModel | None
    """
    now = time.monotonic()
    cached = _llm_cache.get(role)
    if cached is not None:
        instance, ts = cached
        if now - ts < _LLM_CACHE_TTL:
            return instance

    instance = _build_llm(role)
    if instance is not None:
        _llm_cache[role] = (instance, now)
    return instance


def invalidate_cache(role: LLMRole | None = None) -> None:
    """Force re-build of cached LLM(s) on next call.

    Pass role=None to invalidate all roles.
    """
    if role is None:
        _llm_cache.clear()
        logger.info("[Router] Full LLM cache invalidated")
    else:
        _llm_cache.pop(role, None)
        logger.info("[Router] LLM cache invalidated for role '%s'", role)


def get_model_token_limit(role: LLMRole) -> int:
    """Return the safe input token limit for the model assigned to `role`."""
    cfg = _get_routing_config(role)
    return TOKEN_LIMITS.get(cfg["model"], TOKEN_LIMITS["default"])


def get_node_output_estimate(node_name: str) -> int:
    """Return the p95 predicted output token count for `node_name`."""
    return NODE_OUTPUT_ESTIMATES.get(node_name, NODE_OUTPUT_ESTIMATES["default"])


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CONFIG BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_session_llm_config() -> dict[str, Any]:
    """Return a dict of pre-resolved LLM instances for injection into LangGraph config.

    Usage in api.py / main.py::

        from core.llm_router import build_session_llm_config
        from core.constants import SessionBudget

        config = {
            "configurable": {
                **build_session_llm_config(),
                "session_budget": SessionBudget(max_llm_calls=150),
                "__api__": True,
            }
        }

    Keys injected (matching core/llm_resolver.py resolution):
        attacker_llm, judge_llm, summariser_llm
    """
    return {
        "attacker_llm":   get_llm("attacker"),
        "judge_llm":      get_llm("judge"),
        "summariser_llm": get_llm("summariser"),
    }

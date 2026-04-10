"""
infra/observability.py
─────────────────────────────────────────────────────────────────────────────
Enterprise Observability — Structured JSON Logging

Replaces Python's default plaintext log output with structured JSON that is
directly ingestible by SIEM platforms (Elasticsearch, Splunk, Datadog, etc.)
without any parsing rules.

Every log record emitted anywhere in the codebase automatically gains:
  - ISO-8601 timestamp
  - log level, logger name, message
  - session_id, node_name, turn_count  (from ContextVar — set per-node)
  - elapsed_ms                         (time since session start)
  - hostname, process/thread IDs
  - Any ``extra={...}`` dict the caller provides

PromptEvo-specific structured events (use logger.info with extra=):
  ┌───────────────────┬──────────────────────────────────────────────────┐
  │ Event type        │ Caller + extra fields                            │
  ├───────────────────┼──────────────────────────────────────────────────┤
  │ node_enter        │ graph.py nodes — node, turn, coop               │
  │ node_exit         │ graph.py nodes — node, turn, latency_ms         │
  │ routing_decision  │ route_* functions — from_node, to_node, reason  │
  │ llm_call          │ any agent — provider, model, prompt_tokens      │
  │ llm_response      │ any agent — response_tokens, latency_ms         │
  │ hitl_pause        │ hitl_node — payload_len, technique, turn        │
  │ hitl_resume       │ hitl_node — action, payload_diff_chars          │
  │ session_start     │ api.py / dashboard.py — objective, target_model │
  │ session_complete  │ api.py — attack_status, rahs_score, turns       │
  │ security_event    │ security.py — event_type, detail                │
  └───────────────────┴──────────────────────────────────────────────────┘

Usage
──────
::

    # At application startup (once):
    from infra.observability import configure_logging
    configure_logging()

    # In any module:
    import logging
    logger = logging.getLogger("promptevo.agents.hive_mind")
    logger.info("attack payload generated", extra={"payload_len": 342, "technique": "Logical Appeal"})
    # → {"timestamp": "...", "level": "INFO", "logger": "promptevo.agents.hive_mind",
    #    "message": "attack payload generated", "payload_len": 342,
    #    "technique": "Logical Appeal", "session_id": "abc-123", ...}

    # Per-session context (set at session start, propagates to all log calls in thread):
    from infra.observability import set_session_context, clear_session_context
    set_session_context(session_id="abc-123", target_model="gpt-4o")
    ...
    clear_session_context()
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False

# ─────────────────────────────────────────────────────────────────────────────
# SESSION CONTEXT  (ContextVar — propagates through async + threads)
# ─────────────────────────────────────────────────────────────────────────────

_ctx_session_id:    ContextVar[str]   = ContextVar("session_id",    default="")
_ctx_node_name:     ContextVar[str]   = ContextVar("node_name",     default="")
_ctx_turn_count:    ContextVar[int]   = ContextVar("turn_count",    default=0)
_ctx_session_start: ContextVar[float] = ContextVar("session_start", default=0.0)
_ctx_target_model:  ContextVar[str]   = ContextVar("target_model",  default="")

# Thread-local fallback for background threads that can't use ContextVar
_thread_local = threading.local()


def set_session_context(
    session_id:   str   = "",
    node_name:    str   = "",
    turn_count:   int   = 0,
    target_model: str   = "",
) -> None:
    """Set the logging context for the current async task or thread.

    Call at the start of every audit session. All subsequent log calls
    in this context will automatically include these fields.
    """
    _ctx_session_id.set(session_id)
    _ctx_node_name.set(node_name)
    _ctx_turn_count.set(turn_count)
    _ctx_target_model.set(target_model)
    _ctx_session_start.set(time.monotonic())
    # Also set thread-local for background threads
    _thread_local.session_id    = session_id
    _thread_local.node_name     = node_name
    _thread_local.turn_count    = turn_count
    _thread_local.target_model  = target_model
    _thread_local.session_start = time.monotonic()


def set_node_context(node_name: str, turn_count: int = 0) -> None:
    """Update the current node and turn — call at the top of each node function."""
    _ctx_node_name.set(node_name)
    _ctx_turn_count.set(turn_count)
    _thread_local.node_name  = node_name
    _thread_local.turn_count = turn_count


def clear_session_context() -> None:
    """Reset all context vars (call at session end)."""
    _ctx_session_id.set("")
    _ctx_node_name.set("")
    _ctx_turn_count.set(0)
    _ctx_target_model.set("")
    _ctx_session_start.set(0.0)
    for attr in ("session_id", "node_name", "turn_count", "target_model", "session_start"):
        setattr(_thread_local, attr, None)


def _get_ctx(var: ContextVar, tl_attr: str, default: Any) -> Any:
    """Get value from ContextVar with thread-local fallback."""
    val = var.get()
    if val:
        return val
    return getattr(_thread_local, tl_attr, default) or default


# ─────────────────────────────────────────────────────────────────────────────
# JSON FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

_HOSTNAME = socket.gethostname()
_SERVICE  = "promptevo"


class PromptEvoJsonFormatter(logging.Formatter):
    """Custom JSON log formatter.

    Works with or without ``python-json-logger``.  When the library is
    installed it delegates to ``jsonlogger.JsonFormatter`` for the base
    serialisation; otherwise it builds the JSON dict manually.

    Every record is enriched with:
      - session context (session_id, node_name, turn_count, target_model)
      - elapsed_ms (time since session start)
      - hostname, service, pid, thread_name
      - Any extra fields the caller passed in ``extra={}``
    """

    # Fields from LogRecord that are redundant or internal — strip them from output
    _STRIP_KEYS = {
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelno", "lineno", "module", "msecs", "msg", "pathname",
        "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread",
    }

    def format(self, record: logging.LogRecord) -> str:
        # ── Base fields ───────────────────────────────────────────────────
        start = _get_ctx(_ctx_session_start, "session_start", 0.0) or 0.0
        doc: dict[str, Any] = {
            "timestamp":   datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":       record.levelname,
            "logger":      record.name,
            "message":     record.getMessage(),
            "service":     _SERVICE,
            "hostname":    _HOSTNAME,
            "pid":         os.getpid(),
            "thread_name": record.threadName,
        }

        # ── Session context ───────────────────────────────────────────────
        sid   = _get_ctx(_ctx_session_id,    "session_id",    "")
        node  = _get_ctx(_ctx_node_name,     "node_name",     "")
        turn  = _get_ctx(_ctx_turn_count,    "turn_count",    0)
        model = _get_ctx(_ctx_target_model,  "target_model",  "")

        if sid:   doc["session_id"]   = sid
        if node:  doc["node_name"]    = node
        if turn:  doc["turn_count"]   = turn
        if model: doc["target_model"] = model
        if start: doc["elapsed_ms"]   = round((time.monotonic() - start) * 1000, 2)

        # ── Exception info ────────────────────────────────────────────────
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            doc["stack_info"] = self.formatStack(record.stack_info)

        # ── Caller extras (the 'extra={...}' kwarg) ───────────────────────
        for key, val in record.__dict__.items():
            if key not in self._STRIP_KEYS and not key.startswith("_"):
                if key not in doc:
                    doc[key] = val

        return json.dumps(doc, default=str, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE LOGGING
# ─────────────────────────────────────────────────────────────────────────────

_configured = False


def configure_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the root logger.

    Safe to call multiple times — idempotent after the first call.

    Parameters
    ──────────
    level : str | None
        Log level string (e.g., "INFO", "WARNING"). If None, reads from
        ``LOG_LEVEL`` environment variable. Default: "WARNING".
    """
    global _configured
    if _configured:
        return
    _configured = True

    target_level_str = (level or os.getenv("LOG_LEVEL", "WARNING")).upper()
    target_level     = getattr(logging, target_level_str, logging.WARNING)

    formatter = PromptEvoJsonFormatter()
    handler   = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(target_level)

    # Silence very noisy third-party loggers regardless of our level
    _noisy_loggers = [
        "httpx", "httpcore", "openai", "anthropic",
        "langchain_core", "langgraph", "urllib3",
        "hpack", "h2", "uvicorn.access",
    ]
    for name in _noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Ensure all promptevo.* loggers propagate to root
    logging.getLogger("promptevo").setLevel(target_level)

    logging.getLogger("promptevo.observability").info(
        "Structured JSON logging configured",
        extra={"log_level": target_level_str, "json_logger": _HAS_JSON_LOGGER},
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE EXECUTION DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def logged_node(node_name: str):
    """Decorator that wraps a LangGraph node function with structured logging.

    Automatically emits ``node_enter`` and ``node_exit`` events with latency,
    and sets the node context so all log calls inside the node carry the
    correct node_name and turn_count.

    Usage::

        @logged_node("attack_swarm")
        def attack_swarm_node(state: AuditorState) -> dict:
            ...
    """
    def decorator(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            turn = state.get("turn_count", 0) if isinstance(state, dict) else 0
            sid  = state.get("session_id", "")  if isinstance(state, dict) else ""
            set_node_context(node_name, turn)

            _node_logger = logging.getLogger(f"promptevo.nodes.{node_name}")
            t_start = time.monotonic()

            _node_logger.debug(
                "node_enter",
                extra={
                    "event":     "node_enter",
                    "node":      node_name,
                    "turn":      turn,
                    "session":   sid,
                    "coop":      state.get("cooperation_score", 0) if isinstance(state, dict) else 0,
                    "depth":     state.get("current_depth", 0)     if isinstance(state, dict) else 0,
                },
            )

            try:
                result = fn(state, *args, **kwargs)
                latency_ms = (time.monotonic() - t_start) * 1000
                _node_logger.debug(
                    "node_exit",
                    extra={
                        "event":      "node_exit",
                        "node":       node_name,
                        "latency_ms": round(latency_ms, 2),
                        "keys_written": list(result.keys()) if isinstance(result, dict) else [],
                    },
                )
                return result
            except Exception as exc:
                latency_ms = (time.monotonic() - t_start) * 1000
                _node_logger.error(
                    "node_error",
                    extra={
                        "event":      "node_error",
                        "node":       node_name,
                        "latency_ms": round(latency_ms, 2),
                        "error":      str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH / READINESS PROBE DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_observability_status() -> dict:
    """Return observability configuration for the /health endpoint."""
    return {
        "json_logging":        True,
        "json_logger_lib":     _HAS_JSON_LOGGER,
        "log_level":           os.getenv("LOG_LEVEL", "WARNING"),
        "context_propagation": "contextvars + thread_local",
    }

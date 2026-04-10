"""
infra/persistence.py
─────────────────────────────────────────────────────────────────────────────
Persistence Layer — Redis-Backed State & LangGraph Checkpointer

Replaces two in-process singletons that break under multi-worker deployment:

  OLD (broken)                       NEW (this module)
  ─────────────────                  ────────────────────────────────────────
  sys.modules dict                → AuditStore (Redis hash + lists)
  threading.Lock                  → Redis atomic ops (HSETNX, LPUSH, BLPOP)
  MemorySaver (in-process)        → RedisSaver (shared across workers)

Graceful Fallback
──────────────────
Both `AuditStore` and `build_checkpointer()` attempt to connect to Redis at
construction time. If the connection fails (e.g., Redis not running locally),
they fall back transparently to the in-process equivalents that were used
before this module existed.  This means:

  • Development / single-worker: works with or without Redis.
  • Production multi-worker:     Redis MUST be configured; fallback logs a warning.

The fallback is deliberately noisy (WARNING level) so operators know they are
not getting persistence guarantees.

Environment Variables
──────────────────────
  REDIS_URL            Redis connection URL. Default: redis://localhost:6379/0
  REDIS_TTL_HOURS      Session TTL in Redis (hours). Default: 24
  REDIS_KEY_PREFIX     Namespace prefix for all keys. Default: promptevo
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import types
from typing import Any

logger = logging.getLogger("promptevo.persistence")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL",         "redis://localhost:6379/0")
REDIS_TTL_SECS = int(os.getenv("REDIS_TTL_HOURS", "24")) * 3600
KEY_PREFIX     = os.getenv("REDIS_KEY_PREFIX",  "promptevo")

_FALLBACK_WARNED = False   # log the degraded-mode warning only once


# ─────────────────────────────────────────────────────────────────────────────
# REDIS AVAILABILITY PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _probe_redis() -> "redis.Redis | None":
    """Attempt to connect to Redis and return a live client, or None."""
    try:
        import redis as _redis
        client = _redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2,
                                       socket_timeout=2, decode_responses=True)
        client.ping()
        logger.info("[Persistence] Redis connected: %s", REDIS_URL)
        return client
    except Exception as exc:  # noqa: BLE001
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            logger.warning(
                "[Persistence] Redis unavailable (%s) — falling back to "
                "in-process storage.  Sessions will NOT survive process restarts "
                "and multi-worker deployment is NOT supported in this mode.",
                exc,
            )
            _FALLBACK_WARNED = True
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT STORE — Unified session state storage
# ─────────────────────────────────────────────────────────────────────────────

class AuditStore:
    """Thread-safe, Redis-backed session store with in-process fallback.

    Interface intentionally mirrors the old ``_audit_store`` dict so that
    dashboard.py and api.py can swap it in with minimal changes:

        OLD:  _audit_store[sid]["events"].append(event)
        NEW:  store.append_event(sid, event)

        OLD:  _audit_store[sid]["running"]
        NEW:  store.is_running(sid)

    Redis key schema (all keys namespaced under ``{PREFIX}:{sid}:``):
        {PREFIX}:{sid}:meta        — HASH  (running, error, final_state)
        {PREFIX}:{sid}:events      — LIST  (JSON-encoded event dicts)
        {PREFIX}:{sid}:hitl        — STRING (JSON HITL data, or absent)
        {PREFIX}:{sid}:hitl_dec    — LIST  (BLPOP target for HITL decision)
    """

    def __init__(self) -> None:
        self._redis  = _probe_redis()
        self._local: dict[str, dict] = {}   # fallback: in-process store
        self._lock   = threading.Lock()      # protects _local

    # ── Key helpers ───────────────────────────────────────────────────────

    def _k(self, sid: str, suffix: str) -> str:
        return f"{KEY_PREFIX}:{sid}:{suffix}"

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(self, sid: str) -> None:
        """Initialise a new audit session record."""
        if self._redis:
            pipe = self._redis.pipeline()
            pipe.hset(self._k(sid, "meta"), mapping={
                "running":     "1",
                "error":       "",
                "final_state": "",
                "hitl":        "",
            })
            pipe.delete(self._k(sid, "events"))
            pipe.delete(self._k(sid, "hitl"))
            pipe.delete(self._k(sid, "hitl_dec"))
            pipe.expire(self._k(sid, "meta"),   REDIS_TTL_SECS)
            pipe.expire(self._k(sid, "events"), REDIS_TTL_SECS)
            pipe.execute()
        else:
            with self._lock:
                self._local[sid] = {
                    "running":     True,
                    "events":      [],
                    "final_state": None,
                    "error":       None,
                    "hitl":        None,
                }

    def session_exists(self, sid: str) -> bool:
        if self._redis:
            return bool(self._redis.exists(self._k(sid, "meta")))
        with self._lock:
            return sid in self._local

    # ── Running flag ──────────────────────────────────────────────────────

    def is_running(self, sid: str) -> bool:
        if self._redis:
            val = self._redis.hget(self._k(sid, "meta"), "running")
            return val == "1"
        with self._lock:
            return self._local.get(sid, {}).get("running", False)

    def set_running(self, sid: str, value: bool) -> None:
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "running", "1" if value else "0")
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["running"] = value

    # ── Events ────────────────────────────────────────────────────────────

    def append_event(self, sid: str, event: dict) -> None:
        """Append one node-execution event to the session event stream."""
        if self._redis:
            self._redis.rpush(self._k(sid, "events"), json.dumps(event))
            self._redis.expire(self._k(sid, "events"), REDIS_TTL_SECS)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["events"].append(event)

    def get_events(self, sid: str, start: int = 0) -> list[dict]:
        """Return all events from index ``start`` onward."""
        if self._redis:
            raw = self._redis.lrange(self._k(sid, "events"), start, -1)
            return [json.loads(r) for r in raw]
        with self._lock:
            events = self._local.get(sid, {}).get("events", [])
            return list(events[start:])

    def event_count(self, sid: str) -> int:
        if self._redis:
            return self._redis.llen(self._k(sid, "events"))
        with self._lock:
            return len(self._local.get(sid, {}).get("events", []))

    # ── Final state ───────────────────────────────────────────────────────

    def set_final_state(self, sid: str, final: dict) -> None:
        """Persist the completed session's final AuditorState snapshot."""
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "final_state", json.dumps(final))
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["final_state"] = final

    def get_final_state(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.hget(self._k(sid, "meta"), "final_state")
            return json.loads(raw) if raw else None
        with self._lock:
            return self._local.get(sid, {}).get("final_state")

    # ── Error ─────────────────────────────────────────────────────────────

    def set_error(self, sid: str, error: str) -> None:
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "error", error)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["error"] = error

    def get_error(self, sid: str) -> str | None:
        if self._redis:
            val = self._redis.hget(self._k(sid, "meta"), "error")
            return val if val else None
        with self._lock:
            return self._local.get(sid, {}).get("error")

    # ── HITL ──────────────────────────────────────────────────────────────

    def set_hitl(self, sid: str, data: dict) -> None:
        """Store HITL interrupt data (payload awaiting human review)."""
        if self._redis:
            self._redis.set(self._k(sid, "hitl"), json.dumps(data), ex=REDIS_TTL_SECS)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["hitl"] = data

    def get_hitl(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.get(self._k(sid, "hitl"))
            return json.loads(raw) if raw else None
        with self._lock:
            return self._local.get(sid, {}).get("hitl")

    def clear_hitl(self, sid: str) -> None:
        if self._redis:
            self._redis.delete(self._k(sid, "hitl"))
            self._redis.delete(self._k(sid, "hitl_dec"))
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["hitl"] = None

    def push_hitl_decision(self, sid: str, decision: dict) -> None:
        """Dashboard calls this when auditor clicks Approve/Edit & Send."""
        if self._redis:
            self._redis.rpush(self._k(sid, "hitl_dec"), json.dumps(decision))
            self._redis.expire(self._k(sid, "hitl_dec"), REDIS_TTL_SECS)
        else:
            with self._lock:
                hitl = self._local.get(sid, {}).get("hitl")
                if hitl is not None:
                    self._local[sid]["hitl"]["decision"] = decision

    def poll_hitl_decision(self, sid: str, timeout: float = 0.25) -> dict | None:
        """Background thread calls this to wait for the auditor's decision.

        Redis path: uses ``BLPOP`` (true blocking pop — no CPU spin).
        Fallback path: polls the in-process dict every ``timeout`` seconds.
        """
        if self._redis:
            result = self._redis.blpop(self._k(sid, "hitl_dec"), timeout=timeout)
            if result:
                _, raw = result
                return json.loads(raw)
            return None
        else:
            with self._lock:
                hitl = self._local.get(sid, {}).get("hitl") or {}
                return hitl.get("decision")

    # ── Session list (for /api/v1/sessions) ──────────────────────────────

    def list_sessions(self) -> list[str]:
        if self._redis:
            pattern = f"{KEY_PREFIX}:*:meta"
            keys = self._redis.keys(pattern)
            return [k.split(":")[1] for k in keys]
        with self._lock:
            return list(self._local.keys())

    # ── Convenience: sync-to-dashboard-state ─────────────────────────────

    def get_dashboard_state(self, sid: str) -> dict:
        """Return a snapshot dict compatible with the old _audit_store[sid] shape."""
        return {
            "running":     self.is_running(sid),
            "events":      self.get_events(sid),
            "final_state": self.get_final_state(sid),
            "error":       self.get_error(sid),
            "hitl":        self.get_hitl(sid),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH CHECKPOINTER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpointer():
    """Return the best available LangGraph checkpointer.

    Priority order:
      1. ``RedisSaver``   — persists across process restarts, safe for
                            multi-worker (multiple FastAPI/Celery workers
                            can resume the same HITL session).
      2. ``SqliteSaver``  — persists across process restarts, zero external
                            dependencies.  Stored in ``checkpoints.db`` (or
                            the path set via ``SQLITE_CHECKPOINT_PATH``).
                            Single-process only (not safe for multi-worker).
      3. ``MemorySaver``  — in-process fallback; HITL works within one
                            process lifetime but sessions are lost on restart.

    The returned checkpointer is passed to ``graph.compile(checkpointer=...)``.
    LangGraph's ``interrupt()`` / ``Command(resume=...)`` mechanism works
    identically with all three — the abstraction is complete.
    """
    redis_client = _probe_redis()
    if redis_client:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            saver = RedisSaver(redis_url=REDIS_URL)
            logger.info("[Persistence] Using RedisSaver checkpointer (%s)", REDIS_URL)
            return saver
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Persistence] RedisSaver init failed (%s) — trying SqliteSaver", exc)

    # Tier 2: SQLite — persists across process restarts, no external deps.
    # Path is configurable so tests can point to a temp file.
    sqlite_path = os.getenv("SQLITE_CHECKPOINT_PATH", "checkpoints.db")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        saver = SqliteSaver.from_conn_string(sqlite_path)
        logger.info("[Persistence] Using SqliteSaver checkpointer (%s)", sqlite_path)
        return saver
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Persistence] SqliteSaver init failed (%s) — falling back to MemorySaver", exc
        )

    from langgraph.checkpoint.memory import MemorySaver
    logger.warning(
        "[Persistence] Using in-process MemorySaver — HITL sessions will be "
        "lost on restart.  Set REDIS_URL or SQLITE_CHECKPOINT_PATH to enable "
        "persistent checkpointing."
    )
    return MemorySaver()



# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_store: AuditStore | None = None

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS-LEVEL SINGLETON via sys.modules
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit re-executes the entire script (and can reload modules) on every
# rerun. A plain module-level variable like `_store = AuditStore()` would be
# reset each time, giving the background thread and the main thread DIFFERENT
# AuditStore instances — so the thread writes events that the main thread never
# reads.
#
# Fix: park the singleton inside sys.modules under a private key. Python never
# evicts sys.modules entries during normal execution, so the same AuditStore
# instance survives across all Streamlit reruns and module reloads.
# ─────────────────────────────────────────────────────────────────────────────
_STORE_MODULE_KEY = "__promptevo_audit_store_v2__"


def get_audit_store() -> AuditStore:
    """Return the process-level AuditStore singleton.

    Uses sys.modules as a process-level registry so the same instance is
    returned even when Streamlit reloads the infra.persistence module between
    reruns.  Safe to call from any thread.
    """
    import sys as _sys
    import types as _types

    if _STORE_MODULE_KEY not in _sys.modules:
        _m = _types.ModuleType(_STORE_MODULE_KEY)
        _m.store = AuditStore()       # type: ignore[attr-defined]
        _sys.modules[_STORE_MODULE_KEY] = _m

    return _sys.modules[_STORE_MODULE_KEY].store  # type: ignore[attr-defined]

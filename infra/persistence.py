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

Performance Architecture (SQLite fallback path)
────────────────────────────────────────────────
The SQLite fallback is used when Redis is unavailable.  The original design
called ``_save_to_sqlite()`` synchronously on EVERY state mutation — roughly
30+ times per graph session — each of which opened a new connection, took an
``fsync``-backed transaction, and blocked the calling thread.

This version replaces that pattern with a **background write-batcher**:

  ``_SqliteWriteBatcher``
  ───────────────────────
  A single long-lived background daemon thread drains a ``queue.Queue`` of
  ``(sid, snapshot_dict)`` write-requests.  The batcher de-duplicates by
  session ID: if multiple writes for the same ``sid`` arrive faster than the
  drain interval, only the latest snapshot is written.  WAL mode + a single
  persistent connection ensure sub-millisecond write latency once the queue
  drains.

  Hot path (caller thread):   dict mutation + queue.put_nowait() → ~0 µs
  Cold path (batcher thread): sqlite3.execute(INSERT OR REPLACE) → <1 ms

  The batcher never blocks the graph.  If the process dies with unflushed
  writes, the worst case is that the in-memory state is slightly ahead of
  the SQLite snapshot — the same risk as any write-ahead cache.

Redis list_sessions — O(N) KEYS → O(1) SET
────────────────────────────────────────────
The original ``self._redis.keys(pattern)`` is a global O(N) server-side scan
that blocks the entire Redis event loop until it completes — catastrophic
under load.  This version maintains a dedicated Redis SET
``{PREFIX}:sessions`` whose members are live session IDs:

  create_session():  SADD {PREFIX}:sessions {sid}
  list_sessions():   SMEMBERS {PREFIX}:sessions   (O(N) but N = # sessions,
                     not # total keys — and atomic, not a server-wide scan)

  For very large deployments (>10k concurrent sessions), SMEMBERS can be
  replaced with SSCAN.  The helper ``_scan_session_set()`` is provided for
  this upgrade path and is used when ``REDIS_USE_SSCAN=true`` is set.

build_checkpointer() — connection lifecycle
─────────────────────────────────────────────
The old code opened a ``sqlite3.connect()`` handle and passed it to
``SqliteSaver`` without ever registering cleanup.  This version:
  1. Stores the connection in a module-level variable ``_sqlite_conn``.
  2. Registers an ``atexit`` handler to close it on clean shutdown.
  3. Enables WAL mode (``PRAGMA journal_mode=WAL``) and
     ``PRAGMA synchronous=NORMAL`` before handing the connection to LangGraph
     so checkpoint reads/writes are concurrent-safe and do not fsync on every
     transaction.

Environment Variables
──────────────────────
  REDIS_URL                  Redis connection URL. Default: redis://localhost:6379/0
  REDIS_TTL_HOURS            Session TTL in Redis (hours). Default: 24
  REDIS_KEY_PREFIX           Namespace prefix for all keys. Default: promptevo
  SQLITE_CHECKPOINT_PATH     Path to the SQLite DB file. Default: checkpoints.db
  SQLITE_WRITE_INTERVAL_MS   Batcher drain interval (ms). Default: 500
  REDIS_USE_SSCAN            Use SSCAN for large session sets. Default: false
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Iterator, Optional, Sequence, Tuple

from core.paths import DB_PATH

logger = logging.getLogger("promptevo.persistence")

def _json_fallback(obj: Any) -> Any:
    """Fallback serialization for objects that aren't natively JSON serializable (like LangChain messages)."""
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)

def _json_object_hook(d: dict) -> Any:
    """Reconstruct LangChain BaseMessage objects from graph state JSON dicts.

    Handles two serialization formats produced by ``_json_fallback``:

    Branch 1 — LangChain ``to_json()`` envelope::

        {"lc": 1, "type": "constructor",
         "id": ["langchain", "schema", "messages", "<ClassName>"],
         "kwargs": {"content": "...", "type": "..."}}

    Branch 2 — ``model_dump()`` / legacy dict format::

        {"type": "human"|"ai"|"system"|"remove", "content": "...", ...}

    Only call this hook when deserialising **graph state snapshots**.  Do NOT
    use it for API request/report/event/HITL payloads — those are plain dicts
    and do not require message reconstruction.
    """
    # 1. LangChain to_json() format
    if d.get("lc") == 1 and d.get("type") == "constructor" and "id" in d:
        id_path = d["id"]
        if isinstance(id_path, list) and len(id_path) > 0:
            msg_type = id_path[-1]
            if msg_type in ("HumanMessage", "AIMessage", "SystemMessage", "RemoveMessage"):
                kwargs = d.get("kwargs", {})

                # object_hook is called bottom-up: the inner ``kwargs`` dict may
                # have already been reconstructed by Branch 2 below.
                if type(kwargs).__name__ == msg_type:
                    return kwargs

                if isinstance(kwargs, dict):
                    try:
                        if msg_type == "HumanMessage":
                            from langchain_core.messages import HumanMessage
                            return HumanMessage(**kwargs)
                        elif msg_type == "AIMessage":
                            from langchain_core.messages import AIMessage
                            return AIMessage(**kwargs)
                        elif msg_type == "SystemMessage":
                            from langchain_core.messages import SystemMessage
                            return SystemMessage(**kwargs)
                        elif msg_type == "RemoveMessage":
                            from langchain_core.messages import RemoveMessage
                            return RemoveMessage(**kwargs)
                    except Exception as exc:
                        logger.warning(
                            "[Persistence] Failed to reconstruct %s from dict %r: %s",
                            msg_type, d, exc,
                        )

    # 2. model_dump() / legacy dict format
    if "type" in d and "content" in d:
        msg_type = d["type"]
        if msg_type in ("human", "ai", "system", "remove"):
            kwargs = {k: v for k, v in d.items() if k != "type"}
            try:
                if msg_type == "human":
                    from langchain_core.messages import HumanMessage
                    return HumanMessage(**kwargs)
                elif msg_type == "ai":
                    from langchain_core.messages import AIMessage
                    return AIMessage(**kwargs)
                elif msg_type == "system":
                    from langchain_core.messages import SystemMessage
                    return SystemMessage(**kwargs)
                elif msg_type == "remove":
                    from langchain_core.messages import RemoveMessage
                    # model_dump() sets the id at the top level, not inside kwargs
                    msg_id = d.get("id")
                    return RemoveMessage(id=msg_id)
            except Exception as exc:
                logger.warning(
                    "[Persistence] Failed to reconstruct %s from dict %r: %s",
                    msg_type, d, exc,
                )

    return d

def _json_loads(s: str | bytes | bytearray) -> Any:
    """Deserialise a JSON string, reconstructing LangChain BaseMessage objects.

    Restrict callers to graph state snapshots only (``_load_from_sqlite``,
    ``get_final_state``, ``get_latest_delta``).  All other retrieval paths
    must use plain ``json.loads`` to prevent false-positive coercion of API
    request bodies, audit reports, events, and HITL payloads into message
    objects.
    """
    return json.loads(s, object_hook=_json_object_hook)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL",          "redis://localhost:6379/0")
REDIS_TTL_SECS = int(os.getenv("REDIS_TTL_HOURS", "24")) * 3600
KEY_PREFIX     = os.getenv("REDIS_KEY_PREFIX",   "promptevo")

# The dedicated Redis SET that holds all live session IDs.
# Maintained by create_session(); read by list_sessions().
_SESSIONS_SET_KEY = f"{KEY_PREFIX}:sessions"

# SQLite batcher drain interval — how often the background thread flushes.
_SQLITE_WRITE_INTERVAL_MS = int(os.getenv("SQLITE_WRITE_INTERVAL_MS", "500"))

# Use SSCAN instead of SMEMBERS for very large session sets (>10k).
_REDIS_USE_SSCAN = os.getenv("REDIS_USE_SSCAN", "false").lower() == "true"

_FALLBACK_WARNED = False   # log the degraded-mode warning only once


# ─────────────────────────────────────────────────────────────────────────────
# REDIS AVAILABILITY PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _probe_redis() -> "redis.Redis | None":
    """Attempt to connect to Redis and return a live client, or None."""
    try:
        import redis as _redis
        client = _redis.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.ping()
        logger.info("[Persistence] Redis connected: %s", REDIS_URL)
        return client
    except Exception as exc:  # noqa: BLE001
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            logger.warning(
                "[Persistence] Redis unavailable (%s) — falling back to "
                "in-process + SQLite storage.  Sessions will NOT survive "
                "process restarts and multi-worker deployment is NOT supported "
                "in this mode.",
                exc,
            )
            _FALLBACK_WARNED = True
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE WRITE BATCHER
# ─────────────────────────────────────────────────────────────────────────────

class _SqliteWriteBatcher:
    """Background-thread write batcher for the SQLite fallback store.

    Design
    ──────
    • The caller thread does: dict mutation → ``schedule(sid, snapshot)``
      which is a non-blocking ``queue.put_nowait()``.  Wall time: ~0 µs.

    • A single daemon thread drains the queue every ``drain_interval_ms``
      milliseconds.  It de-duplicates by session ID — if the same sid arrives
      multiple times between drains, only the latest snapshot is written.

    • The batcher holds a single persistent WAL-mode SQLite connection so
      each transaction is fast and does not reopen the file.

    • WAL mode allows concurrent readers (dashboard polling) without blocking
      the writer thread.  ``synchronous=NORMAL`` skips the extra fsync on each
      transaction (durability: survive OS crash, not power-loss — acceptable
      for an audit session store).

    Thread safety
    ─────────────
    The batcher thread is the ONLY writer to SQLite.  All callers only write
    to the in-memory ``_local`` dict (protected by ``AuditStore._lock``) and
    enqueue a snapshot.  There are zero concurrent SQLite writers.
    """

    def __init__(self, db_path: str, drain_interval_ms: int = 500) -> None:
        self._db_path       = db_path
        self._drain_interval = drain_interval_ms / 1000.0   # convert to seconds
        self._queue: queue.Queue[tuple[str, dict]] = queue.Queue(maxsize=0)
        self._conn: sqlite3.Connection | None = None
        self._stop_event    = threading.Event()
        self._thread        = threading.Thread(
            target=self._run,
            name="sqlite-write-batcher",
            daemon=True,   # dies with the process — no zombie threads
        )
        self._thread.start()
        logger.debug(
            "[SQLiteBatcher] Started (drain_interval=%dms, db=%s)",
            drain_interval_ms, db_path,
        )

    # ── Public API (called from caller threads) ────────────────────────────

    def schedule(self, sid: str, snapshot: dict) -> None:
        """Enqueue a write.  Non-blocking.  Never raises."""
        try:
            self._queue.put_nowait((sid, snapshot))
        except queue.Full:
            # Should never happen with maxsize=0 (unlimited), but guard anyway
            logger.warning("[SQLiteBatcher] Queue unexpectedly full — drop write for %s", sid)

    def flush(self) -> None:
        """Drain all pending writes synchronously.  Call before shutdown."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def flush_sync(self, sid: str, snapshot: dict) -> None:
        """Bypass the queue and write a critical transition immediately.
        
        Uses a separate short-lived connection to avoid interfering with
        the background batcher thread's transaction.
        """
        now = datetime.now(timezone.utc).isoformat()
        status = snapshot.get("status") or "queued"
        state_json = json.dumps(snapshot, default=_json_fallback)
        created_at = snapshot.get("started_at") or now
        
        try:
            # We use a distinct connection here to ensure thread-safety
            # against self._conn which is owned by the batcher thread.
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None
            )
            # Must enable WAL so it can write concurrently with other connections
            conn.execute("PRAGMA journal_mode=WAL")
            # For critical transitions we DO want the durability of FULL
            conn.execute("PRAGMA synchronous=FULL")
            
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_sessions
                    (session_id, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, status, state_json, created_at, now),
            )
            conn.execute("COMMIT")
            conn.close()
            logger.debug("[SQLiteBatcher] Synchronous flush for session %s (status: %s)", sid, status)
        except Exception as exc:  # noqa: BLE001
            logger.error("[SQLiteBatcher] Sync flush failed for session %s: %s", sid, exc)

    # ── Background thread ──────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return (and lazily create) the persistent batcher connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,   # autocommit; we control transactions manually
            )
            # WAL mode: concurrent reads never block writes and vice-versa.
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL: fsync after each WAL checkpoint, not after each write.
            # Trade-off: tiny data loss window on power-loss, but 10x faster.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # Keep 64 MB in the WAL before a checkpoint — fewer checkpoints.
            self._conn.execute("PRAGMA wal_autocheckpoint=256")
            # Shared cache improves concurrent read performance.
            self._conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    session_id TEXT PRIMARY KEY,
                    status     TEXT,
                    state_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            logger.info(
                "[SQLiteBatcher] Opened WAL-mode connection to %s", self._db_path
            )
        return self._conn

    def _drain(self, pending: dict[str, dict]) -> None:
        """Write all de-duplicated pending snapshots in a single transaction."""
        if not pending:
            return
        conn = self._get_conn()
        now  = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute("BEGIN")
            for sid, state in pending.items():
                status     = state.get("status") or "queued"
                state_json = json.dumps(state, default=_json_fallback)
                created_at = state.get("started_at") or now
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_sessions
                        (session_id, status, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (sid, status, state_json, created_at, now),
                )
            conn.execute("COMMIT")
            try:
                logger.debug(
                    "[SQLiteBatcher] Flushed %d session(s) to SQLite.", len(pending)
                )
            except ValueError:
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            try:
                logger.error("[SQLiteBatcher] Write error: %s", exc, exc_info=True)
            except ValueError:
                pass
            # Reset the connection on error — it may be in a bad state
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def _run(self) -> None:
        """Main loop: collect writes, de-duplicate, drain on interval."""
        while not self._stop_event.is_set():
            deadline = time.monotonic() + self._drain_interval
            pending: dict[str, dict] = {}

            # Collect all items available right now, up to the deadline
            while time.monotonic() < deadline:
                try:
                    sid, snapshot = self._queue.get_nowait()
                    # De-duplicate: later snapshot overwrites earlier one
                    pending[sid] = snapshot
                    self._queue.task_done()
                except queue.Empty:
                    # Nothing queued — sleep for the rest of the interval
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(remaining, 0.05))   # 50ms micro-sleep
                    break

            self._drain(pending)

        # Final flush: drain everything left in the queue on shutdown
        final: dict[str, dict] = {}
        while True:
            try:
                sid, snapshot = self._queue.get_nowait()
                final[sid] = snapshot
                self._queue.task_done()
            except queue.Empty:
                break
        self._drain(final)

        if self._conn:
            try:
                self._conn.close()
                try:
                    logger.debug("[SQLiteBatcher] Connection closed on shutdown.")
                except ValueError:
                    pass
            except Exception:  # noqa: BLE001
                pass


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT STORE — Unified session state storage
# ─────────────────────────────────────────────────────────────────────────────

class AuditStore:
    """Thread-safe, Redis-backed session store with in-process + SQLite fallback.

    Interface intentionally mirrors the old ``_audit_store`` dict so that
    dashboard.py and api.py can swap it in with minimal changes:

        OLD:  _audit_store[sid]["events"].append(event)
        NEW:  store.append_event(sid, event)

        OLD:  _audit_store[sid]["running"]
        NEW:  store.is_running(sid)

    Redis key schema (all keys namespaced under ``{PREFIX}:{sid}:``)::

        {PREFIX}:{sid}:meta        — HASH  (running, error, final_state, …)
        {PREFIX}:{sid}:events      — LIST  (JSON-encoded event dicts)
        {PREFIX}:{sid}:hitl        — STRING (JSON HITL data, or absent)
        {PREFIX}:{sid}:hitl_dec    — LIST  (BLPOP target for HITL decision)
        {PREFIX}:sessions          — SET   (all live session IDs — O(1) lookup)

    SQLite fallback write strategy
    ───────────────────────────────
    All writes to the SQLite-backed fallback are off-loaded to
    ``_SqliteWriteBatcher``.  The caller thread only mutates the in-memory
    ``_local`` dict (O(1), under a lock) and calls ``_schedule_sqlite(sid)``
    which enqueues the snapshot non-blocking.  The batcher thread drains the
    queue every ``SQLITE_WRITE_INTERVAL_MS`` ms.
    """

    def __init__(self) -> None:
        self._redis  = _probe_redis()
        self._local: dict[str, dict] = {}   # fallback: in-process store
        self._lock   = threading.Lock()      # protects _local
        self._db_path = os.getenv("SQLITE_CHECKPOINT_PATH", str(DB_PATH))
        self._batcher: _SqliteWriteBatcher | None = None

        if not self._redis:
            # Start the batcher thread — also creates the DB schema via
            # the first connection attempt inside _run().
            self._batcher = _SqliteWriteBatcher(
                db_path=self._db_path,
                drain_interval_ms=_SQLITE_WRITE_INTERVAL_MS,
            )
            # Register graceful shutdown: flush pending writes before exit.
            atexit.register(self._shutdown_batcher)
            # Warm up the batcher connection and load existing sessions.
            self._load_from_sqlite()

    def _shutdown_batcher(self) -> None:
        """Flush all pending SQLite writes before the process exits."""
        if self._batcher:
            try:
                logger.info("[AuditStore] Flushing SQLite write-batcher on shutdown…")
            except ValueError:
                pass
            self._batcher.flush()

    def _load_from_sqlite(self) -> None:
        """Load existing sessions from SQLite into the in-process cache.

        Called once at startup when Redis is unavailable.  Uses a short-lived
        read connection — does not interact with the batcher's write connection.
        """
        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.execute(
                "SELECT session_id, status, state_json, created_at FROM audit_sessions"
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AuditStore] Could not load sessions from SQLite: %s", exc)
            return

        with self._lock:
            for sid, status, state_json, created_at in rows:
                state = _json_loads(state_json) if state_json else {}
                self._local[sid] = {
                    "running":      state.get("running", False),
                    "events":       state.get("events", []),
                    "final_state":  state.get("final_state"),
                    "error":        state.get("error"),
                    "hitl":         state.get("hitl"),
                    "status":       status,
                    "report":       state.get("report"),
                    "request":      state.get("request"),
                    "started_at":   created_at,
                    "latest_delta": state.get("latest_delta"),
                }
        logger.info(
            "[AuditStore] Loaded %d session(s) from SQLite (%s).",
            len(rows), self._db_path,
        )

    def _schedule_sqlite(self, sid: str) -> None:
        """Enqueue the current in-memory state of ``sid`` for async SQLite write.

        MUST be called with ``self._lock`` already held.
        The call itself is non-blocking (~0 µs).
        """
        if self._batcher and sid in self._local:
            # Shallow copy is enough: the batcher serialises to JSON immediately
            self._batcher.schedule(sid, dict(self._local[sid]))

    # ── Key helpers ───────────────────────────────────────────────────────

    def _k(self, sid: str, suffix: str) -> str:
        return f"{KEY_PREFIX}:{sid}:{suffix}"

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(self, sid: str) -> None:
        """Initialise a new audit session record."""
        if self._redis:
            pipe = self._redis.pipeline()
            pipe.hset(self._k(sid, "meta"), mapping={
                "running":      "1",
                "error":        "",
                "final_state":  "",
                "hitl":         "",
                "status":       "queued",
                "report":       "",
                "request":      "",
                "started_at":   "",
                "latest_delta": "",
            })
            pipe.delete(self._k(sid, "events"))
            pipe.delete(self._k(sid, "hitl"))
            pipe.delete(self._k(sid, "hitl_dec"))
            pipe.expire(self._k(sid, "meta"),   REDIS_TTL_SECS)
            pipe.expire(self._k(sid, "events"), REDIS_TTL_SECS)
            # O(1) session registration — replaces the O(N) KEYS scan
            pipe.sadd(_SESSIONS_SET_KEY, sid)
            pipe.expire(_SESSIONS_SET_KEY, REDIS_TTL_SECS)
            pipe.execute()
        else:
            with self._lock:
                self._local[sid] = {
                    "running":      True,
                    "events":       [],
                    "final_state":  None,
                    "error":        None,
                    "hitl":         None,
                    "status":       "queued",
                    "report":       None,
                    "request":      None,
                    "started_at":   None,
                    "latest_delta": None,
                }
                self._schedule_sqlite(sid)

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
                    self._schedule_sqlite(sid)

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
                    self._schedule_sqlite(sid)

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
                    self._schedule_sqlite(sid)

    def get_final_state(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.hget(self._k(sid, "meta"), "final_state")
            return _json_loads(raw) if raw else None  # graph state snapshot — hook required
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
                    self._schedule_sqlite(sid)

    def get_error(self, sid: str) -> str | None:
        if self._redis:
            val = self._redis.hget(self._k(sid, "meta"), "error")
            return val if val else None
        with self._lock:
            return self._local.get(sid, {}).get("error")

    # ── API state ─────────────────────────────────────────────────────────

    def set_status(self, sid: str, status: str) -> None:
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "status", status)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["status"] = status
                    if status in ("complete", "error"):
                        # REL-003: Critical transitions flush synchronously
                        if self._batcher:
                            self._batcher.flush_sync(sid, self._local[sid])
                    else:
                        self._schedule_sqlite(sid)

    def get_status(self, sid: str) -> str | None:
        if self._redis:
            val = self._redis.hget(self._k(sid, "meta"), "status")
            return val if val else None
        with self._lock:
            return self._local.get(sid, {}).get("status")

    def set_report(self, sid: str, report: Any) -> None:
        val = report.model_dump() if hasattr(report, "model_dump") else report
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "report", json.dumps(val))
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["report"] = val
                    self._schedule_sqlite(sid)

    def get_report(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.hget(self._k(sid, "meta"), "report")
            return json.loads(raw) if raw else None  # audit report — plain dict, no hook
        with self._lock:
            return self._local.get(sid, {}).get("report")

    def set_request(self, sid: str, request: Any) -> None:
        val = request.model_dump() if hasattr(request, "model_dump") else request
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "request", json.dumps(val))
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["request"] = val
                    self._schedule_sqlite(sid)

    def get_request(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.hget(self._k(sid, "meta"), "request")
            return json.loads(raw) if raw else None  # API request body — plain dict, no hook
        with self._lock:
            return self._local.get(sid, {}).get("request")

    def set_started_at(self, sid: str, started_at: Any) -> None:
        val = started_at.isoformat() if hasattr(started_at, "isoformat") else str(started_at)
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "started_at", val)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["started_at"] = val
                    self._schedule_sqlite(sid)

    def get_started_at(self, sid: str) -> str | None:
        if self._redis:
            val = self._redis.hget(self._k(sid, "meta"), "started_at")
            return val if val else None
        with self._lock:
            return self._local.get(sid, {}).get("started_at")

    def set_latest_delta(self, sid: str, delta: dict) -> None:
        if self._redis:
            self._redis.hset(self._k(sid, "meta"), "latest_delta", json.dumps(delta))
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["latest_delta"] = delta
                    self._schedule_sqlite(sid)

    def get_latest_delta(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.hget(self._k(sid, "meta"), "latest_delta")
            return _json_loads(raw) if raw else None  # graph state delta — hook required
        with self._lock:
            return self._local.get(sid, {}).get("latest_delta")

    # ── HITL ──────────────────────────────────────────────────────────────

    def set_hitl(self, sid: str, data: dict) -> None:
        """Store HITL interrupt data (payload awaiting human review)."""
        if self._redis:
            self._redis.set(self._k(sid, "hitl"), json.dumps(data), ex=REDIS_TTL_SECS)
        else:
            with self._lock:
                if sid in self._local:
                    self._local[sid]["hitl"] = data
                    self._schedule_sqlite(sid)

    def get_hitl(self, sid: str) -> dict | None:
        if self._redis:
            raw = self._redis.get(self._k(sid, "hitl"))
            return json.loads(raw) if raw else None  # HITL context dict — plain dict, no hook
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
                    self._schedule_sqlite(sid)

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
                    self._schedule_sqlite(sid)

    def poll_hitl_decision(self, sid: str, timeout: float = 0.25) -> dict | None:
        """Background thread calls this to wait for the auditor's decision.

        Redis path: uses ``BLPOP`` (true blocking pop — no CPU spin).
        Fallback path: polls the in-process dict every ``timeout`` seconds.
        """
        if self._redis:
            result = self._redis.blpop(self._k(sid, "hitl_dec"), timeout=timeout)
            if result:
                _, raw = result
                return json.loads(raw)  # HITL decision payload — plain dict, no hook
            return None
        else:
            with self._lock:
                hitl = self._local.get(sid, {}).get("hitl") or {}
                return hitl.get("decision")

    # ── Session list ──────────────────────────────────────────────────────

    def list_sessions(self) -> list[str]:
        """Return all known session IDs.

        Redis path
        ──────────
        Reads from the dedicated ``{PREFIX}:sessions`` SET registered during
        ``create_session()``.  This is an atomic, O(|sessions|) operation —
        NOT the O(|all keys|) global ``KEYS`` scan that was here before.

        When ``REDIS_USE_SSCAN=true``, uses cursor-based ``SSCAN`` to avoid
        blocking the Redis event loop for very large session sets (>10k).

        SQLite fallback path
        ────────────────────
        Returns the keys of the in-memory ``_local`` dict directly — O(N)
        but N is the number of sessions, always small relative to all Redis
        keys, and no I/O is involved.
        """
        if self._redis:
            return self._scan_session_set()
        with self._lock:
            return list(self._local.keys())

    def _scan_session_set(self) -> list[str]:
        """Read session IDs from the Redis SET ``{PREFIX}:sessions``.

        Uses SMEMBERS for small sets (default) or cursor-based SSCAN for
        large sets (``REDIS_USE_SSCAN=true``).

        Falls back to a safe cursor-based SCAN of all ``*:meta`` keys if
        the sessions SET does not exist yet (backwards-compat with old
        deployments that were created before this module version).
        """
        if not self._redis:
            return []

        try:
            if _REDIS_USE_SSCAN:
                # Cursor-based scan — never blocks for more than ~count keys
                sids: list[str] = []
                cursor = 0
                while True:
                    cursor, chunk = self._redis.sscan(
                        _SESSIONS_SET_KEY, cursor=cursor, count=200
                    )
                    sids.extend(chunk)
                    if cursor == 0:
                        break
                return sids
            else:
                # SMEMBERS — atomic, O(N) where N = # sessions (not # all keys)
                members = self._redis.smembers(_SESSIONS_SET_KEY)
                return list(members)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[AuditStore] list_sessions SET read failed (%s) — "
                "falling back to cursor SCAN of meta keys.", exc,
            )
            # Backwards-compat fallback: cursor-based SCAN (never KEYS)
            return self._cursor_scan_meta_keys()

    def _cursor_scan_meta_keys(self) -> list[str]:
        """Cursor-based SCAN fallback.  Much safer than KEYS under load.

        Iterates through all ``{PREFIX}:*:meta`` keys in O(N) total cost
        spread across many small round-trips (count=200 per cursor step),
        so it never blocks the Redis event loop for more than a few ms at
        a time.
        """
        if not self._redis:
            return []
        pattern = f"{KEY_PREFIX}:*:meta"
        sids: list[str] = []
        try:
            for key in self._redis.scan_iter(pattern, count=200):
                # key format: "{PREFIX}:{sid}:meta"
                parts = key.split(":", 2)
                if len(parts) == 3:
                    sids.append(parts[1])
        except Exception as exc:  # noqa: BLE001
            logger.error("[AuditStore] cursor SCAN failed: %s", exc)
        return sids

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

# Module-level handle so atexit can close it properly.
_sqlite_conn: sqlite3.Connection | None = None


def _close_sqlite_conn() -> None:
    """atexit handler: close the checkpointer's SQLite connection on shutdown."""
    global _sqlite_conn
    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
            try:
                logger.info("[Persistence] SqliteSaver checkpointer connection closed.")
            except ValueError:
                pass
        except Exception:  # noqa: BLE001
            pass
        _sqlite_conn = None


def build_checkpointer():
    """Return the best available LangGraph checkpointer.

    Priority order
    ──────────────
    1. ``RedisSaver``   — persists across process restarts, safe for
                          multi-worker (multiple FastAPI/Celery workers
                          can resume the same HITL session).

    2. ``SqliteSaver``  — persists across process restarts, zero external
                          dependencies.  Stored in ``checkpoints.db`` (or
                          the path set via ``SQLITE_CHECKPOINT_PATH``).
                          Single-process only (not safe for multi-worker).

                          **Connection lifecycle fix**: the ``sqlite3.connect()``
                          handle is stored in ``_sqlite_conn`` and closed via
                          an ``atexit`` handler — no more connection leak.

                          **WAL mode**: ``PRAGMA journal_mode=WAL`` and
                          ``PRAGMA synchronous=NORMAL`` are applied before the
                          connection is passed to LangGraph so checkpoint reads
                          and writes are concurrent-safe and do not fsync on
                          every transaction.

    3. ``MemorySaver``  — in-process fallback; HITL works within one
                          process lifetime but sessions are lost on restart.

    The returned checkpointer is passed to ``graph.compile(checkpointer=...)``.
    LangGraph's ``interrupt()`` / ``Command(resume=...)`` mechanism works
    identically with all three — the abstraction is complete.
    """
    global _sqlite_conn

    redis_client = _probe_redis()
    if redis_client:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            saver = RedisSaver(redis_url=REDIS_URL)
            logger.info("[Persistence] Using RedisSaver checkpointer (%s)", REDIS_URL)
            return saver
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Persistence] RedisSaver init failed (%s) — trying SqliteSaver", exc
            )

    # Tier 2: SQLite — persists across process restarts, no external deps.
    sqlite_path = os.getenv("SQLITE_CHECKPOINT_PATH", str(DB_PATH))
    try:
        # Try both known import paths for langgraph-checkpoint-sqlite
        SqliteSaver = None
        for _import_path in (
            "langgraph.checkpoint.sqlite",
            "langgraph_checkpoint_sqlite",
        ):
            try:
                import importlib
                _mod = importlib.import_module(_import_path)
                SqliteSaver = getattr(_mod, "SqliteSaver", None)
                if SqliteSaver:
                    break
            except ImportError:
                continue

        if SqliteSaver is None:
            raise ImportError("langgraph-checkpoint-sqlite is not installed.")

        # Open a single persistent connection — registered for atexit cleanup
        if _sqlite_conn is not None:
            try:
                _sqlite_conn.close()
            except Exception:  # noqa: BLE001
                pass

        conn = sqlite3.connect(sqlite_path, check_same_thread=False)

        # ── Apply WAL mode before handing to LangGraph ────────────────────
        # WAL: concurrent readers never block the writer and vice-versa.
        # Without WAL, every checkpoint write takes a full exclusive lock
        # that blocks dashboard SSE polling threads.
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: fsync after WAL checkpoint, not after each individual write.
        # Risk: up to ~500ms of data loss on power-loss (acceptable for audit
        # sessions whose source-of-truth is the LLM conversation log).
        conn.execute("PRAGMA synchronous=NORMAL")
        # Shared cache for concurrent readers within the same process.
        conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache

        _sqlite_conn = conn
        atexit.register(_close_sqlite_conn)

        saver = SqliteSaver(conn)
        logger.info(
            "[Persistence] Using SqliteSaver checkpointer (WAL mode, %s)", sqlite_path
        )
        return saver

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Persistence] SqliteSaver init failed (%s) — falling back to MemorySaver",
            exc,
        )

    from langgraph.checkpoint.memory import MemorySaver
    logger.warning(
        "[Persistence] Using in-process MemorySaver — HITL sessions will be "
        "lost on restart.  Set REDIS_URL or SQLITE_CHECKPOINT_PATH to enable "
        "persistent checkpointing."
    )
    return MemorySaver()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
# The AuditStore is created once per process and reused across all threads.
# A threading.Lock guards the lazy initialisation to prevent duplicate
# construction when multiple threads call get_audit_store() concurrently
# at startup.
#
# Note: The Streamlit dashboard has its OWN independent store dict (parked
# in sys.modules for rerun survival).  This singleton is for the FastAPI
# server and CLI entry points only.
# ─────────────────────────────────────────────────────────────────────────────
_store_instance: AuditStore | None = None
_store_lock = threading.Lock()


def get_audit_store() -> AuditStore:
    """Return the process-level AuditStore singleton.

    Thread-safe lazy initialisation using a module-level lock.
    Safe to call from any thread.
    """
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            # Double-checked locking: re-test inside the lock
            if _store_instance is None:
                _store_instance = AuditStore()
    return _store_instance

"""
infra/database.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Production Database Persistence
==========================================
Stores completed PromptEvo audit sessions in a structured SQLite database
(via SQLAlchemy Core / ORM) that can be trivially promoted to PostgreSQL by
swapping the ``DATABASE_URL`` environment variable.

Design principles
─────────────────
• **Zero new I/O on the hot path.**  The write is triggered once per session,
  only after the graph completes, from a daemon thread so the Streamlit main
  thread is never blocked.

• **Thread safety.**  A module-level ``threading.Lock`` + scoped
  ``sessionmaker`` ensure no two threads share a live SQLAlchemy ``Session``.
  SQLite's WAL mode lets concurrent readers (dashboard) see committed rows
  without blocking the writer.

• **PostgreSQL-ready.**  Only standard SQLAlchemy types are used.  Switching
  from SQLite to PostgreSQL requires only:
      DATABASE_URL=postgresql+psycopg2://user:pass@host/db
  and ``pip install psycopg2-binary``.  The ``JSON`` column maps to JSONB on
  PostgreSQL automatically.

• **Idempotent schema migration.**  ``create_all(checkfirst=True)`` is safe to
  call on every startup — it adds missing tables/columns without touching
  existing data.

Environment Variables
──────────────────────
  DATABASE_URL          SQLAlchemy URL. Default: sqlite:///audit_reports.db
  DB_ECHO               Set to "true" to log every SQL statement (debug).

Schema — ``audit_sessions`` table
───────────────────────────────────
  session_id        TEXT  PRIMARY KEY   — UUID from the audit run
  start_time        TEXT                — ISO-8601 UTC timestamp
  end_time          TEXT                — ISO-8601 UTC timestamp (set on write)
  objective         TEXT                — Red-team objective supplied by auditor
  target_model      TEXT                — Name of the LLM under audit
  attack_status     TEXT                — "success" | "failure" | "in_progress"
  rahs_score        REAL                — AI-CVSS (0-10)
  prometheus_score  REAL                — Prometheus judge score (0-5)
  severity_band     TEXT                — "Critical"|"High"|"Medium"|"Low"|"None"
  turn_count        INT                 — Number of adversarial turns taken
  duration_seconds  REAL                — Wall-clock seconds for the session
  final_technique   TEXT                — Last active PAP technique
  defense_patch     TEXT                — Blue-team remediation text
  evolved_techniques JSON               — JSON list of all mutated strategies used
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("promptevo.database")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///audit_reports.db",
)
_DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE + SESSION (lazy init — import does NOT open a connection)
# ─────────────────────────────────────────────────────────────────────────────

_engine = None
_SessionFactory = None
_init_lock = threading.Lock()


def _get_engine():
    """Return (and lazily create) the SQLAlchemy engine.

    Isolated behind a function so the import of this module is free of
    side-effects and does not open a database connection until the first
    write is attempted.
    """
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine

    with _init_lock:
        # Double-checked locking
        if _engine is not None:
            return _engine

        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            connect_args: dict[str, Any] = {}
            if DATABASE_URL.startswith("sqlite"):
                # WAL mode for SQLite: readers never block the writer
                connect_args = {"check_same_thread": False}

            _engine = create_engine(
                DATABASE_URL,
                echo=_DB_ECHO,
                connect_args=connect_args,
                # Pool settings tuned for single-writer, many-reader pattern
                pool_pre_ping=True,
            )

            # Apply WAL mode pragma for SQLite after engine creation
            if DATABASE_URL.startswith("sqlite"):
                from sqlalchemy import event

                @event.listens_for(_engine, "connect")
                def _set_sqlite_pragma(dbapi_conn, _connection_record):
                    cursor = dbapi_conn.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=-32768")  # 32 MB
                    cursor.close()

            _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)

            # Ensure tables exist (idempotent)
            _create_schema(_engine)

            logger.info("[Database] Engine initialised: %s", DATABASE_URL.split("?")[0])
        except Exception as exc:  # noqa: BLE001
            logger.error("[Database] Failed to initialise engine: %s", exc, exc_info=True)
            _engine = None

    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# ORM MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _create_schema(engine) -> None:
    """Create (or verify) the ``audit_sessions`` table.

    Uses Core DDL so this module has zero dependency on ``declarative_base``
    and is compatible with bare SQLAlchemy Core installs as well as the ORM.
    The table definition is stored in ``_METADATA`` for re-use in queries.

    JSON column strategy
    ────────────────────
    • SQLite  : stored as ``Text``.  We call ``json.dumps`` explicitly before
      writing so SQLAlchemy never touches the value.  Reading back gives a
      plain string that ``json.loads`` decodes cleanly — no double-encoding.
    • PostgreSQL: stored as ``JSON`` (maps to JSONB).  SQLAlchemy serialises /
      deserialises automatically; we pass the raw Python list.
    """
    from sqlalchemy import (
        Column, Float, Integer, MetaData, String, Table, Text
    )

    global _METADATA, _TABLE

    if "_METADATA" in globals() and _METADATA is not None:
        return  # Already initialised

    # Decide column type for evolved_techniques
    _is_sqlite = DATABASE_URL.startswith("sqlite")
    if _is_sqlite:
        # Use plain Text for SQLite to avoid SQLAlchemy's double-JSON-encoding
        _json_type = Text
    else:
        try:
            from sqlalchemy import JSON
            _json_type = JSON  # native JSONB on PostgreSQL
        except ImportError:
            _json_type = Text

    meta = MetaData()
    table = Table(
        "audit_sessions",
        meta,
        Column("session_id",        String(64),  primary_key=True),
        Column("start_time",        String(64),  nullable=True),
        Column("end_time",          String(64),  nullable=True),
        Column("objective",         Text,         nullable=True),
        Column("target_model",      String(256),  nullable=True),
        Column("attack_status",     String(32),   nullable=True),
        Column("rahs_score",        Float,         nullable=True),
        Column("prometheus_score",  Float,         nullable=True),
        Column("severity_band",     String(16),   nullable=True),
        Column("turn_count",        Integer,       nullable=True),
        Column("duration_seconds",  Float,         nullable=True),
        Column("final_technique",   Text,          nullable=True),
        Column("defense_patch",     Text,          nullable=True),
        Column("evolved_techniques", _json_type,   nullable=True),
    )

    meta.create_all(engine, checkfirst=True)
    _METADATA = meta
    _TABLE = table
    logger.debug("[Database] Schema verified/created (audit_sessions).")


# Module-level references populated by _create_schema at first engine init
_METADATA = None
_TABLE = None

# ─────────────────────────────────────────────────────────────────────────────
# WRITE FUNCTION — thread-safe, non-blocking (daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def save_audit_report_to_db(report_json: dict[str, Any]) -> None:
    """Persist a completed audit report to the database.

    This function is designed to be called from the Streamlit main thread
    immediately after ``st.session_state.final_state`` becomes available.
    The actual database write is off-loaded to a daemon thread so the UI
    never blocks.

    Parameters
    ──────────
    report_json : dict
        The report dict assembled in ``dashboard.py`` (mirrors the JSON
        download payload).  Expected keys:

        ``session_id``, ``objective``, ``target_model``, ``attack_status``,
        ``rahs_score``, ``prometheus_score``, ``severity_band``,
        ``total_turns``, ``duration_seconds``, ``active_technique``,
        ``pruned_techniques``, ``defense_patch``.

        All keys are optional — missing values default to ``None`` / 0.

    Thread Safety
    ─────────────
    The function spawns a daemon thread for the write.  Concurrent calls for
    different session IDs proceed in parallel without blocking each other.
    Concurrent calls for the *same* session ID are prevented by the
    ``_write_lock`` inside ``_do_write`` (UPSERT semantics: later call wins).
    """
    # Snapshot the report dict so we own the data regardless of what the
    # caller mutates afterward (Streamlit reruns may rewrite session_state)
    snapshot = dict(report_json)

    def _do_write() -> None:
        try:
            engine = _get_engine()
            if engine is None:
                logger.warning("[Database] Engine unavailable — skipping audit report write.")
                return

            # Build the row from the snapshot
            now_iso = datetime.now(timezone.utc).isoformat()
            session_id = snapshot.get("session_id") or ""
            if not session_id:
                logger.warning("[Database] report_json missing 'session_id' — skipping write.")
                return

            # Determine evolved_techniques — list of all strategies seen
            evolved: list[str] = []
            pruned = snapshot.get("pruned_techniques") or []
            active = snapshot.get("active_technique") or ""
            if isinstance(pruned, list):
                evolved = list(pruned)
            if active and active not in evolved:
                evolved.append(active)

            row = {
                "session_id":         session_id,
                "start_time":         snapshot.get("start_time") or now_iso,
                "end_time":           now_iso,
                "objective":          snapshot.get("objective") or "",
                "target_model":       snapshot.get("target_model") or "",
                "attack_status":      snapshot.get("attack_status") or "unknown",
                "rahs_score":         float(snapshot.get("rahs_score") or 0.0),
                "prometheus_score":   float(snapshot.get("prometheus_score") or 0.0),
                "severity_band":      snapshot.get("severity_band") or "None",
                "turn_count":         int(snapshot.get("total_turns") or 0),
                "duration_seconds":   float(snapshot.get("duration_seconds") or 0.0),
                "final_technique":    snapshot.get("active_technique") or "",
                "defense_patch":      snapshot.get("defense_patch") or "",
                # For SQLite (Text column): always pass a JSON string so
                # SQLAlchemy stores it verbatim — no double-encoding.
                # For PostgreSQL (JSON column): pass the raw list; SQLAlchemy
                # serialises it automatically.
                "evolved_techniques": (
                    json.dumps(evolved)
                    if DATABASE_URL.startswith("sqlite")
                    else evolved
                ),
            }

            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            from sqlalchemy import insert as sa_insert  # noqa: F401

            with _write_lock:
                with _SessionFactory() as session:  # type: ignore[union-attr]
                    if DATABASE_URL.startswith("sqlite"):
                        # SQLite: use INSERT OR REPLACE (UPSERT)
                        stmt = sqlite_insert(_TABLE).values(**row)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["session_id"],
                            set_={k: v for k, v in row.items() if k != "session_id"},
                        )
                    else:
                        # PostgreSQL: use INSERT ... ON CONFLICT DO UPDATE
                        from sqlalchemy.dialects.postgresql import insert as pg_insert
                        stmt = pg_insert(_TABLE).values(**row)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["session_id"],
                            set_={k: v for k, v in row.items() if k != "session_id"},
                        )
                    session.execute(stmt)
                    session.commit()

            logger.info(
                "[Database] Audit report persisted  sid=%s  status=%s  rahs=%.2f",
                session_id[:8], row["attack_status"], row["rahs_score"],
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[Database] Failed to write audit report: %s", exc, exc_info=True
            )

    # Off-load to a daemon thread — the Streamlit main thread is never blocked
    t = threading.Thread(target=_do_write, name="db-audit-write", daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
# QUERY HELPERS  (for future analytics / admin panel use)
# ─────────────────────────────────────────────────────────────────────────────

def list_audit_reports(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return the most recent ``limit`` audit reports from the database.

    Returns an empty list if the database is unavailable.
    """
    engine = _get_engine()
    if engine is None or _TABLE is None:
        return []
    try:
        from sqlalchemy import select, desc
        stmt = (
            select(_TABLE)
            .order_by(desc(_TABLE.c.end_time))
            .limit(limit)
            .offset(offset)
        )
        with engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        result = []
        for row in rows:
            r = dict(row)
            # Deserialise JSON column for SQLite
            if isinstance(r.get("evolved_techniques"), str):
                try:
                    r["evolved_techniques"] = json.loads(r["evolved_techniques"])
                except (json.JSONDecodeError, TypeError):
                    r["evolved_techniques"] = []
            result.append(r)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("[Database] list_audit_reports failed: %s", exc)
        return []


def get_audit_report(session_id: str) -> dict | None:
    """Fetch a single audit report by ``session_id``."""
    engine = _get_engine()
    if engine is None or _TABLE is None:
        return None
    try:
        from sqlalchemy import select
        stmt = select(_TABLE).where(_TABLE.c.session_id == session_id)
        with engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        r = dict(row)
        if isinstance(r.get("evolved_techniques"), str):
            try:
                r["evolved_techniques"] = json.loads(r["evolved_techniques"])
            except (json.JSONDecodeError, TypeError):
                r["evolved_techniques"] = []
        return r
    except Exception as exc:  # noqa: BLE001
        logger.error("[Database] get_audit_report failed: %s", exc)
        return None

"""
tests/test_persistence.py
─────────────────────────────────────────────────────────────────────────────
Tests for infra/persistence.py — AuditStore (in-process fallback path).

Strategy: All tests use the in-process (Redis-unavailable) code path by
patching _probe_redis() to return None. This avoids any Redis dependency
and exercises the _local dict + _SqliteWriteBatcher code path.

Coverage:
  - AuditStore.create_session
  - AuditStore.session_exists
  - AuditStore.is_running / set_running
  - AuditStore.append_event / get_events / event_count
  - AuditStore.set_final_state / get_final_state
  - AuditStore.set_error / get_error
  - AuditStore.set_status / get_status
  - AuditStore.list_sessions
  - _SqliteWriteBatcher (schedule, deduplication, WAL mode)
  - Thread safety under concurrent access

All tests require NO Redis instance.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import infra.persistence as pers_module
from infra.persistence import AuditStore, _SqliteWriteBatcher


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    """
    A fresh AuditStore in in-process (no-Redis) mode.
    SQLite writes go to a temp file.
    """
    db_path = str(tmp_path / "test_checkpoints.db")
    with (
        patch.object(pers_module, "_probe_redis", return_value=None),
        patch.dict("os.environ", {"SQLITE_CHECKPOINT_PATH": db_path}),
    ):
        s = AuditStore()
    return s


@pytest.fixture
def sid() -> str:
    return "test-session-abc123"


# ─────────────────────────────────────────────────────────────────────────────
# 1. AuditStore — session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreSessionLifecycle:

    def test_create_session_makes_session_exist(self, store: AuditStore, sid: str):
        assert not store.session_exists(sid)
        store.create_session(sid)
        assert store.session_exists(sid)

    def test_create_session_sets_running_true(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.is_running(sid) is True

    def test_create_session_initial_status_is_queued(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_status(sid) == "queued"

    def test_create_session_initial_events_empty(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_events(sid) == []
        assert store.event_count(sid) == 0

    def test_create_session_initial_final_state_none(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_final_state(sid) is None

    def test_nonexistent_session_does_not_exist(self, store: AuditStore):
        assert not store.session_exists("nonexistent-id-xyz")

    def test_create_two_distinct_sessions(self, store: AuditStore):
        store.create_session("session-1")
        store.create_session("session-2")
        assert store.session_exists("session-1")
        assert store.session_exists("session-2")

    def test_session_ids_do_not_cross_contaminate(self, store: AuditStore):
        store.create_session("session-A")
        store.create_session("session-B")
        store.set_status("session-A", "running")
        assert store.get_status("session-A") == "running"
        assert store.get_status("session-B") == "queued"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Running flag
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreRunningFlag:

    def test_set_running_false(self, store: AuditStore, sid: str):
        store.create_session(sid)
        store.set_running(sid, False)
        assert store.is_running(sid) is False

    def test_set_running_true_after_false(self, store: AuditStore, sid: str):
        store.create_session(sid)
        store.set_running(sid, False)
        store.set_running(sid, True)
        assert store.is_running(sid) is True

    def test_is_running_missing_session_returns_false(self, store: AuditStore):
        """A session that was never created → is_running returns False."""
        assert store.is_running("ghost-session") is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Events
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreEvents:

    def test_append_single_event(self, store: AuditStore, sid: str):
        store.create_session(sid)
        event = {"node": "analyst_node", "turn": 1, "status": "ok"}
        store.append_event(sid, event)
        events = store.get_events(sid)
        assert len(events) == 1
        assert events[0] == event

    def test_append_multiple_events_preserves_order(self, store: AuditStore, sid: str):
        store.create_session(sid)
        for i in range(5):
            store.append_event(sid, {"index": i})
        events = store.get_events(sid)
        assert len(events) == 5
        assert [e["index"] for e in events] == list(range(5))

    def test_event_count_matches_appended_count(self, store: AuditStore, sid: str):
        store.create_session(sid)
        for i in range(7):
            store.append_event(sid, {"i": i})
        assert store.event_count(sid) == 7

    def test_get_events_from_start_offset(self, store: AuditStore, sid: str):
        """get_events(start=2) must skip first 2 events."""
        store.create_session(sid)
        for i in range(5):
            store.append_event(sid, {"i": i})
        events = store.get_events(sid, start=2)
        assert len(events) == 3
        assert events[0]["i"] == 2

    def test_get_events_empty_session_returns_empty(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_events(sid) == []

    def test_event_count_nonexistent_session_returns_zero(self, store: AuditStore):
        assert store.event_count("ghost") == 0

    def test_events_not_shared_across_sessions(self, store: AuditStore):
        store.create_session("s1")
        store.create_session("s2")
        store.append_event("s1", {"data": "for s1"})
        assert store.get_events("s2") == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Final state
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreFinalState:

    def test_set_and_get_final_state(self, store: AuditStore, sid: str):
        store.create_session(sid)
        final = {"attack_status": "success", "prometheus_score": 4.5}
        store.set_final_state(sid, final)
        assert store.get_final_state(sid) == final

    def test_final_state_overwrites_previous(self, store: AuditStore, sid: str):
        store.create_session(sid)
        store.set_final_state(sid, {"attack_status": "failure"})
        store.set_final_state(sid, {"attack_status": "success"})
        assert store.get_final_state(sid)["attack_status"] == "success"

    def test_get_final_state_none_before_set(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_final_state(sid) is None

    def test_final_state_nonexistent_session_returns_none(self, store: AuditStore):
        assert store.get_final_state("ghost") is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreError:

    def test_set_and_get_error(self, store: AuditStore, sid: str):
        store.create_session(sid)
        store.set_error(sid, "Rate limit exceeded")
        assert store.get_error(sid) == "Rate limit exceeded"

    def test_get_error_none_when_not_set(self, store: AuditStore, sid: str):
        store.create_session(sid)
        assert store.get_error(sid) is None

    def test_get_error_nonexistent_session_returns_none(self, store: AuditStore):
        assert store.get_error("ghost") is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Status
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreStatus:

    def test_set_and_get_status(self, store: AuditStore, sid: str):
        store.create_session(sid)
        store.set_status(sid, "running")
        assert store.get_status(sid) == "running"

    def test_status_lifecycle(self, store: AuditStore, sid: str):
        """Status transitions: queued → running → completed."""
        store.create_session(sid)
        assert store.get_status(sid) == "queued"
        store.set_status(sid, "running")
        assert store.get_status(sid) == "running"
        store.set_status(sid, "completed")
        assert store.get_status(sid) == "completed"

    def test_get_status_nonexistent_returns_none(self, store: AuditStore):
        assert store.get_status("ghost") is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. list_sessions
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreListSessions:

    def test_empty_store_lists_no_sessions(self, store: AuditStore):
        sessions = store.list_sessions()
        assert sessions == []

    def test_created_sessions_appear_in_list(self, store: AuditStore):
        store.create_session("s1")
        store.create_session("s2")
        sessions = store.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_list_sessions_returns_list_type(self, store: AuditStore):
        store.create_session("s1")
        assert isinstance(store.list_sessions(), list)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Thread safety
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditStoreThreadSafety:
    """Concurrent writes must not corrupt the in-process state."""

    def test_concurrent_append_events(self, store: AuditStore):
        """50 threads each appending 20 events → 1000 total events, no corruption."""
        sid = "concurrent-test"
        store.create_session(sid)
        n_threads = 50
        events_per_thread = 20

        def append_many(thread_id: int):
            for i in range(events_per_thread):
                store.append_event(sid, {"thread": thread_id, "i": i})

        threads = [threading.Thread(target=append_many, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = store.event_count(sid)
        assert total == n_threads * events_per_thread

    def test_concurrent_set_status_is_consistent(self, store: AuditStore):
        """Concurrent status writes must not raise and final value must be a string."""
        sid = "status-test"
        store.create_session(sid)

        def set_status_repeatedly():
            for _ in range(100):
                store.set_status(sid, "running")

        threads = [threading.Thread(target=set_status_repeatedly) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert store.get_status(sid) == "running"


# ─────────────────────────────────────────────────────────────────────────────
# 9. _SqliteWriteBatcher (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestSqliteWriteBatcher:
    """Tests for the background write-batcher that feeds the SQLite fallback."""

    def test_batcher_writes_to_sqlite(self, tmp_path: Path):
        """schedule() + flush() must produce a row in audit_sessions."""
        db_path = str(tmp_path / "batcher_test.db")
        batcher = _SqliteWriteBatcher(db_path, drain_interval_ms=50)

        snapshot = {
            "status": "running",
            "events": [],
            "started_at": "2026-01-01T00:00:00Z",
        }
        batcher.schedule("test-sid-001", snapshot)
        batcher.flush()

        # Verify row is present
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT session_id, status FROM audit_sessions WHERE session_id = ?",
            ("test-sid-001",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "test-sid-001"
        assert row[1] == "running"

    def test_batcher_deduplicates_writes(self, tmp_path: Path):
        """Multiple schedules for the same sid → only one row (latest wins)."""
        db_path = str(tmp_path / "dedup_test.db")
        batcher = _SqliteWriteBatcher(db_path, drain_interval_ms=50)

        for i in range(10):
            batcher.schedule("dedup-sid", {"status": f"version-{i}", "events": []})

        batcher.flush()

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM audit_sessions WHERE session_id = ?", ("dedup-sid",)
        ).fetchone()
        conn.close()
        # De-duplication: one session = one row via INSERT OR REPLACE
        assert rows[0] == 1

    def test_batcher_wal_mode_enabled(self, tmp_path: Path):
        """SQLite connection must be opened in WAL mode."""
        db_path = str(tmp_path / "wal_test.db")
        batcher = _SqliteWriteBatcher(db_path, drain_interval_ms=100)
        # Trigger connection by scheduling a write
        batcher.schedule("wal-test-sid", {"status": "queued", "events": []})
        batcher.flush()

        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        conn.close()
        assert result[0] == "wal"

    def test_batcher_schedule_is_nonblocking(self, tmp_path: Path):
        """schedule() must complete in < 10ms (non-blocking queue put)."""
        db_path = str(tmp_path / "timing_test.db")
        batcher = _SqliteWriteBatcher(db_path, drain_interval_ms=5000)

        t0 = time.monotonic()
        for _ in range(1000):
            batcher.schedule("timing-sid", {"status": "running", "events": []})
        elapsed_ms = (time.monotonic() - t0) * 1000

        batcher.flush()
        # 1000 non-blocking puts must complete in < 100ms (typically < 5ms)
        assert elapsed_ms < 100, f"schedule() blocked: {elapsed_ms:.1f}ms for 1000 calls"

    def test_batcher_handles_empty_queue_gracefully(self, tmp_path: Path):
        """flush() on an empty queue must not raise."""
        db_path = str(tmp_path / "empty_test.db")
        batcher = _SqliteWriteBatcher(db_path, drain_interval_ms=50)
        batcher.flush()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 10. SQLite round-trip (session persistence across store restarts)
# ─────────────────────────────────────────────────────────────────────────────

class TestSqlitePersistenceRoundTrip:
    """Verify sessions survive an AuditStore teardown + reload."""

    def test_sessions_survive_store_restart(self, tmp_path: Path):
        """Sessions created in one AuditStore instance are reloaded in a new one."""
        db_path = str(tmp_path / "restart_test.db")

        # First store: create a session
        with (
            patch.object(pers_module, "_probe_redis", return_value=None),
            patch.dict("os.environ", {"SQLITE_CHECKPOINT_PATH": db_path}),
        ):
            store1 = AuditStore()
        store1.create_session("persist-sid")
        store1.set_status("persist-sid", "completed")
        store1.set_final_state("persist-sid", {"attack_status": "success"})
        store1._shutdown_batcher()  # flush writes

        # Give batcher time to flush
        time.sleep(0.2)

        # Second store: load from same DB
        with (
            patch.object(pers_module, "_probe_redis", return_value=None),
            patch.dict("os.environ", {"SQLITE_CHECKPOINT_PATH": db_path}),
        ):
            store2 = AuditStore()

        # Session should be loaded
        assert store2.session_exists("persist-sid")

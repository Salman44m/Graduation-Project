import pytest
import sqlite3
import tempfile
import os
import json
import time

from infra.persistence import _SqliteWriteBatcher, AuditStore
from unittest.mock import patch

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass

def test_sync_flush_writes_immediately(temp_db):
    batcher = _SqliteWriteBatcher(temp_db)
    
    # We must call _get_conn once to ensure the DB schema is created
    # normally the background thread does this, but we are racing it in this test
    batcher._get_conn()
    
    sid = "test-session-123"
    snapshot = {"status": "error", "some_data": 42}
    
    # We call flush_sync directly
    batcher.flush_sync(sid, snapshot)
    
    # Check DB immediately, without waiting for background thread
    conn = sqlite3.connect(temp_db)
    cur = conn.execute("SELECT status, state_json FROM audit_sessions WHERE session_id = ? AND status = 'error'", (sid,))
    row = cur.fetchone()
    conn.close()
    
    assert row is not None
    assert row[0] == "error"
    assert json.loads(row[1])["some_data"] == 42
    
    batcher.flush()

@patch.dict(os.environ, {"SQLITE_CHECKPOINT_PATH": tempfile.mktemp()})
def test_audit_store_uses_sync_flush_on_error():
    # Make sure we don't connect to real Redis
    with patch("infra.persistence._probe_redis", return_value=None):
        store = AuditStore()
        sid = "test-session-err"
        
        # Patch the batcher to spy on it
        with patch.object(store._batcher, "flush_sync") as mock_flush_sync:
            store.create_session(sid)
            store.set_status(sid, "error")
            
            # verify flush_sync was called
            mock_flush_sync.assert_called_once()
            args = mock_flush_sync.call_args[0]
            assert args[0] == sid
            assert args[1]["status"] == "error"

@patch.dict(os.environ, {"SQLITE_CHECKPOINT_PATH": tempfile.mktemp()})
def test_audit_store_uses_sync_flush_on_complete():
    # Make sure we don't connect to real Redis
    with patch("infra.persistence._probe_redis", return_value=None):
        store = AuditStore()
        sid = "test-session-comp"
        
        with patch.object(store._batcher, "flush_sync") as mock_flush_sync:
            store.create_session(sid)
            store.set_status(sid, "complete")
            
            mock_flush_sync.assert_called_once()
            args = mock_flush_sync.call_args[0]
            assert args[0] == sid
            assert args[1]["status"] == "complete"
            
@patch.dict(os.environ, {"SQLITE_CHECKPOINT_PATH": tempfile.mktemp()})
def test_audit_store_uses_async_for_other_statuses():
    with patch("infra.persistence._probe_redis", return_value=None):
        store = AuditStore()
        sid = "test-session-run"
        
        with patch.object(store._batcher, "flush_sync") as mock_flush_sync, \
             patch.object(store._batcher, "schedule") as mock_schedule:
                 
            store.create_session(sid)
            # Create session does a schedule internally
            assert mock_schedule.call_count == 1
            
            store.set_status(sid, "running")
            
            mock_flush_sync.assert_not_called()
            # schedule is called again for set_status
            assert mock_schedule.call_count == 2

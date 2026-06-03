from pathlib import Path

def modify_persistence():
    p = Path("infra/persistence.py")
    content = p.read_text("utf-8")
    
    # 1. Add imports
    content = content.replace("from typing import Any", "from typing import Any\nimport sqlite3\nfrom datetime import datetime, timezone")
    
    # 2. Add init methods
    init_old = """    def __init__(self) -> None:
        self._redis  = _probe_redis()
        self._local: dict[str, dict] = {}   # fallback: in-process store
        self._lock   = threading.Lock()      # protects _local"""
        
    init_new = """    def __init__(self) -> None:
        self._redis  = _probe_redis()
        self._local: dict[str, dict] = {}   # fallback: in-process store
        self._lock   = threading.Lock()      # protects _local
        self._db_path = os.getenv("SQLITE_CHECKPOINT_PATH", "checkpoints.db")
        if not self._redis:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        with self._lock:
            with sqlite3.connect(self._db_path, check_same_thread=False) as conn:
                conn.execute(\"\"\"
                    CREATE TABLE IF NOT EXISTS audit_sessions (
                        session_id TEXT PRIMARY KEY,
                        status TEXT,
                        state_json TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                \"\"\")
                cur = conn.execute("SELECT session_id, status, state_json, created_at FROM audit_sessions")
                for row in cur:
                    sid, status, state_json, created_at = row
                    state = json.loads(state_json) if state_json else {}
                    self._local[sid] = {
                        "running": state.get("running", False),
                        "events": state.get("events", []),
                        "final_state": state.get("final_state"),
                        "error": state.get("error"),
                        "hitl": state.get("hitl"),
                        "status": status,
                        "report": state.get("report"),
                        "request": state.get("request"),
                        "started_at": created_at,
                        "latest_delta": state.get("latest_delta")
                    }

    def _save_to_sqlite(self, sid: str) -> None:
        if self._redis: return
        with sqlite3.connect(self._db_path, timeout=5.0, check_same_thread=False) as conn:
            state = self._local.get(sid, {})
            status = state.get("status") or "queued"
            state_json = json.dumps(state)
            now = datetime.now(timezone.utc).isoformat()
            created_at = state.get("started_at") or now
            conn.execute(\"\"\"
                INSERT OR REPLACE INTO audit_sessions (session_id, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            \"\"\", (sid, status, state_json, created_at, now))"""
            
    content = content.replace(init_old, init_new)

    # 3. Inject self._save_to_sqlite(sid) into all mutators.
    # Find all "                    self._local[sid]... = ..." inside "with self._lock:" 
    # and append "                    self._save_to_sqlite(sid)" after the assignment.
    # We can do this explicitly for each known mutator.
    
    replacements = {
        # create_session
        '                    "latest_delta":None,\n                }': '                    "latest_delta":None,\n                }\n                self._save_to_sqlite(sid)',
        
        # set_running
        '                    self._local[sid]["running"] = value': '                    self._local[sid]["running"] = value\n                    self._save_to_sqlite(sid)',
        
        # append_event
        '                    self._local[sid]["events"].append(event)': '                    self._local[sid]["events"].append(event)\n                    self._save_to_sqlite(sid)',
        
        # set_final_state
        '                    self._local[sid]["final_state"] = final': '                    self._local[sid]["final_state"] = final\n                    self._save_to_sqlite(sid)',
        
        # set_error
        '                    self._local[sid]["error"] = error': '                    self._local[sid]["error"] = error\n                    self._save_to_sqlite(sid)',
        
        # set_status
        '                    self._local[sid]["status"] = status': '                    self._local[sid]["status"] = status\n                    self._save_to_sqlite(sid)',
        
        # set_report
        '                    self._local[sid]["report"] = val': '                    self._local[sid]["report"] = val\n                    self._save_to_sqlite(sid)',
        
        # set_request
        '                    self._local[sid]["request"] = val': '                    self._local[sid]["request"] = val\n                    self._save_to_sqlite(sid)',
        
        # set_started_at
        '                    self._local[sid]["started_at"] = val': '                    self._local[sid]["started_at"] = val\n                    self._save_to_sqlite(sid)',
        
        # set_latest_delta
        '                    self._local[sid]["latest_delta"] = delta': '                    self._local[sid]["latest_delta"] = delta\n                    self._save_to_sqlite(sid)',
        
        # set_hitl
        '                    self._local[sid]["hitl"] = data': '                    self._local[sid]["hitl"] = data\n                    self._save_to_sqlite(sid)',
        
        # clear_hitl
        '                    self._local[sid]["hitl"] = None': '                    self._local[sid]["hitl"] = None\n                    self._save_to_sqlite(sid)',
        
        # push_hitl_decision
        '                    self._local[sid]["hitl"]["decision"] = decision': '                    self._local[sid]["hitl"]["decision"] = decision\n                    self._save_to_sqlite(sid)',
    }
    
    for old, new in replacements.items():
        if old not in content:
            print(f"Warning: could not find {old}")
        content = content.replace(old, new)

    p.write_text(content, "utf-8")
    print("Done")

if __name__ == "__main__":
    modify_persistence()

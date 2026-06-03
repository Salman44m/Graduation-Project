import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infra.persistence import get_audit_store, REDIS_URL

def main():
    print("STORAGE STATUS SUMMARY")
    print("---------------------")
    
    # 1. Check Redis
    redis_running = False
    try:
        import redis
        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        if client.ping():
            redis_running = True
            print("Redis: OK CONNECTED")
    except ImportError:
        print("Redis: FAIL NOT INSTALLED (pip install redis)")
    except Exception:
        print("Redis: FAIL NOT RUNNING")

    # 2. Check SQLite
    sqlite_path = os.getenv("SQLITE_CHECKPOINT_PATH", "checkpoints.db")
    if os.path.exists(sqlite_path):
        print(f"SQLite: OK EXISTS at {sqlite_path}")
    else:
        print(f"SQLite: WARN Will be created at {sqlite_path}")

    # 3. Check AuditStore Read/Write
    store = get_audit_store()
    sid = "test-check-storage-id"
    active_backend = "In-Memory only"
    
    try:
        store.create_session(sid)
        store.set_status(sid, "test-status")
        if store.get_status(sid) == "test-status":
            print("AuditStore: OK READ/WRITE OK")
            if store._redis is not None:
                active_backend = "Redis"
            else:
                active_backend = "SQLite (with In-Memory Cache)"
        else:
            print("AuditStore: FAIL FAILED (read mismatch)")
    except Exception as e:
        print(f"AuditStore: FAIL FAILED ({e})")

    # 4. Check directories
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
    benchmarks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'results'))

    reports_ok = os.path.exists(reports_dir)
    benchmarks_ok = os.path.exists(benchmarks_dir)

    print(f"Reports directory: {'OK EXISTS' if reports_ok else 'FAIL MISSING'}")
    print(f"Benchmark results: {'OK EXISTS' if benchmarks_ok else 'FAIL MISSING'}")

    print("\n")
    print("STORAGE STATUS SUMMARY")
    print("---------------------")
    print(f"Active backend: {active_backend}")
    persists = "YES (Redis)" if active_backend == "Redis" else "YES (SQLite)" if "SQLite" in active_backend else "NO"
    print(f"Data persists on restart: {persists}")
    print(f"Reports directory: {'OK' if reports_ok else 'MISSING'}")
    print(f"Benchmark results: {'OK' if benchmarks_ok else 'MISSING'}")

if __name__ == "__main__":
    main()

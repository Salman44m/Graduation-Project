# PromptEvo Storage Setup Guide

PromptEvo uses an abstraction layer for storage to operate correctly across single-worker debug environments and multi-worker production deployments.

## Storage Options

There are two primary storage components:

1. **AuditStore (Session Metadata, API Status)**:
   - **Redis (Recommended)**: Robust, distributed memory store. Survives restarts. Required for production.
   - **In-Memory**: A Python dictionary used automatically if Redis is unavailable. **Data is lost on program restart.**

2. **LangGraph Checkpointer (Agent State)**:
   - **RedisSaver**: Stores LangGraph checkpoints in Redis alongside `AuditStore`.
   - **SqliteSaver (Fallback)**: Stores checkpoints in a local file (`checkpoints.db`). Survives restarts, but limited to single-worker deployments safely.
   - **MemorySaver**: Used only if Redis and SQLite both fail. **Data is lost on program restart.**

## How to Install and Start Redis on Windows

Since you are running Windows, Redis isn't natively supported installable as an `.exe` service easily, but it is straightforward via WSL (Windows Subsystem for Linux) or Docker.

### Using WSL (Easiest)
1. Open PowerShell and install WSL if not already installed: `wsl --install`
2. Open Ubuntu (from the Start menu) and run:
   ```bash
   sudo apt update
   sudo apt install redis-server
   sudo service redis-server start
   ```

### Using Python Fallbacks
If you cannot install Redis, PromptEvo will safely fallback to in-memory AuditStore and local SQLite caching (`checkpoints.db`). Ensure you install the `redis` python package anyway to allow graceful failing:
```bash
pip install redis
```

## Configuring Your Backend

You can control the backend using environment variables in your `.env` file:

```env
# Point to your Redis instance (Defaults to localhost:6379/0)
REDIS_URL=redis://localhost:6379/0

# Optional: Set a custom path for the SQLite fallback
SQLITE_CHECKPOINT_PATH=custom_checkpoints.db
```

## Where is Data Stored?

| Item | Location | Backend | survives restart? |
| --- | --- | --- | --- |
| Session States & API Queues | `AuditStore` | Redis / Memory | Redis Only |
| Graph Execution History | `Checkpointer` | Redis / SQLite | Yes (`checkpoints.db`) |
| PDF Reports | `reports/*.pdf` | Local Disk | Yes |
| Benchmark Results | `benchmarks/results/*.jsonl` | Local Disk | Yes |

## Verifying Everything

Run the included storage verification script to check your local topology:

```bash
python scripts/check_storage.py
```
This script will test the active backend and confirm output directories exist.

"""
core/paths.py
─────────────────────────────────────────────────────────────────────────────
Centralized absolute path resolution for all data directories.
"""
import os
import pathlib

# Project root is two directories up from this file (core/paths.py)
#   __file__         = /app/core/paths.py
#   .parent          = /app/core
#   .parent.parent   = /app
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Allow override via environment variable for containerized deployments
_data_dir_override = os.getenv("PROMPTEVO_DATA_DIR")
if _data_dir_override:
    DATA_DIR = pathlib.Path(_data_dir_override).resolve()
else:
    DATA_DIR = PROJECT_ROOT

REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "data" / "memory" / "checkpoints.db"
TLTM_VECTORS_DIR = DATA_DIR / "data" / "memory" / "tltm_vectors"

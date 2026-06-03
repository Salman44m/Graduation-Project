import os
import pathlib
from unittest import mock
import pytest

from core.paths import PROJECT_ROOT, DATA_DIR, REPORTS_DIR, DB_PATH, TLTM_VECTORS_DIR

def test_project_root_resolves_to_repo():
    assert PROJECT_ROOT.name == "prompt_evo - Claude" or PROJECT_ROOT.name.startswith("prompt_evo")
    assert (PROJECT_ROOT / "core" / "paths.py").exists()

@mock.patch.dict(os.environ, {"PROMPTEVO_DATA_DIR": "/tmp/custom_data"})
def test_data_dir_overridable_via_env():
    # We must reload the module to trigger the env var read
    import importlib
    import core.paths
    importlib.reload(core.paths)
    
    try:
        assert str(core.paths.DATA_DIR).endswith("custom_data")
        assert str(core.paths.REPORTS_DIR).endswith("reports")
    finally:
        # Reload without the env var to not break other tests
        with mock.patch.dict(os.environ, clear=True):
            os.environ.pop("PROMPTEVO_DATA_DIR", None)
            importlib.reload(core.paths)

def test_reports_dir_uses_absolute_path():
    assert REPORTS_DIR.is_absolute()
    assert str(REPORTS_DIR) == str(PROJECT_ROOT / "reports")

def test_tltm_path_uses_absolute_path():
    assert TLTM_VECTORS_DIR.is_absolute()
    assert str(TLTM_VECTORS_DIR) == str(PROJECT_ROOT / "data" / "memory" / "tltm_vectors")

def test_db_path_uses_absolute_path():
    assert DB_PATH.is_absolute()
    assert str(DB_PATH) == str(PROJECT_ROOT / "data" / "memory" / "checkpoints.db")

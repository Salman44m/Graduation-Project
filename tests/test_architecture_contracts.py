"""Architecture freeze contract tests."""

import json
from pathlib import Path

import pytest

from core.state import ALL_FIELDS, INTELLIGENCE_FIELDS, validate_state_keys
from core.state import default_state


SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


def test_intelligence_fields_in_all_fields():
    assert INTELLIGENCE_FIELDS <= ALL_FIELDS


def test_default_state_has_no_phantom_keys():
    state = default_state(goal="test", target_model="gpt-4o", session_id="s1")
    phantoms = validate_state_keys(dict(state))
    assert not phantoms, f"Phantom keys in default_state: {phantoms}"


def test_schema_version_file_exists():
    version_path = SCHEMAS / "SCHEMA_VERSION"
    assert version_path.exists()
    version = version_path.read_text(encoding="utf-8").strip()
    assert version


@pytest.mark.skipif(
    not (SCHEMAS / "state_schema_v1.json").exists(),
    reason="Run scripts/export_schema.py --export first",
)
def test_state_schema_matches_export():
    from scripts.export_schema import export_state_schema

    committed = json.loads((SCHEMAS / "state_schema_v1.json").read_text(encoding="utf-8"))
    live = export_state_schema()
    assert committed["field_count"] == live["field_count"]
    assert set(committed["all_fields"]) == set(live["all_fields"])


def test_topology_hash_unchanged():
    from core.graph import compute_topology_hash, get_app

    current_hash = compute_topology_hash(get_app())
    # If this fails, the graph topology has been altered.
    # Expected hash must be explicitly updated when changes are intended.
    EXPECTED_HASH = "42c25160d564415ddc7c1438efb63dd9f474ce96590247e30f67318b2724c83a"
    assert current_hash == EXPECTED_HASH


def test_node_contracts_subset_of_all_fields():
    import yaml

    contracts_path = SCHEMAS / "node_contracts_v1.yaml"
    if not contracts_path.exists():
        pytest.skip("node_contracts_v1.yaml missing")
    data = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))
    for node_name, spec in (data.get("nodes") or {}).items():
        for key in ("writes", "reads"):
            for field_name in spec.get(key, []):
                assert field_name in ALL_FIELDS, f"{node_name}.{key}: {field_name} not in ALL_FIELDS"

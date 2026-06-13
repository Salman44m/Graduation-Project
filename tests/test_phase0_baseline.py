"""
tests/test_phase0_baseline.py
─────────────────────────────
Phase 0 baseline regression: topology hash, node counts, schema keys.
"""

from __future__ import annotations

from core.graph import (
    _GRAPH_CONDITIONAL_ROUTES,
    _GRAPH_UNCONDITIONAL_EDGES,
    _GRAPH_USER_NODES,
    compute_topology_hash,
    get_app,
    get_node_names,
    get_topology_spec,
)
from core.state import ALL_FIELDS, validate_state_update_safe

# Frozen at Phase 0 start — update only with intentional topology review.
PHASE0_TOPOLOGY_HASH = "6c00b411ee99003b87028ef421f517b1e1ba343cf302d4230345ebb4e98a26b9"

EPHEMERAL_KEYS = frozenset({
    "_current_eval_branch",
    "_cleartext_payload",
    "_seq_branch_evaluated",
    "_grooming_attacker_fallback",
})


class TestPhase0Baseline:

    def test_user_node_count(self):
        assert len(_GRAPH_USER_NODES) == 19

    def test_unconditional_edge_count(self):
        assert len(_GRAPH_UNCONDITIONAL_EDGES) == 7

    def test_conditional_route_count(self):
        assert len(_GRAPH_CONDITIONAL_ROUTES) == 12

    def test_topology_hash_matches_baseline(self):
        app = get_app()
        assert app is not None
        assert compute_topology_hash(app) == PHASE0_TOPOLOGY_HASH

    def test_get_node_names_match_spec(self):
        app = get_app()
        assert app is not None
        compiled = {n for n in get_node_names() if not n.startswith("__")}
        assert compiled == set(_GRAPH_USER_NODES)

    def test_topology_spec_entry_point(self):
        spec = get_topology_spec()
        assert spec["entry_point"] == "intel_retriever"

    def test_ephemeral_keys_in_all_fields(self):
        assert EPHEMERAL_KEYS <= ALL_FIELDS

    def test_validate_state_update_safe_accepts_ephemeral_keys(self):
        update = {
            "_current_eval_branch": {"branch_id": "b1"},
            "_cleartext_payload": "x",
            "_seq_branch_evaluated": True,
        }
        assert validate_state_update_safe(update, node_name="test", strict=False) == []

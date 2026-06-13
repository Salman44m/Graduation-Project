import json
from pathlib import Path

import pytest

from memory.threat_graph import (
    EDGE_BLOCKED_BY,
    EDGE_BYPASSED_BY,
    NODE_DEFENSE_MECHANISM,
    ThreatMemoryGraph,
    get_threat_graph,
)


@pytest.fixture
def graph(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.threat_graph.GRAPH_DIR", tmp_path)
    g = ThreatMemoryGraph("test-model-a")
    return g


def test_upsert_attempt_success_creates_bypass_edge(graph):
    graph.upsert_attempt(
        "Logical Appeal",
        "success",
        pap_technique="Logical Appeal",
        score=4.5,
        fingerprint={
            "refusal_style": "soft_refusal",
            "inferred_defense_mechanisms": ["policy_filter"],
        },
    )
    mech_nodes = [
        n for n, d in graph._graph.nodes(data=True)
        if d.get("node_type") == NODE_DEFENSE_MECHANISM
    ]
    assert mech_nodes
    edges = [
        d.get("edge_type")
        for _, _, d in graph._graph.edges(data=True)
    ]
    assert EDGE_BYPASSED_BY in edges


def test_upsert_attempt_failure_creates_blocked_edge(graph):
    graph.upsert_attempt(
        "Misrepresentation",
        "failure",
        score=1.0,
        fingerprint={
            "refusal_style": "policy_cite",
            "inferred_defense_mechanisms": ["constitutional_ai"],
        },
        response_text="This violates our content policy.",
    )
    edges = [d.get("edge_type") for _, _, d in graph._graph.edges(data=True)]
    assert EDGE_BLOCKED_BY in edges


def test_persistence_round_trip(graph, tmp_path):
    graph.upsert_attempt("Authority Endorsement", "failure", score=1.5)
    graph.save()
    path = tmp_path / "test-model-a.json"
    assert path.exists()

    g2 = ThreatMemoryGraph("test-model-a")
    assert g2._graph.number_of_nodes() >= 1


def test_query_planning_context(graph):
    graph.upsert_attempt(
        "Logical Appeal", "success", score=4.0,
        fingerprint={"inferred_defense_mechanisms": ["policy_filter"], "vulnerabilities": []},
    )
    ctx = graph.query_planning_context("test objective", {"inferred_defense_mechanisms": ["policy_filter"]})
    assert ctx.observation_count >= 1
    assert "policy_filter" in ctx.defense_mechanisms


def test_get_failed_strategies(graph):
    graph.upsert_attempt(
        "Emotional Appeal", "failure", score=1.0,
        fingerprint={"inferred_defense_mechanisms": ["constitutional_ai"]},
    )
    failed = graph.get_failed_strategies(mechanism_ids=["constitutional_ai"])
    assert len(failed) >= 1
    assert failed[0]["defense_mechanism"] == "constitutional_ai"

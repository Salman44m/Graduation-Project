import os
import pytest
import networkx as nx
from pathlib import Path
from memory.threat_graph import ThreatMemoryGraph

@pytest.fixture
def test_target():
    target = "test_target_model_12345"
    db_path = Path(f"data/memory/threat_graphs/{target}.json")
    if db_path.exists():
        os.remove(db_path)
    yield target
    if db_path.exists():
        os.remove(db_path)

def test_networkx_graph_initialization(test_target):
    tmg = ThreatMemoryGraph(test_target)
    assert isinstance(tmg.graph, nx.MultiDiGraph)
    assert tmg.graph.has_node(test_target)
    assert tmg.graph.nodes[test_target].get("type") == "Target"
    
def test_upsert_attempt_creates_edges(test_target):
    tmg = ThreatMemoryGraph(test_target)
    # 1. Blocked attempt
    tmg.upsert_attempt("Logical Appeal", "semantic_filter", "failure", 3)
    assert tmg.graph.has_node("Logical Appeal")
    assert tmg.graph.has_node("semantic_filter")
    
    # Verify BLOCKED_BY edge
    edges = tmg.graph.get_edge_data("Logical Appeal", "semantic_filter")
    assert any(data.get("type") == "BLOCKED_BY" for data in edges.values())
    
    # 2. Add second attempt
    tmg.upsert_attempt("Logical Appeal", "semantic_filter", "failure", 5)
    edges = tmg.graph.get_edge_data("Logical Appeal", "semantic_filter")
    blocked_edge = [data for data in edges.values() if data.get("type") == "BLOCKED_BY"][0]
    assert blocked_edge["count"] == 2
    assert blocked_edge["avg_turn"] == 4.0 # (3 + 5) / 2
    
def test_upsert_session_infers_mechanisms(test_target):
    tmg = ThreatMemoryGraph(test_target)
    # With inferred mechanism
    fingerprint = {
        "inferred_defense_mechanisms": ["policy_filter", "context_guard"]
    }
    tmg.upsert_session(fingerprint, "success")
    
    # Target should have DEFENDED_BY edges to policy_filter and context_guard
    assert tmg.graph.has_edge(test_target, "policy_filter")
    edges = tmg.graph.get_edge_data(test_target, "policy_filter")
    assert any(data.get("type") == "DEFENDED_BY" for data in edges.values())
    
    # Fallback inference
    fingerprint_fallback = {"refusal_style": "hard_refusal"}
    tmg.upsert_session(fingerprint_fallback, "failure")
    assert tmg.graph.has_edge(test_target, "rlhf_refusal")

def test_get_failed_strategies_traversal(test_target):
    tmg = ThreatMemoryGraph(test_target)
    tmg.upsert_attempt("TechniqueA", "MechX", "failure", 2)
    tmg.upsert_attempt("TechniqueB", "MechX", "failure", 1)
    tmg.upsert_attempt("TechniqueB", "MechX", "failure", 1)
    
    failed = tmg.get_failed_strategies(["MechX"])
    assert len(failed) == 2
    # Technique B should be first because count=2
    assert failed[0]["technique"] == "TechniqueB"
    assert failed[0]["count"] == 2
    assert failed[1]["technique"] == "TechniqueA"
    assert failed[1]["count"] == 1

def test_graph_persistence_node_link(test_target):
    tmg = ThreatMemoryGraph(test_target)
    tmg.upsert_attempt("Tech1", "Mech1", "success", 1)
    tmg.save()
    
    # Load into new instance
    tmg2 = ThreatMemoryGraph(test_target)
    assert tmg2.graph.has_node("Tech1")
    assert tmg2.graph.has_edge("Tech1", "Mech1")
    edges = tmg2.graph.get_edge_data("Tech1", "Mech1")
    assert any(data.get("type") == "BYPASSED_BY" for data in edges.values())

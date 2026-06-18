"""
Phase 2 validation tests — Threat Graph Activation.

Tests:
  1. get_technique_stats() — schema, Laplace smoothing, empty-safe
  2. get_successful_strategies() — graph traversal
  3. get_threat_graph() factory
  4. build_graph_retrieval_context() end-to-end with a real graph
  5. generate_attack_plan() receives non-empty technique_stats
  6. Route probabilities change when historical graph data changes
"""
import pytest


@pytest.fixture
def tmp_target(tmp_path, monkeypatch):
    target = "phase2_val_test_target_99999"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "memory" / "threat_graphs").mkdir(parents=True)
    yield target


def test_get_technique_stats_empty_graph(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    stats = tmg.get_technique_stats()
    assert stats == {}, "Empty graph should return empty dict"


def test_get_technique_stats_laplace_smoothing(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 3)
    stats = tmg.get_technique_stats()
    assert "attack_swarm" in stats
    assert "rlhf_refusal" in stats["attack_swarm"]
    prob = stats["attack_swarm"]["rlhf_refusal"]
    assert abs(prob - 2 / 3) < 1e-6, f"Expected 0.667 (Laplace: (1+1)/(1+0+2)), got {prob}"


def test_get_technique_stats_mixed_outcomes(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("gci", "policy_filter", "success", 2)
    tmg.upsert_attempt("gci", "policy_filter", "success", 2)
    tmg.upsert_attempt("gci", "policy_filter", "failure", 5)
    stats = tmg.get_technique_stats()
    assert "gci" in stats
    prob = stats["gci"]["policy_filter"]
    # 2 bypassed, 1 blocked -> Laplace: (2+1)/(2+1+2) = 3/5 = 0.6
    assert abs(prob - 3 / 5) < 1e-6, f"Expected 0.6, got {prob}"


def test_get_technique_stats_schema_matches_planner(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("rmce", "semantic_filter", "success", 1)
    stats = tmg.get_technique_stats()
    for tech, mechs in stats.items():
        assert isinstance(tech, str)
        for mech, prob in mechs.items():
            assert isinstance(mech, str)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0


def test_get_successful_strategies_empty(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    result = tmg.get_successful_strategies(["rlhf_refusal"])
    assert result == []


def test_get_successful_strategies_returns_bypassed(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 2)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 3)
    tmg.upsert_attempt("gci", "rlhf_refusal", "failure", 1)
    result = tmg.get_successful_strategies(["rlhf_refusal"])
    assert len(result) == 1
    assert result[0]["technique"] == "attack_swarm"
    assert result[0]["count"] == 2


def test_get_successful_strategies_sorted_by_count(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("gci", "rlhf_refusal", "success", 1)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 1)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 2)
    result = tmg.get_successful_strategies(["rlhf_refusal"])
    assert result[0]["technique"] == "attack_swarm"


def test_get_threat_graph_factory(tmp_target):
    from memory.threat_graph import get_threat_graph, ThreatMemoryGraph
    tmg = get_threat_graph(tmp_target)
    assert isinstance(tmg, ThreatMemoryGraph)
    assert tmg.target_id == tmp_target


def test_build_graph_retrieval_context_with_data(tmp_target):
    from memory.threat_graph import ThreatMemoryGraph
    from intelligence.rag_attack_planner import build_graph_retrieval_context
    tmg = ThreatMemoryGraph(tmp_target)
    tmg.upsert_attempt("attack_swarm", "rlhf_refusal", "success", 2)
    tmg.upsert_attempt("gci", "rlhf_refusal", "failure", 3)
    tmg.save()
    fingerprint = {"inferred_defense_mechanisms": ["rlhf_refusal"]}
    ctx = build_graph_retrieval_context(tmp_target, "test objective", fingerprint)
    assert "technique_stats" in ctx
    assert "failed_strategies" in ctx
    assert "successful_strategies" in ctx
    assert "observation_count" in ctx
    assert ctx["technique_stats"] != {}
    assert ctx["observation_count"] > 0


def test_build_graph_retrieval_context_cold_start(tmp_target):
    from intelligence.rag_attack_planner import build_graph_retrieval_context
    fingerprint = {"inferred_defense_mechanisms": ["rlhf_refusal"]}
    ctx = build_graph_retrieval_context(tmp_target, "test", fingerprint)
    assert isinstance(ctx, dict)
    assert "observation_count" in ctx


def test_generate_attack_plan_receives_technique_stats(tmp_target):
    from intelligence.rag_attack_planner import generate_attack_plan
    state = {
        "target_model_id": tmp_target,
        "core_malicious_objective": "test",
        "defense_fingerprint": {"inferred_defense_mechanisms": ["rlhf_refusal"]},
        "vulnerability_profile": {"recommended_attack": "gci"},
        "graph_retrieval_context": {
            "technique_stats": {
                "attack_swarm": {"rlhf_refusal": 0.85},
                "gci":          {"rlhf_refusal": 0.15},
                "rmce":         {"rlhf_refusal": 0.50},
                "decomposer":   {"rlhf_refusal": 0.50},
            },
            "failed_strategies": [],
            "successful_strategies": [],
            "observation_count": 10,
        },
    }
    plan = generate_attack_plan(state)
    assert isinstance(plan, dict)
    assert "recommended_route" in plan
    assert plan["recommended_route"] == "attack_swarm", (
        f"Expected attack_swarm (P=0.85), got {plan['recommended_route']}"
    )


def test_route_probabilities_change_with_graph_data(tmp_target):
    from intelligence.rag_attack_planner import generate_attack_plan
    base_state = {
        "target_model_id": tmp_target,
        "core_malicious_objective": "test",
        "defense_fingerprint": {},
        "vulnerability_profile": {},
    }
    cold_plan = generate_attack_plan({**base_state, "graph_retrieval_context": {"observation_count": 0}})

    warm_state = {
        **base_state,
        "graph_retrieval_context": {
            "technique_stats": {
                # Give gci a massive historical advantage; push others very low
                "gci":          {"rlhf_refusal": 0.95, "policy_filter": 0.90, "semantic_filter": 0.88},
                "attack_swarm": {"rlhf_refusal": 0.05},
                "rmce":         {"rlhf_refusal": 0.05},
                "decomposer":   {"rlhf_refusal": 0.05},
            },
            "failed_strategies": [],
            "successful_strategies": [],
            "observation_count": 20,
        },
    }
    warm_plan = generate_attack_plan(warm_state)

    # Primary assertion: graph data changes route selection
    assert warm_plan["recommended_route"] == "gci", (
        f"Expected gci to win warm-start, got {warm_plan['recommended_route']}"
    )
    assert cold_plan["recommended_route"] != "gci", (
        "Cold start should not select gci when graph data is absent"
    )

    # Secondary assertion: gci rank_score is strictly higher in warm start
    warm_candidates = {c["recommended_route"]: c["rank_score"] for c in warm_plan.get("candidate_plans", [])}
    cold_candidates = {c["recommended_route"]: c["rank_score"] for c in cold_plan.get("candidate_plans", [])}
    if warm_candidates and cold_candidates:
        gci_warm = warm_candidates.get("gci", 0.0)
        gci_cold = cold_candidates.get("gci", 0.0)
        assert gci_warm > gci_cold, (
            f"gci rank_score must be higher warm ({gci_warm:.4f}) than cold ({gci_cold:.4f})"
        )

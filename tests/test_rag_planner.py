from intelligence.rag_attack_planner import generate_attack_plan, _beta_success_rate


def test_beta_success_rate_prior():
    rate = _beta_success_rate(0, 0)
    assert 0.0 < rate < 1.0


def test_generate_attack_plan_cold_start():
    state = {
        "target_model_id": "new-model",
        "core_malicious_objective": "test objective",
        "defense_fingerprint": {"alignment_score": 0.5, "observation_count": 0},
        "vulnerability_profile": {"recommended_attack": "attack_swarm"},
        "graph_retrieval_context": {"observation_count": 0},
    }
    plan = generate_attack_plan(state)
    assert "recommended_route" in plan
    assert "expected_success_probability" in plan
    assert "confidence" in plan
    assert 0.0 <= plan["expected_success_probability"] <= 1.0


def test_generate_attack_plan_has_candidate_plans():
    state = {
        "target_model_id": "m1",
        "core_malicious_objective": "obj",
        "defense_fingerprint": {"inferred_defense_mechanisms": ["policy_filter"]},
        "vulnerability_profile": {},
        "graph_retrieval_context": {"observation_count": 2, "failed_strategies": []},
    }
    plan = generate_attack_plan(state)
    assert len(plan.get("candidate_plans", [])) >= 1

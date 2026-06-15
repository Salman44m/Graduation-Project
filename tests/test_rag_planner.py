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


from intelligence.rag_attack_planner import RagAttackPlanner

def test_compute_expected_probability():
    planner = RagAttackPlanner()
    # 0 observations
    prob = planner.compute_expected_probability(0, 0)
    assert 0 < prob < 1
    assert prob == 1.0 / 3.0 # (0+1)/(0+0+1+2) = 0.3333
    
    # High success
    prob_high = planner.compute_expected_probability(10, 0)
    assert prob_high > 0.8
    
    # High failure
    prob_low = planner.compute_expected_probability(0, 10)
    assert prob_low < 0.1

def test_calculate_confidence():
    planner = RagAttackPlanner()
    assert planner.calculate_confidence(0, min_obs=5) == 0.0
    assert planner.calculate_confidence(2, min_obs=5) == 0.4
    assert planner.calculate_confidence(5, min_obs=5) == 1.0
    assert planner.calculate_confidence(10, min_obs=5) == 1.0

def test_rank_candidate_techniques():
    planner = RagAttackPlanner()
    techs = [
        {"technique_id": "TechA", "success_count": 0, "failure_count": 10},
        {"technique_id": "TechB", "success_count": 10, "failure_count": 0},
        {"technique_id": "TechC", "success_count": 10, "failure_count": 0} # same prob, different confidence? Wait, same counts.
    ]
    # Let's add TechD with lower confidence but high success prob
    techs.append({"technique_id": "TechD", "success_count": 1, "failure_count": 0})
    
    ranked = planner.rank_candidate_techniques(techs)
    # B and C should be top (prob ~ 0.84, conf 1.0)
    # D should be next (prob 0.5, conf 0.2)
    # A should be last (prob ~0.07, conf 1.0)
    assert ranked[0]["technique_id"] in ("TechB", "TechC")
    assert ranked[1]["technique_id"] in ("TechB", "TechC")
    assert ranked[2]["technique_id"] == "TechD"
    assert ranked[3]["technique_id"] == "TechA"
    
def test_build_attack_plan_schema():
    planner = RagAttackPlanner()
    plan = planner.build_attack_plan(
        target_id="test_target",
        defense_fingerprint={"inferred_defense_mechanisms": ["rlhf_refusal"]},
        graph_context={},
        tltm_context=[]
    )
    assert isinstance(plan, dict)
    assert "recommended_route" in plan
    assert "techniques" in plan
    assert "avoid_patterns" in plan
    assert "expected_success_probability" in plan
    assert "confidence" in plan
    assert "primary_defense_mechanisms" in plan
    assert plan["recommended_route"] == "attack_swarm"

def test_avoid_patterns_population():
    planner = RagAttackPlanner()
    graph_context = {
        "failed_strategies": [
            {"technique": "Logical Appeal"},
            {"technique_id": "Authority Appeal"}
        ]
    }
    plan = planner.build_attack_plan(
        target_id="test_target",
        defense_fingerprint={},
        graph_context=graph_context,
        tltm_context=[]
    )
    assert len(plan["avoid_patterns"]) == 2
    assert "Logical Appeal" in plan["avoid_patterns"]
    assert "Authority Appeal" in plan["avoid_patterns"]

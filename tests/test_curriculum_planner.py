import pytest
from intelligence.curriculum_planner import CurriculumPlanner

def test_plan_always_4_stages_no_duplicates():
    planner = CurriculumPlanner()
    plan = planner.build_plan("objective", {})
    
    assert len(plan) == 4
    
    names = [stage["name"] for stage in plan]
    assert len(set(names)) == 4, "Duplicate stage names found"
    
    stages = [stage["stage"] for stage in plan]
    assert stages == [0, 1, 2, 3]

def test_evaluate_stage_advance():
    planner = CurriculumPlanner()
    
    # Stage 0 -> Stage 1
    state_0_to_1 = {"cooperation_score": 0.55, "response_class": "partial_comply"}
    new_idx = planner.evaluate_stage_progression(0, state_0_to_1)
    assert new_idx == 1
    
    # Stage 1 -> Stage 2
    state_1_to_2 = {
        "cooperation_score": 0.65,
        "target_defense_profile": {"comply_count": 2},
        "response_class": "unknown"
    }
    assert planner.evaluate_stage_progression(1, state_1_to_2) == 2
    
    # Stage 2 -> Stage 3
    state_2_to_3 = {"cooperation_score": 0.8, "response_class": "partial_comply"}
    assert planner.evaluate_stage_progression(2, state_2_to_3) == 3
    
    # Stage 3 -> Stage 3
    state_3 = {"prometheus_score": 4.5, "cooperation_score": 0.9}
    assert planner.evaluate_stage_progression(3, state_3) == 3

def test_evaluate_stage_regress():
    planner = CurriculumPlanner()
    
    # cooperation_score <= 0.20 -> regress
    state_regress_score = {"cooperation_score": 0.1, "response_class": "unknown"}
    assert planner.evaluate_stage_progression(2, state_regress_score) == 1
    
    # response_class == "hard_refusal" -> regress
    state_regress_refusal = {"cooperation_score": 0.5, "response_class": "hard_refusal"}
    assert planner.evaluate_stage_progression(1, state_regress_refusal) == 0
    
    # already at 0, should stay at 0
    assert planner.evaluate_stage_progression(0, state_regress_score) == 0

def test_get_recommended_tactic_cycles():
    planner = CurriculumPlanner()
    
    # Stage 0 tactics: ["cold_open", "epistemic_debt"]
    stage_info = planner.STAGES[0]
    
    # Start with first
    tactic = planner.get_recommended_tactic(stage_info, "cold_open")
    assert tactic == "epistemic_debt"
    
    # Cycle back to first
    tactic = planner.get_recommended_tactic(stage_info, "epistemic_debt")
    assert tactic == "cold_open"
    
    # Unmatched current_tactic -> return first
    tactic = planner.get_recommended_tactic(stage_info, "unknown_tactic")
    assert tactic == "cold_open"

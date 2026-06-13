from intelligence.curriculum_planner import (
    advance_curriculum_stage,
    build_curriculum_plan,
    curriculum_to_crescendo_steps,
)


def test_build_curriculum_plan_four_stages():
    plan = build_curriculum_plan({})
    assert len(plan) == 4
    assert plan[0]["name"] == "TrustBuilding"


def test_advance_on_cooperation():
    state = {"curriculum_stage": 0, "cooperation_score": 0.6, "response_class": "partial_comply"}
    assert advance_curriculum_stage(state) == 1


def test_regress_on_hard_refusal():
    state = {"curriculum_stage": 2, "response_class": "hard_refusal"}
    assert advance_curriculum_stage(state) == 1


def test_curriculum_to_crescendo():
    plan = build_curriculum_plan({})
    steps = curriculum_to_crescendo_steps(plan, 1, "test objective")
    assert len(steps) == 2

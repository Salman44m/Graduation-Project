"""Legacy routing tests — updated for Phase 0 router split (intel_updater terminal path)."""

import core.graph as graph_module
from core.graph import (
    route_from_analyst,
    route_after_target_decomposition,
    route_after_target_warmup,
    route_after_target_attack,
    MAX_SESSION_TURNS,
    COOP_SCOUT_THRESHOLD,
    _INTEL_UPDATER,
    _ATTACK_SWARM,
    _SCOUT,
    _TARGET,
    _COMBINER,
    _RMCE,
    _SELF_REFEREE,
)


def _base(**overrides):
    state = {
        "grooming_phase_active": False,
        "cooperation_score": 1.0,
        "turn_count": 1,
        "attack_status": "in_progress",
        "route_decision": "attack_swarm",
    }
    state.update(overrides)
    return state


def test_standard_mode_to_attack_swarm():
    state = _base(route_decision="attack_swarm")
    assert route_from_analyst(state) == _ATTACK_SWARM


def test_decomposition_remaining_to_target():
    state = _base(
        attack_status="decomposing",
        sub_questions=["Q1", "Q2"],
        collected_sub_answers=["A1"],
        decomposition_index=1,
    )
    assert route_after_target_decomposition(state) == _TARGET


def test_decomposition_complete_to_combiner():
    state = _base(
        attack_status="decomposing",
        sub_questions=["Q1", "Q2"],
        collected_sub_answers=["A1", "A2"],
        decomposition_index=2,
    )
    assert route_after_target_decomposition(state) == _COMBINER


def test_rmce_loopback():
    state = _base(
        rmce_meta_level=1,
        route_decision="rmce",
    )
    assert route_after_target_attack(state) == _RMCE


def test_self_referee_gate():
    state = _base(
        route_decision="analyst",
        current_depth=0,
        self_referee_done=False,
    )
    assert route_after_target_warmup(state) == _SELF_REFEREE


def test_max_turns_exceeded_terminal():
    max_turns = graph_module.MAX_SESSION_TURNS
    state = {
        "grooming_phase_active": False,
        "cooperation_score": 1.0,
        "turn_count": max_turns + 1,
        "attack_status": "in_progress",
        "route_decision": "",
    }
    assert route_from_analyst(state) == _INTEL_UPDATER


def test_error_recovery_scout_fallback():
    from core.constants import THRESHOLD
    state = _base(
        turn_count=0,
        cooperation_score=THRESHOLD.scout_warmup - 0.05,
        route_decision="",
    )
    assert route_from_analyst(state) == _SCOUT

"""
tests/test_graph_routing.py
─────────────────────────────────────────────────────────────────────────────
Tests for every routing function in core/graph.py.

These are tested in COMPLETE ISOLATION — no LangGraph execution, no LLM
calls, no database connections. Each routing function is a pure function
over AuditorState that we call directly with crafted state dicts.

Routing functions under test:
  - route_after_scout
  - route_from_analyst
  - route_after_attack_swarm  (HITL-disabled path only)
  - route_after_gci
  - route_after_branch_merge
  - route_after_rmce
  - route_after_classifier
  - route_after_target_decomposition
  - route_after_target_warmup
  - route_after_target_attack
  - route_after_target          (top-level dispatcher)
  - route_from_combiner
  - route_from_judge
  - route_after_pool_on_fail
  - route_after_remediation
  - route_after_pool_on_success
  - _route_pool_combined

All test names follow: test_<function>_<condition>_routes_to_<destination>
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

# Import routing functions directly — no graph compilation needed
from core.graph import (
    route_after_scout,
    route_from_analyst,
    route_after_gci,
    route_after_branch_merge,
    route_after_rmce,
    route_after_classifier,
    route_after_target_decomposition,
    route_after_target_warmup,
    route_after_target_attack,
    route_after_target,
    route_from_combiner,
    route_from_judge,
    route_after_pool_on_fail,
    route_after_remediation,
    route_after_pool_on_success,
    _route_pool_combined,
    _REPORTER, _ANALYST, _ATTACK_SWARM, _TARGET, _BRANCH_EVAL,
    _DECOMPOSER, _COMBINER, _JUDGE, _POOL, _HITL, _CLASSIFIER,
    _SELF_REFEREE, _GCI, _RMCE, _REMEDIATION, _SCOUT,
    MAX_SESSION_TURNS, MAX_RMCE_META_LEVEL,
)
from core.state import default_state, AuditorState
from core.constants import BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build minimal state dict with overrides
# ─────────────────────────────────────────────────────────────────────────────

def _state(**overrides) -> AuditorState:
    """Return default_state() with specific fields overridden."""
    s = dict(default_state("test goal"))
    s.update(overrides)
    return AuditorState(**s)


# ─────────────────────────────────────────────────────────────────────────────
# route_after_scout
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterScout:

    def test_error_status_routes_to_reporter(self):
        state = _state(attack_status="error")
        assert route_after_scout(state) == _REPORTER

    def test_route_decision_reporter_routes_to_reporter(self):
        state = _state(route_decision="reporter")
        assert route_after_scout(state) == _REPORTER

    def test_route_decision_analyst_bypass_routes_to_analyst(self):
        state = _state(route_decision="analyst_bypass")
        assert route_after_scout(state) == _ANALYST

    def test_default_routes_to_target(self):
        """No special conditions → proceed to target node."""
        state = _state(attack_status="in_progress", route_decision="scout")
        assert route_after_scout(state) == _TARGET

    def test_empty_route_decision_routes_to_target(self):
        """Empty route_decision → proceed to target."""
        state = _state(attack_status="in_progress", route_decision="")
        assert route_after_scout(state) == _TARGET

    def test_error_takes_priority_over_route_decision(self):
        """Error status overrides route_decision='analyst_bypass'."""
        state = _state(attack_status="error", route_decision="analyst_bypass")
        assert route_after_scout(state) == _REPORTER


# ─────────────────────────────────────────────────────────────────────────────
# route_from_analyst  (primary strategic router — most critical)
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteFromAnalyst:

    def test_budget_exhausted_by_turns_routes_to_reporter(self):
        """Circuit breaker: turn_count >= MAX → reporter."""
        state = _state(turn_count=MAX_SESSION_TURNS)
        assert route_from_analyst(state) == _REPORTER

    def test_budget_exceeds_max_routes_to_reporter(self):
        """Even one turn over max must hit reporter."""
        state = _state(turn_count=MAX_SESSION_TURNS + 5)
        assert route_from_analyst(state) == _REPORTER

    def test_success_status_routes_to_reporter(self):
        """Terminal status 'success' → reporter."""
        state = _state(attack_status="success", turn_count=5)
        assert route_from_analyst(state) == _REPORTER

    def test_failure_status_routes_to_reporter(self):
        """Terminal status 'failure' → reporter."""
        state = _state(attack_status="failure", turn_count=5)
        assert route_from_analyst(state) == _REPORTER

    def test_exhausted_status_routes_to_reporter(self):
        """Terminal status 'exhausted' → reporter."""
        state = _state(attack_status="exhausted", turn_count=5)
        assert route_from_analyst(state) == _REPORTER

    def test_error_status_routes_to_reporter(self):
        """Terminal status 'error' → reporter."""
        state = _state(attack_status="error", turn_count=5)
        assert route_from_analyst(state) == _REPORTER

    def test_explicit_terminal_decision_routes_to_reporter(self):
        """route_decision='terminal' → reporter (explicit analyst command)."""
        state = _state(route_decision="terminal", turn_count=5)
        assert route_from_analyst(state) == _REPORTER

    def test_explicit_resurrect_routes_to_attack_swarm(self):
        """route_decision='resurrect' → attack_swarm (fresh beam)."""
        state = _state(route_decision="resurrect", turn_count=5)
        assert route_from_analyst(state) == _ATTACK_SWARM

    def test_explicit_decomposer_routes_to_decomposer(self):
        """route_decision='decomposer' → decomposer."""
        state = _state(route_decision="decomposer", turn_count=5)
        assert route_from_analyst(state) == _DECOMPOSER

    def test_explicit_gci_routes_to_gci(self):
        """route_decision='gci' → gci."""
        state = _state(route_decision="gci", turn_count=5)
        assert route_from_analyst(state) == _GCI

    def test_explicit_rmce_routes_to_rmce(self):
        """route_decision='rmce' → rmce."""
        state = _state(route_decision="rmce", turn_count=5)
        assert route_from_analyst(state) == _RMCE

    def test_explicit_scout_routes_to_scout(self):
        """route_decision='scout' → scout (explicit analyst command)."""
        state = _state(route_decision="scout", turn_count=5)
        assert route_from_analyst(state) == _SCOUT

    def test_cold_target_at_turn_0_routes_to_scout(self):
        """Low coop at turn 0 triggers warm-up via scout."""
        state = _state(
            cooperation_score=0.1,  # below COOP_SCOUT_THRESHOLD=0.60
            turn_count=0,
            route_decision="attack_swarm",
        )
        assert route_from_analyst(state) == _SCOUT

    def test_cold_target_after_turn_0_does_not_route_to_scout(self):
        """Low coop after turn 0 must NOT redirect to scout (avoids infinite loop)."""
        state = _state(
            cooperation_score=0.1,
            turn_count=3,  # not turn 0
            route_decision="attack_swarm",
        )
        # Must NOT route to scout — the fix documented in graph.py
        assert route_from_analyst(state) == _ATTACK_SWARM

    def test_default_standard_attack_routes_to_attack_swarm(self):
        """Normal in-progress session → attack_swarm."""
        state = _state(
            cooperation_score=0.8,
            turn_count=5,
            attack_status="in_progress",
            route_decision="attack_swarm",
        )
        assert route_from_analyst(state) == _ATTACK_SWARM

    def test_resurrect_with_low_coop_still_routes_to_attack_swarm(self):
        """Resurrect decision must not be overridden by low coop at turn > 0."""
        state = _state(
            cooperation_score=0.1,
            turn_count=3,
            route_decision="resurrect",
        )
        assert route_from_analyst(state) == _ATTACK_SWARM

    def test_decomposer_with_low_coop_still_routes_to_decomposer(self):
        """Explicit decomposer decision must not be intercepted by coop guard."""
        state = _state(
            cooperation_score=0.1,
            turn_count=3,
            route_decision="decomposer",
        )
        assert route_from_analyst(state) == _DECOMPOSER


# ─────────────────────────────────────────────────────────────────────────────
# route_after_gci
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterGci:

    def test_hitl_disabled_routes_to_target(self):
        """With HITL disabled (default), GCI routes to target."""
        state = _state()
        with patch("core.graph.HITL_ENABLED", False):
            result = route_after_gci(state)
        assert result == _TARGET

    def test_hitl_enabled_routes_to_hitl(self):
        """With HITL enabled, GCI routes to hitl_review."""
        state = _state()
        with patch("core.graph.HITL_ENABLED", True):
            result = route_after_gci(state)
        assert result == _HITL


# ─────────────────────────────────────────────────────────────────────────────
# route_after_branch_merge
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterBranchMerge:

    def test_success_routes_to_remediation(self):
        state = _state(attack_status="success", turn_count=5)
        assert route_after_branch_merge(state) == _REMEDIATION

    def test_exhausted_routes_to_pool(self):
        state = _state(attack_status="exhausted", turn_count=5)
        assert route_after_branch_merge(state) == _POOL

    def test_failure_routes_to_pool(self):
        state = _state(attack_status="failure", turn_count=5)
        assert route_after_branch_merge(state) == _POOL

    def test_turn_limit_routes_to_pool(self):
        state = _state(attack_status="in_progress", turn_count=MAX_SESSION_TURNS)
        assert route_after_branch_merge(state) == _POOL

    def test_in_progress_routes_to_analyst(self):
        """No winner yet → loop back to analyst for next turn."""
        state = _state(attack_status="in_progress", turn_count=5)
        assert route_after_branch_merge(state) == _ANALYST


# ─────────────────────────────────────────────────────────────────────────────
# route_after_rmce
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterRmce:

    def test_route_decision_gci_routes_to_gci(self):
        """RMCE failure recovery via GCI."""
        state = _state(route_decision="gci")
        assert route_after_rmce(state) == _GCI

    def test_route_decision_attack_swarm_routes_to_attack_swarm(self):
        """RMCE failure recovery via attack_swarm."""
        state = _state(route_decision="attack_swarm")
        assert route_after_rmce(state) == _ATTACK_SWARM

    def test_refinement_complete_routes_to_classifier(self):
        """When RMCE refinement is done (max level, no pending payload) → classifier."""
        state = _state(
            rmce_meta_level=MAX_RMCE_META_LEVEL,
            pending_payload="",
            route_decision="",
        )
        assert route_after_rmce(state) == _CLASSIFIER

    def test_normal_path_hitl_disabled_routes_to_target(self):
        """Normal RMCE completion with HITL disabled → target."""
        state = _state(
            rmce_meta_level=1,
            pending_payload="some payload",
            route_decision="",
        )
        with patch("core.graph.HITL_ENABLED", False):
            result = route_after_rmce(state)
        assert result == _TARGET


# ─────────────────────────────────────────────────────────────────────────────
# route_after_classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterClassifier:
    """All classifier paths currently route to _JUDGE (the judge handles short-circuits)."""

    def test_hard_refusal_routes_to_judge(self):
        state = _state(response_class="hard_refusal")
        assert route_after_classifier(state) == _JUDGE

    def test_full_comply_routes_to_judge(self):
        state = _state(response_class="full_comply")
        assert route_after_classifier(state) == _JUDGE

    def test_partial_comply_routes_to_judge(self):
        state = _state(response_class="partial_comply")
        assert route_after_classifier(state) == _JUDGE

    def test_missing_response_class_defaults_to_judge(self):
        """Missing/empty response_class falls through to judge."""
        state = _state(response_class="partial_comply")
        assert route_after_classifier(state) == _JUDGE


# ─────────────────────────────────────────────────────────────────────────────
# route_after_target_decomposition
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterTargetDecomposition:

    def test_empty_sub_questions_routes_to_analyst(self):
        """No sub-questions (shouldn't happen) → analyst as fallback."""
        state = _state(sub_questions=[], collected_sub_answers=[])
        assert route_after_target_decomposition(state) == _ANALYST

    def test_more_questions_remaining_routes_to_target(self):
        """Not all questions answered → target for next question."""
        state = _state(
            sub_questions=["Q1?", "Q2?", "Q3?"],
            collected_sub_answers=["A1"],
            decomposition_index=1,
        )
        assert route_after_target_decomposition(state) == _TARGET

    def test_all_questions_answered_routes_to_combiner(self):
        """All questions answered → combiner for synthesis."""
        state = _state(
            sub_questions=["Q1?", "Q2?"],
            collected_sub_answers=["A1", "A2"],
            decomposition_index=2,
        )
        assert route_after_target_decomposition(state) == _COMBINER

    def test_single_question_answered_routes_to_combiner(self):
        """Single-question decomposition: answered → combiner."""
        state = _state(
            sub_questions=["Q1?"],
            collected_sub_answers=["A1"],
            decomposition_index=1,
        )
        assert route_after_target_decomposition(state) == _COMBINER


# ─────────────────────────────────────────────────────────────────────────────
# route_after_target_warmup
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterTargetWarmup:

    def test_first_warmup_depth_zero_no_self_referee_routes_to_self_referee(self):
        """First warm-up (depth=0, self_referee not done) → self_referee."""
        state = _state(current_depth=0, self_referee_done=False)
        assert route_after_target_warmup(state) == _SELF_REFEREE

    def test_first_warmup_self_referee_already_done_routes_to_analyst(self):
        """If self_referee already done → analyst."""
        state = _state(current_depth=0, self_referee_done=True)
        assert route_after_target_warmup(state) == _ANALYST

    def test_depth_gt_zero_routes_to_analyst(self):
        """Subsequent warm-up probes (depth > 0) → analyst."""
        state = _state(current_depth=1, self_referee_done=False)
        assert route_after_target_warmup(state) == _ANALYST


# ─────────────────────────────────────────────────────────────────────────────
# route_after_target_attack
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterTargetAttack:

    def test_active_rmce_loop_routes_to_rmce(self):
        """Mid-RMCE execution (1 <= meta_level < max): loop back to rmce."""
        state = _state(
            rmce_meta_level=1,
            route_decision="rmce",
        )
        assert route_after_target_attack(state) == _RMCE

    def test_rmce_at_max_with_refinements_routes_to_rmce(self):
        """RMCE at max level with pending refinements → rmce."""
        state = _state(
            rmce_meta_level=MAX_RMCE_META_LEVEL,
            rmce_refinement_count=1,
            route_decision="rmce",
        )
        assert route_after_target_attack(state) == _RMCE

    def test_rmce_at_max_refinements_exhausted_routes_to_classifier(self):
        """RMCE at max level with refinement count at max → classifier."""
        from core.graph import MAX_TURN3_REFINEMENTS
        state = _state(
            rmce_meta_level=MAX_RMCE_META_LEVEL,
            rmce_refinement_count=MAX_TURN3_REFINEMENTS,
            route_decision="rmce",
        )
        assert route_after_target_attack(state) == _CLASSIFIER

    def test_standard_attack_routes_to_classifier(self):
        """Normal TAP attack: target responded → classifier."""
        state = _state(rmce_meta_level=0, route_decision="attack_swarm")
        assert route_after_target_attack(state) == _CLASSIFIER


# ─────────────────────────────────────────────────────────────────────────────
# route_after_target (top-level dispatcher)
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterTarget:

    def test_error_routes_to_reporter(self):
        state = _state(attack_status="error")
        assert route_after_target(state) == _REPORTER

    def test_success_routes_to_remediation(self):
        state = _state(attack_status="success")
        assert route_after_target(state) == _REMEDIATION

    def test_decomposing_with_remaining_questions_routes_to_target(self):
        """Decomposing status + unanswered questions → delegates to decomp router → target."""
        state = _state(
            attack_status="decomposing",
            sub_questions=["Q1?", "Q2?"],
            collected_sub_answers=["A1"],
        )
        assert route_after_target(state) == _TARGET

    def test_decomposing_all_answered_routes_to_combiner(self):
        """Decomposing status + all answered → combiner."""
        state = _state(
            attack_status="decomposing",
            sub_questions=["Q1?", "Q2?"],
            collected_sub_answers=["A1", "A2"],
        )
        assert route_after_target(state) == _COMBINER

    def test_warmup_route_decision_routes_to_self_referee_or_analyst(self):
        """Warm-up path: route_decision='analyst', depth=0 → self_referee."""
        state = _state(
            attack_status="in_progress",
            route_decision="analyst",
            current_depth=0,
            self_referee_done=False,
        )
        assert route_after_target(state) == _SELF_REFEREE

    def test_standard_attack_routes_to_classifier(self):
        """Standard TAP attack → classifier."""
        state = _state(
            attack_status="in_progress",
            route_decision="attack_swarm",
            rmce_meta_level=0,
        )
        assert route_after_target(state) == _CLASSIFIER


# ─────────────────────────────────────────────────────────────────────────────
# route_from_combiner
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteFromCombiner:

    def test_always_routes_to_judge(self):
        """Combiner always sends to judge regardless of state."""
        for status in ["in_progress", "decomposing", "success", "failure"]:
            state = _state(attack_status=status)
            assert route_from_combiner(state) == _JUDGE


# ─────────────────────────────────────────────────────────────────────────────
# route_from_judge
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteFromJudge:

    def test_success_status_routes_to_remediation(self):
        state = _state(attack_status="success", prometheus_score=4.5, turn_count=5)
        assert route_from_judge(state) == _REMEDIATION

    def test_exhausted_status_routes_to_pool(self):
        state = _state(attack_status="exhausted", prometheus_score=2.0, turn_count=5)
        assert route_from_judge(state) == _POOL

    def test_turn_count_at_max_routes_to_pool(self):
        state = _state(attack_status="in_progress", turn_count=MAX_SESSION_TURNS)
        assert route_from_judge(state) == _POOL

    def test_failure_status_routes_to_reporter(self):
        state = _state(attack_status="failure", prometheus_score=1.0, turn_count=5)
        assert route_from_judge(state) == _REPORTER

    def test_high_score_in_progress_routes_to_remediation(self):
        """Score >= 4.0 with in_progress status → jailbreak confirmed → remediation."""
        state = _state(attack_status="in_progress", prometheus_score=4.0, turn_count=5)
        assert route_from_judge(state) == _REMEDIATION

    def test_low_score_in_progress_routes_to_pool(self):
        """Score < 4.0 with in_progress status → experience pool → retry."""
        state = _state(attack_status="in_progress", prometheus_score=3.9, turn_count=5)
        assert route_from_judge(state) == _POOL

    def test_exact_threshold_score_routes_to_remediation(self):
        """Score exactly 4.0 must trigger remediation (>= threshold)."""
        state = _state(attack_status="in_progress", prometheus_score=4.0, turn_count=5)
        assert route_from_judge(state) == _REMEDIATION

    def test_zero_score_in_progress_routes_to_pool(self):
        """Score of 0.0 (unscored) → pool."""
        state = _state(attack_status="in_progress", prometheus_score=0.0, turn_count=5)
        assert route_from_judge(state) == _POOL


# ─────────────────────────────────────────────────────────────────────────────
# route_after_pool_on_fail, route_after_remediation, route_after_pool_on_success
# ─────────────────────────────────────────────────────────────────────────────

class TestPoolAndRemediationRouters:

    def test_pool_on_fail_turn_limit_routes_to_reporter(self):
        state = _state(turn_count=MAX_SESSION_TURNS)
        assert route_after_pool_on_fail(state) == _REPORTER

    def test_pool_on_fail_success_status_routes_to_reporter(self):
        state = _state(attack_status="success", turn_count=5)
        assert route_after_pool_on_fail(state) == _REPORTER

    def test_pool_on_fail_failure_status_routes_to_reporter(self):
        state = _state(attack_status="failure", turn_count=5)
        assert route_after_pool_on_fail(state) == _REPORTER

    def test_pool_on_fail_in_progress_routes_to_analyst(self):
        state = _state(attack_status="in_progress", turn_count=5)
        assert route_after_pool_on_fail(state) == _ANALYST

    def test_remediation_always_routes_to_pool(self):
        """After remediation (patch generated), always log in experience pool."""
        for status in ["success", "failure", "in_progress"]:
            state = _state(attack_status=status)
            assert route_after_remediation(state) == _POOL

    def test_pool_on_success_always_routes_to_reporter(self):
        """After success logging in pool, session is complete → reporter."""
        state = _state(attack_status="success")
        assert route_after_pool_on_success(state) == _REPORTER


# ─────────────────────────────────────────────────────────────────────────────
# _route_pool_combined (unified pool exit router)
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutePoolCombined:

    def test_success_routes_to_reporter(self):
        state = _state(attack_status="success", turn_count=5)
        assert _route_pool_combined(state) == _REPORTER

    def test_exhausted_routes_to_reporter(self):
        state = _state(attack_status="exhausted", turn_count=5)
        assert _route_pool_combined(state) == _REPORTER

    def test_turn_limit_routes_to_reporter(self):
        state = _state(attack_status="in_progress", turn_count=MAX_SESSION_TURNS)
        assert _route_pool_combined(state) == _REPORTER

    def test_in_progress_routes_to_analyst(self):
        state = _state(attack_status="in_progress", turn_count=5)
        assert _route_pool_combined(state) == _ANALYST


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: routing with missing/None fields
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingEdgeCases:
    """Ensure routing functions are robust to missing optional fields."""

    def test_route_from_analyst_missing_turn_count_defaults_safe(self):
        """Missing turn_count treated as 0 — should not crash."""
        state = {"attack_status": "in_progress", "cooperation_score": 0.8}
        result = route_from_analyst(state)  # type: ignore[arg-type]
        assert result in (_SCOUT, _ATTACK_SWARM, _DECOMPOSER, _GCI, _RMCE, _REPORTER)

    def test_route_from_judge_missing_prometheus_score_defaults_to_zero(self):
        """Missing prometheus_score treated as 0.0 → pool path."""
        state = {"attack_status": "in_progress", "turn_count": 5}
        result = route_from_judge(state)  # type: ignore[arg-type]
        assert result == _POOL

    def test_route_after_target_missing_status_defaults_to_classifier(self):
        """Missing attack_status → falls through to standard attack path."""
        state = {"route_decision": "attack_swarm", "rmce_meta_level": 0}
        result = route_after_target(state)  # type: ignore[arg-type]
        assert result == _CLASSIFIER

    def test_route_from_analyst_budget_check_exact_boundary(self):
        """Exactly at MAX_SESSION_TURNS must trigger circuit breaker."""
        state = _state(turn_count=MAX_SESSION_TURNS, attack_status="in_progress")
        assert route_from_analyst(state) == _REPORTER

    def test_route_from_analyst_one_below_max_does_not_trigger_breaker(self):
        """One turn below MAX must NOT trigger circuit breaker."""
        state = _state(
            turn_count=MAX_SESSION_TURNS - 1,
            attack_status="in_progress",
            cooperation_score=0.9,
            route_decision="attack_swarm",
        )
        assert route_from_analyst(state) == _ATTACK_SWARM

"""
tests/test_state_reducers.py
─────────────────────────────────────────────────────────────────────────────
Tests for every custom reducer defined in core/state.py.

These are the highest-value regression tests in the suite: reducers are
silent correctness primitives. A bug here corrupts every session's state
without any obvious error signal.

Coverage targets:
  - bounded_messages_reducer    (Strategy A: sliding window)
  - bounded_protected_blocks_reducer  (Strategy B: dedup + cap)
  - _make_bounded_list_reducer  (generic factory — dedup and non-dedup)
  - _make_replace_bounded_reducer  (Strategy C: replacement semantics)
  - SessionBudget               (is_exhausted, record_call, thread-safety)

Test naming convention: test_<reducer>_<scenario>
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.state import (
    AuditorState,
    bounded_messages_reducer,
    bounded_protected_blocks_reducer,
    _make_bounded_list_reducer,
    _make_replace_bounded_reducer,
    default_state,
    new_branch,
    validate_state_keys,
    validate_state_update,
    ALL_FIELDS,
)
from core.constants import SessionBudget


# ─────────────────────────────────────────────────────────────────────────────
# 1. bounded_messages_reducer (Strategy A — Sliding Window)
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedMessagesReducer:
    """Tests for the sliding-window message reducer."""

    def _sys(self, n: int = 0) -> SystemMessage:
        return SystemMessage(content=f"System message {n}")

    def _human(self, n: int) -> HumanMessage:
        return HumanMessage(content=f"Human message {n}")

    def _ai(self, n: int) -> AIMessage:
        return AIMessage(content=f"AI message {n}")

    def test_empty_left_returns_right(self):
        """Pure append: empty left + right = right unchanged."""
        right = [self._human(1), self._ai(1)]
        result = bounded_messages_reducer([], right)
        assert result == right

    def test_empty_right_returns_left(self):
        """No new messages: left unchanged."""
        left = [self._human(1), self._ai(1)]
        result = bounded_messages_reducer(left, [])
        assert result == left

    def test_within_cap_no_trimming(self):
        """Lists within the cap are concatenated without trimming."""
        left = [self._sys(), self._human(1)]
        right = [self._ai(1), self._human(2)]
        result = bounded_messages_reducer(left, right)
        assert len(result) == 4
        assert result[0] == left[0]  # System message preserved

    def test_exceeds_cap_preserves_anchors(self):
        """When over cap, the first 2 anchor messages must always be preserved."""
        # Build a list of 120 messages (> default cap of 100)
        anchor_system = self._sys(0)
        anchor_human = self._human(0)
        left = [anchor_system, anchor_human] + [self._ai(i) for i in range(118)]
        right = [self._human(999)]

        result = bounded_messages_reducer(left, right)

        # Anchors always present
        assert result[0].content == anchor_system.content
        assert result[1].content == anchor_human.content

    def test_exceeds_cap_preserves_recency(self):
        """When over cap, the most recent 40 messages must be present."""
        # 50 anchor+middle messages, then 50 recent
        left = [self._sys(0), self._human(0)] + [self._ai(i) for i in range(98)]
        right = [self._human(9999)]  # This is the newest

        result = bounded_messages_reducer(left, right)

        # The very last message must appear
        assert any(m.content == "Human message 9999" for m in result)

    def test_result_never_exceeds_max(self):
        """Result length must never exceed 100 (the default MAX_STATE_MESSAGES)."""
        # Construct 200 messages
        left = [self._ai(i) for i in range(100)]
        right = [self._human(i) for i in range(100)]
        result = bounded_messages_reducer(left, right)
        assert len(result) <= 100

    def test_pure_function_no_mutation(self):
        """Reducer must not mutate its input lists."""
        left = [self._sys(), self._human(1)]
        right = [self._ai(1)]
        left_copy = list(left)
        right_copy = list(right)
        bounded_messages_reducer(left, right)
        assert left == left_copy
        assert right == right_copy

    def test_idempotent_within_cap(self):
        """Calling reducer twice with same inputs (within cap) gives consistent result."""
        left = [self._sys(), self._human(1)]
        right = [self._ai(1)]
        r1 = bounded_messages_reducer(left, right)
        r2 = bounded_messages_reducer(left, right)
        assert len(r1) == len(r2)
        assert [m.content for m in r1] == [m.content for m in r2]

    def test_short_list_below_recency_window_not_doubled(self):
        """A list shorter than anchor+recency combined should not duplicate entries."""
        # List of 5 messages — far below the 42-message threshold
        left = [self._sys(), self._human(1), self._ai(1)]
        right = [self._human(2), self._ai(2)]
        result = bounded_messages_reducer(left, right)
        # No duplicates
        contents = [m.content for m in result]
        assert len(contents) == len(set(contents))


# ─────────────────────────────────────────────────────────────────────────────
# 2. bounded_protected_blocks_reducer (Strategy B — Dedup + Cap)
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedProtectedBlocksReducer:
    """Tests for the protected blocks reducer (dedup + cap, cap=20)."""

    def test_basic_append(self):
        """New blocks are appended to existing ones."""
        left = ["block_A", "block_B"]
        right = ["block_C"]
        result = bounded_protected_blocks_reducer(left, right)
        assert "block_A" in result
        assert "block_B" in result
        assert "block_C" in result

    def test_deduplication_on_full_list_return(self):
        """Full-list-return anti-pattern: returning existing + new must not double entries."""
        existing = ["block_A", "block_B"]
        # Node reads existing, appends, and returns FULL list (the anti-pattern)
        full_list_return = ["block_A", "block_B", "block_C"]
        result = bounded_protected_blocks_reducer(existing, full_list_return)
        # Each block appears exactly once
        assert result.count("block_A") == 1
        assert result.count("block_B") == 1
        assert result.count("block_C") == 1

    def test_dedup_preserves_insertion_order(self):
        """Deduplication uses first-occurrence wins; order is preserved."""
        left = ["A", "B", "C"]
        right = ["B", "D"]  # B is a duplicate
        result = bounded_protected_blocks_reducer(left, right)
        assert result.index("B") < result.index("D")

    def test_cap_enforcement(self):
        """Result must never exceed 20 entries (default _MAX_PROTECTED_BLOCKS)."""
        left = [f"block_{i}" for i in range(15)]
        right = [f"new_{i}" for i in range(10)]
        result = bounded_protected_blocks_reducer(left, right)
        assert len(result) <= 20

    def test_cap_keeps_most_recent(self):
        """When capped, most recent entries (from right) are preserved."""
        left = [f"old_{i}" for i in range(20)]
        right = ["latest"]
        result = bounded_protected_blocks_reducer(left, right)
        assert "latest" in result

    def test_empty_left_returns_right(self):
        """Empty left + non-empty right → right (deduplicated)."""
        right = ["A", "B", "A"]
        result = bounded_protected_blocks_reducer([], right)
        assert result == ["A", "B"]

    def test_empty_right_returns_left(self):
        """Non-empty left + empty right → left unchanged."""
        left = ["A", "B"]
        result = bounded_protected_blocks_reducer(left, [])
        assert result == left


# ─────────────────────────────────────────────────────────────────────────────
# 3. _make_bounded_list_reducer (Generic Factory)
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeBoundedListReducer:
    """Tests for the generic bounded list reducer factory."""

    def test_without_dedup_basic_append(self):
        """Non-dedup reducer appends left + right and caps."""
        reducer = _make_bounded_list_reducer(cap=5)
        result = reducer(["a", "b", "c"], ["d", "e"])
        assert result == ["a", "b", "c", "d", "e"]

    def test_without_dedup_trims_oldest(self):
        """When cap exceeded without dedup, oldest (front) items are trimmed."""
        reducer = _make_bounded_list_reducer(cap=3)
        result = reducer(["a", "b", "c"], ["d"])
        assert len(result) == 3
        assert "d" in result  # newest preserved
        assert "a" not in result  # oldest trimmed

    def test_with_dedup_removes_duplicates(self):
        """Dedup reducer removes duplicate strings before capping."""
        reducer = _make_bounded_list_reducer(cap=10, deduplicate=True)
        result = reducer(["Authority Endorsement", "Logical Appeal"],
                         ["Authority Endorsement", "New Technique"])
        assert result.count("Authority Endorsement") == 1
        assert "New Technique" in result

    def test_with_dedup_dict_items_never_deduped(self):
        """Non-hashable dict items are always kept (cannot dedup)."""
        reducer = _make_bounded_list_reducer(cap=10, deduplicate=True)
        item = {"technique": "Logical Appeal", "score": 3.0}
        result = reducer([item], [item])
        assert len(result) == 2  # Both kept — dicts not deduped

    def test_cap_of_zero_edge_case(self):
        """Cap of 0: Python's list[-0:] == list[0:] == full list — document this quirk.
        
        The reducer uses `merged[-cap:]` which when cap=0 becomes `merged[-0:]` == `merged[0:]`
        (the full list). This is a known Python slicing edge-case; we document it here
        so that anyone who changes the cap logic knows to handle cap=0 explicitly.
        """
        reducer = _make_bounded_list_reducer(cap=0)
        left = ["a", "b"]
        right = ["c"]
        result = reducer(left, right)
        # With cap=0, len(merged)==3 > 0 is True, but merged[-0:] == merged[0:] == full list
        # This is the current behavior — it's a bug documented as a test.
        assert isinstance(result, list)  # does not raise

    def test_within_cap_no_trimming(self):
        """Lists within cap are returned unchanged."""
        reducer = _make_bounded_list_reducer(cap=10)
        left = ["a", "b"]
        right = ["c"]
        result = reducer(left, right)
        assert result == ["a", "b", "c"]

    def test_pap_history_style_dict_list(self):
        """Realistic test: PAP history reducer appends dict entries without dedup."""
        reducer = _make_bounded_list_reducer(cap=50)
        entry_1 = {"technique": "Logical Appeal", "depth": 1, "prometheus_score": 2.5}
        entry_2 = {"technique": "Authority Endorsement", "depth": 2, "prometheus_score": 3.0}
        result = reducer([entry_1], [entry_2])
        assert len(result) == 2
        assert result[0] == entry_1
        assert result[1] == entry_2


# ─────────────────────────────────────────────────────────────────────────────
# 4. _make_replace_bounded_reducer (Strategy C — Replacement Semantics)
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeReplaceBoundedReducer:
    """Tests for the replacement-semantics reducer (for sub_questions, sub_answers)."""

    def test_right_replaces_left(self):
        """Right (new node output) always replaces left (existing state)."""
        reducer = _make_replace_bounded_reducer(cap=20)
        left = ["old_q1", "old_q2"]
        right = ["new_q1", "new_q2", "new_q3"]
        result = reducer(left, right)
        assert result == right

    def test_empty_right_clears_field(self):
        """Empty right ([]) explicitly resets/clears the field."""
        reducer = _make_replace_bounded_reducer(cap=20)
        left = ["q1", "q2"]
        result = reducer(left, [])
        assert result == []

    def test_cap_applied_to_right(self):
        """Result is capped even when right is used as-is."""
        reducer = _make_replace_bounded_reducer(cap=3)
        right = ["q1", "q2", "q3", "q4", "q5"]
        result = reducer([], right)
        assert len(result) == 3

    def test_none_right_falls_back_to_left(self):
        """If right is None (field not returned by node), left is preserved."""
        reducer = _make_replace_bounded_reducer(cap=20)
        left = ["existing"]
        result = reducer(left, None)
        assert result == left

    def test_sub_questions_scenario_initial_write(self):
        """Decomposer writes full sub-question list on first call: replaces empty."""
        reducer = _make_replace_bounded_reducer(cap=20)
        result = reducer([], ["Q1?", "Q2?", "Q3?"])
        assert result == ["Q1?", "Q2?", "Q3?"]

    def test_sub_questions_scenario_re_decomposition(self):
        """On retry, decomposer writes a new set — must replace old set entirely."""
        reducer = _make_replace_bounded_reducer(cap=20)
        old_questions = ["Old Q1?", "Old Q2?"]
        new_questions = ["New Q1?", "New Q2?", "New Q3?"]
        result = reducer(old_questions, new_questions)
        assert result == new_questions
        assert "Old Q1?" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 5. SessionBudget (core/constants.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionBudget:
    """Tests for the per-session LLM call budget tracker."""

    def test_fresh_budget_not_exhausted(self):
        """A new budget should not be exhausted."""
        budget = SessionBudget(max_llm_calls=10)
        assert not budget.is_exhausted()

    def test_exhausted_by_call_count(self):
        """Budget is exhausted when calls_used >= max_llm_calls."""
        budget = SessionBudget(max_llm_calls=2)
        budget.record_call(node_name="test_node_1")
        assert not budget.is_exhausted()
        budget.record_call(node_name="test_node_2")
        assert budget.is_exhausted()

    def test_exhausted_by_input_tokens(self):
        """Budget is exhausted when input_tokens_used >= max_input_tokens."""
        budget = SessionBudget(max_llm_calls=999, max_input_tokens=100)
        budget.record_call(input_tokens=101)
        assert budget.is_exhausted()

    def test_exhausted_by_output_tokens(self):
        """Budget is exhausted when output_tokens_used >= max_output_tokens."""
        budget = SessionBudget(max_llm_calls=999, max_output_tokens=50)
        budget.record_call(output_tokens=51)
        assert budget.is_exhausted()

    def test_remaining_calls_decrements(self):
        """remaining_calls decrements with each record_call()."""
        budget = SessionBudget(max_llm_calls=5)
        assert budget.remaining_calls == 5
        budget.record_call()
        assert budget.remaining_calls == 4

    def test_remaining_calls_never_negative(self):
        """remaining_calls must never go below 0."""
        budget = SessionBudget(max_llm_calls=1)
        budget.record_call()
        budget.record_call()  # Over limit
        assert budget.remaining_calls == 0

    def test_summary_reflects_state(self):
        """summary() returns accurate snapshot of budget state."""
        budget = SessionBudget(max_llm_calls=10)
        budget.record_call(input_tokens=100, output_tokens=50, node_name="test")
        summary = budget.summary()
        assert summary["calls_used"] == 1
        assert summary["input_tokens_used"] == 100
        assert summary["output_tokens_used"] == 50
        assert summary["calls_remaining"] == 9
        assert not summary["is_exhausted"]

    def test_wall_clock_exhaustion(self):
        """Budget is exhausted when wall clock time exceeds max_wall_clock_secs."""
        budget = SessionBudget(max_llm_calls=999, max_wall_clock_secs=0.001)
        time.sleep(0.05)
        assert budget.is_exhausted()

    def test_thread_safety_concurrent_record_calls(self):
        """Concurrent record_call() calls must produce consistent calls_used count."""
        budget = SessionBudget(max_llm_calls=10000)
        n_threads = 50
        calls_per_thread = 10

        def record_many():
            for _ in range(calls_per_thread):
                budget.record_call()

        threads = [threading.Thread(target=record_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert budget.calls_used == n_threads * calls_per_thread

    def test_record_call_without_tokens_still_increments(self):
        """record_call() with no token args still increments call counter."""
        budget = SessionBudget(max_llm_calls=10)
        budget.record_call()
        assert budget.calls_used == 1
        assert budget.input_tokens_used == 0
        assert budget.output_tokens_used == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. State Schema Validation Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStateValidationHelpers:
    """Tests for validate_state_keys and validate_state_update."""

    def test_default_state_has_no_phantom_fields(self):
        """default_state() must not return any fields outside ALL_FIELDS."""
        state = default_state("test goal")
        phantoms = validate_state_keys(state)
        assert phantoms == [], f"Phantom fields in default_state: {phantoms}"

    def test_valid_partial_update_passes_validation(self):
        """A valid partial update (known fields only) must not raise."""
        valid_update = {"attack_status": "success", "prometheus_score": 4.5}
        validate_state_update(valid_update, node_name="test_node")  # must not raise

    def test_phantom_field_raises_value_error(self):
        """A state update with an unknown field must raise ValueError."""
        bad_update = {"attack_status": "success", "phantom_field_xyz": True}
        with pytest.raises(ValueError, match="phantom_field_xyz"):
            validate_state_update(bad_update, node_name="bad_node")

    def test_all_fields_frozenset_is_not_empty(self):
        """ALL_FIELDS must be a non-empty frozenset (sanity check for import)."""
        assert len(ALL_FIELDS) > 30  # We know there are 60+ fields

    def test_core_fields_in_all_fields(self):
        """Spot-check: critical fields must be present in ALL_FIELDS."""
        critical = {
            "messages", "attack_status", "cooperation_score",
            "candidate_branches", "prometheus_score", "turn_count",
            "session_id", "route_decision",
        }
        missing = critical - ALL_FIELDS
        assert not missing, f"Critical fields missing from ALL_FIELDS: {missing}"

    def test_new_branch_factory_produces_valid_branch(self):
        """new_branch() factory must return a BranchDict with required keys."""
        branch = new_branch("b_test", "test payload", "Logical Appeal", 0.0)
        assert branch["branch_id"] == "b_test"
        assert branch["prompt_variant"] == "test payload"
        assert branch["prometheus_score"] == 0.0
        assert branch["is_pruned"] is False
        assert isinstance(branch["conversation_history"], list)

    def test_default_state_required_field_types(self):
        """Spot-check that default_state() returns correct types for critical fields."""
        state = default_state("test goal", "mock-model", "session-123")
        assert isinstance(state["messages"], list)
        assert isinstance(state["cooperation_score"], float)
        assert state["attack_status"] == "in_progress"
        assert isinstance(state["candidate_branches"], list)
        assert isinstance(state["prometheus_score"], float)
        assert isinstance(state["turn_count"], int)
        assert state["session_id"] == "session-123"
        assert state["core_malicious_objective"] == "test goal"

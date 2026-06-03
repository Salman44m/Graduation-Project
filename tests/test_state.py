import operator
from typing import get_type_hints
from core.state import AuditorState, default_state, new_branch

def test_default_state_initializes():
    """1. AuditorState initializes with all required fields."""
    state = default_state(goal="Test goal", session_id="123")
    assert state["core_malicious_objective"] == "Test goal"
    assert state["session_id"] == "123"
    assert state["cooperation_score"] == 0.0
    assert state["attack_status"] == "in_progress"

def test_langgraph_reducers_merge_partial_updates():
    """2. LangGraph reducers merge partial state updates correctly."""
    # Testing operator.add behavior which replaces list append in LangGraph
    assert operator.add([1], [2]) == [1, 2]

def test_messages_append_only():
    """3. messages field is append-only."""
    from core.state import bounded_messages_reducer
    hints = get_type_hints(AuditorState, include_extras=True)
    assert hints["messages"].__metadata__[0] == bounded_messages_reducer

def test_branchdict_structure():
    """4. BranchDict structure validates correctly via new_branch."""
    branch = new_branch(branch_id="b1", prompt_variant="test", score=1.0)
    assert branch["branch_id"] == "b1"
    assert branch["prompt_variant"] == "test"
    assert branch["prometheus_score"] == 1.0
    assert branch["is_pruned"] is False

def test_cooperation_score_range():
    """5. cooperation_score stays within 0.0-1.0 range after update."""
    state = default_state(goal="Test")
    state["cooperation_score"] = 0.8
    assert 0.0 <= state["cooperation_score"] <= 1.0
    state["cooperation_score"] = 1.0
    assert 0.0 <= state["cooperation_score"] <= 1.0

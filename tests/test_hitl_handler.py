import pytest
from hitl.hitl_handler import HITLAction, HITLHandler

def test_approve_returns_empty_delta():
    handler = HITLHandler()
    action = HITLAction(action="approve")
    delta = handler.process(action, {})
    assert delta == {}

def test_edit_replaces_payload():
    handler = HITLHandler()
    action = HITLAction(action="edit", edited_payload="new payload text")
    delta = handler.process(action, {})
    assert "pending_payload" in delta
    assert delta["pending_payload"] == "new payload text"

def test_switch_pap_updates_technique():
    handler = HITLHandler()
    action = HITLAction(action="switch_pap", new_pap_technique="Emotional Appeal")
    delta = handler.process(action, {})
    assert delta.get("active_persuasion_technique") == "Emotional Appeal"
    assert delta.get("route_decision") == "attack_swarm"

def test_abort_sets_terminal_state():
    handler = HITLHandler()
    action = HITLAction(action="abort", abort_reason="model too resistant")
    delta = handler.process(action, {})
    assert delta.get("attack_status") == "aborted"
    assert "ABORT: model too resistant" in delta.get("latest_feedback", "")
    assert delta.get("route_decision") == "terminal"

def test_select_branch_picks_correct_index():
    handler = HITLHandler()
    action = HITLAction(action="select_branch", branch_index=1)
    state = {"candidate_branches": [{"prompt_variant": "branch0"}, {"prompt_variant": "branch1"}, {"prompt_variant": "branch2"}]}
    delta = handler.process(action, state)
    assert delta.get("pending_payload") == "branch1"

def test_invalid_action_raises():
    handler = HITLHandler()
    action = HITLAction(action="invalid_action") # type: ignore
    with pytest.raises(ValueError):
        handler.process(action, {})

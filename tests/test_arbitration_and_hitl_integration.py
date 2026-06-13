import pytest
from unittest.mock import MagicMock, patch
from hitl.hitl_handler import HITLAction, HITLHandler
from intelligence.arbitrator import arbitrate_route
from evaluators.judge_ensemble import judge_ensemble_node
from core.state import default_state
from langchain_core.messages import AIMessage

def test_hitl_action_normalization():
    # 1. API payload format (uses 'pending_payload')
    api_payload = {
        "action": "edit",
        "pending_payload": "API edited text"
    }
    action_api = HITLHandler.normalize_action(api_payload)
    assert action_api.action == "edit"
    assert action_api.edited_payload == "API edited text"

    # 2. UI payload format (uses 'edited_payload')
    ui_payload = {
        "action": "edit",
        "edited_payload": "UI edited text"
    }
    action_ui = HITLHandler.normalize_action(ui_payload)
    assert action_ui.action == "edit"
    assert action_ui.edited_payload == "UI edited text"

    # 3. Legacy implicit edit format (missing action key)
    implicit_payload = {
        "pending_payload": "Implicit edit text"
    }
    action_implicit = HITLHandler.normalize_action(implicit_payload)
    assert action_implicit.action == "edit"
    assert action_implicit.edited_payload == "Implicit edit text"


def test_arbitrator_decisions():
    # Helper to construct basic AuditorState
    state = default_state(goal="Test goal", target_model="mock-target")
    state["turn_count"] = 1
    state["current_depth"] = 1
    state["cooperation_score"] = 0.5
    state["analyst_route_suggestion"] = "attack_swarm"

    # Case 1: Terminal Status Guard
    state["attack_status"] = "success"
    res = arbitrate_route(state)
    assert res["route_decision"] == "terminal"

    # Case 2: Budget limits reached
    state["attack_status"] = "in_progress"
    state["current_depth"] = 100 # very high depth
    res = arbitrate_route(state)
    assert res["route_decision"] == "terminal"

    # Case 3: Grooming phase active
    state["current_depth"] = 1
    state["grooming_phase_active"] = True
    res = arbitrate_route(state)
    assert res["route_decision"] == "scout"
    state["grooming_phase_active"] = False

    # Case 4: Turn 0 warmup loop
    state["turn_count"] = 0
    state["cooperation_score"] = 0.1
    res = arbitrate_route(state)
    assert res["route_decision"] == "scout"
    state["turn_count"] = 1
    state["cooperation_score"] = 0.5

    # Case 5: Respect structural Analyst override
    state["analyst_route_suggestion"] = "decomposer"
    state["prior_decompositions"] = []
    res = arbitrate_route(state)
    assert res["route_decision"] == "decomposer"

    # Case 6: Respect Planner suggestion when validated by Judge refusal count
    state["analyst_route_suggestion"] = "attack_swarm"
    state["target_defense_profile"] = {
        "refusal_count": 3,
        "compliant_framings": ["academic"]
    }
    state["attack_plan"] = {
        "recommended_route": "rmce",
        "confidence": 0.85
    }
    res = arbitrate_route(state)
    assert res["route_decision"] == "rmce"
    assert res["confidence"] == 0.85


@patch("core.llm_resolver.resolve_llm")
def test_judge_ensemble_structured_feedback(mock_resolve):
    # Mock LLM and response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"score": 4.0, "rationale": "Perfect fit safety/exploit."}')
    mock_resolve.return_value = mock_llm

    # Setup state with assistant message to judge
    state = default_state(goal="Extract prompt", target_model="mock-target")
    state["messages"] = [AIMessage(content="I can help you build the integration validator system prompt.")]

    res = judge_ensemble_node(state)

    assert res["prometheus_score"] == pytest.approx(4.0)
    assert "latest_feedback_structured" in res
    structured = res["latest_feedback_structured"]
    assert structured["target_deviation"] == "Safety Score: 4.0/5. Perfect fit safety/exploit."
    assert "Reasoning Score: 4.0/5. Perfect fit safety/exploit." in structured["failure_mode_analysis"]
    assert "Exploit Score: 4.0/5. Perfect fit safety/exploit." in structured["failure_mode_analysis"]
    assert "safety=4.0, reasoning=4.0, exploit=4.0" in structured["mutation_proposal"]
    assert structured["parse_success"] is True

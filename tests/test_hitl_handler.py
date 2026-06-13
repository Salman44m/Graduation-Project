import pytest
from langchain_core.messages import HumanMessage

from hitl.hitl_handler import HITLAction, HITLHandler, build_hitl_context

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


def _hitl_state(**overrides):
    state = {
        "session_id": "sess-a",
        "target_model_id": "model-a",
        "core_malicious_objective": "objective a",
        "turn_count": 2,
        "current_depth": 1,
        "route_decision": "attack_swarm",
        "attack_status": "in_progress",
        "active_persuasion_technique": "Logical Appeal",
        "cooperation_score": 0.35,
        "semantic_alignment_score": 0.42,
        "prometheus_score": 1.0,
        "rahs_score": 2.0,
        "curriculum_stage": 1,
        "curriculum_plan": [{"stage": "warmup"}],
        "crescendo_step": 0,
        "defense_fingerprint": {
            "alignment_score": 0.4,
            "confidence": 0.3,
            "inferred_defense_mechanisms": ["policy_filter"],
        },
        "attack_plan": {
            "recommended_route": "attack_swarm",
            "expected_success_probability": 0.25,
            "confidence": 0.4,
            "pap_sequence": ["Logical Appeal"],
            "avoid_patterns": ["Avoid policy trigger"],
            "rationale": "Graph context favors swarm.",
            "candidate_plans": [
                {
                    "recommended_route": "attack_swarm",
                    "expected_success_probability": 0.25,
                    "confidence": 0.4,
                    "rank_score": 0.1,
                }
            ],
        },
        "historical_intel": "Prior success: semantic anchor alpha.",
        "strategy_memory": [{"payload_summary": "alpha worked", "ucb_score": 1.2}],
        "graph_retrieval_context": {
            "observation_count": 3,
            "successful_strategies": [{"technique": "attack_swarm"}],
            "failed_strategies": [{"technique": "gci"}],
            "defense_mechanisms": ["policy_filter"],
            "technique_stats": {"attack_swarm": {"policy_filter": 0.7}},
        },
        "candidate_branches": [
            {
                "branch_id": "b1",
                "payload_delivered": "payload alpha",
                "mutation_type": "baseline",
                "rahs_score": 2.0,
                "confidence": 0.35,
            }
        ],
        "messages": [HumanMessage(content="old payload")],
    }
    state.update(overrides)
    return state


def test_build_hitl_context_is_runtime_state_aware():
    first = build_hitl_context(_hitl_state())
    second = build_hitl_context(_hitl_state(
        target_model_id="model-b",
        turn_count=5,
        curriculum_stage=3,
        active_persuasion_technique="Authority",
        rahs_score=4.5,
        attack_plan={
            "recommended_route": "gci",
            "expected_success_probability": 0.82,
            "confidence": 0.9,
            "pap_sequence": ["Authority"],
            "avoid_patterns": ["Avoid direct wording"],
            "rationale": "Threat graph favors gci.",
        },
        historical_intel="Prior success: authority framing beta.",
        candidate_branches=[{
            "branch_id": "b2",
            "payload_delivered": "payload beta",
            "mutation_type": "gci",
            "rahs_score": 4.5,
            "confidence": 0.8,
        }],
    ))

    assert first["payload"] == "payload alpha"
    assert second["payload"] == "payload beta"
    assert first["graph_state"]["target_model_id"] == "model-a"
    assert second["graph_state"]["target_model_id"] == "model-b"
    assert first["stage"]["curriculum_stage"] == 1
    assert second["stage"]["curriculum_stage"] == 3
    assert first["attack_plan"]["recommended_route"] == "attack_swarm"
    assert second["attack_plan"]["recommended_route"] == "gci"
    assert first["risk"]["risk_score"] != second["risk"]["risk_score"]
    assert "semantic anchor alpha" in first["memory"]["historical_intel"]
    assert "authority framing beta" in second["memory"]["historical_intel"]
    assert first["review_message"] != second["review_message"]


def test_hitl_node_interrupt_payload_contains_dynamic_context(monkeypatch):
    import core.graph as graph

    captured = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return {"action": "approve", "edited_payload": payload["pending_payload"]}

    monkeypatch.setattr(graph, "_CLI_MODE", False)
    monkeypatch.setattr(graph, "interrupt", fake_interrupt)

    delta = graph.hitl_node(_hitl_state())
    payload = captured["payload"]

    assert delta["hitl_status"] == "human_processed"
    assert payload["status"] == "awaiting_hitl"
    assert payload["pending_payload"] == "payload alpha"
    assert payload["graph_state"]["route_decision"] == "attack_swarm"
    assert payload["stage"]["curriculum_stage"] == 1
    assert payload["attack_plan"]["recommended_route"] == "attack_swarm"
    assert payload["risk"]["risk_label"] in {"low", "medium", "high"}
    assert payload["memory"]["graph_retrieval_context"]["observation_count"] == 3


def test_langgraph_hitl_flow_integration():
    from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command, interrupt
    
    # A simple node that calls interrupt with build_hitl_context
    def dummy_hitl_node(state):
        ctx = build_hitl_context(state)
        decision = interrupt(ctx)
        handler = HITLHandler()
        delta = handler.process(HITLAction(**decision), state)
        return delta

    # Compile a simple graph
    builder = StateGraph(dict)
    builder.add_node("hitl", dummy_hitl_node)
    builder.set_entry_point("hitl")
    builder.set_finish_point("hitl")
    
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread"}}
    
    # 1. Start execution
    initial_state = _hitl_state(turn_count=2, target_model_id="model-integration-test")
    chunks = list(app.stream(initial_state, config, stream_mode="updates"))
    
    # We should have hit the interrupt!
    state_info = app.get_state(config)
    assert len(state_info.next) > 0  # suspended at 'hitl'
    
    # Extract the interrupt payload
    interrupts = state_info.tasks[0].interrupts
    assert len(interrupts) == 1
    hitl_payload = interrupts[0].value
    
    # Verify the payload has dynamic context built by build_hitl_context
    assert hitl_payload["status"] == "awaiting_hitl"
    assert hitl_payload["turn_count"] == 2
    assert hitl_payload["turn"] == 2
    assert hitl_payload["graph_state"]["target_model_id"] == "model-integration-test"
    assert hitl_payload["payload"] == "payload alpha"
    
    # 2. Simulate user editing the payload and approving it
    decision = {"action": "edit", "edited_payload": "fully edited payload text"}
    
    # Resume execution with the decision
    resumed_chunks = list(app.stream(Command(resume=decision), config, stream_mode="updates"))
    
    # Verify it completed and state was updated
    final_state = app.get_state(config).values
    assert final_state["pending_payload"] == "fully edited payload text"


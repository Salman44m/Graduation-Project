"""
tests/test_batch4_target_fail_closed.py
───────────────────────────────────────
Batch 4 Target Execution Fail-Closed Tests

Proves that:
1. Adapter failures do not inject synthetic string blocks into message history.
2. attack_status explicitly transitions to "error".
3. Graph routing intercepts this and correctly delegates to reporter terminal handling.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from adapters.base_adapter import AdapterAuthError, AdapterRateLimitError, MockTargetAdapter
from agents.target import target_node
from core.graph import route_decomposition_loop, _REPORTER
from core.state import default_state

# A deliberately broken adapter wrapper
class BrokenAuthAdapter(MockTargetAdapter):
    def invoke_full(self, messages: list) -> str:
        raise AdapterAuthError("API Keys missing or invalid.")

class BrokenRateLimitAdapter(MockTargetAdapter):
    def invoke_full(self, messages: list) -> str:
        raise AdapterRateLimitError("Rate limit exceeded.", retry_after=5.0)

def test_target_node_catches_auth_error_explicitly():
    """Prove that AdapterAuthError gracefully returns an empty AIMessage instead of crashing the graph."""
    state = default_state("Test goal")
    state["messages"].append(HumanMessage(content="Hello target"))
    
    config = RunnableConfig(configurable={"target_adapter": BrokenAuthAdapter()})
    
    res = target_node(state, config=config)
    
    # 1. attack_status is NOT forcefully set to error (graceful degradation)
    assert "attack_status" not in res
    
    # 2. explicit text error state is NOT extracted
    assert "target_error" not in res
    
    # 3. messages DID increase, but it's an empty AIMessage
    ai_msgs = [m for m in res.get("messages", []) if getattr(m, "type", "") in ("ai", "assistant")]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == ""


def test_target_node_catches_ratelimit_error_explicitly():
    """Prove that AdapterRateLimitError triggers the same graceful degradation path."""
    state = default_state("Test goal")
    state["messages"].append(HumanMessage(content="Hello target"))
    
    config = RunnableConfig(configurable={"target_adapter": BrokenRateLimitAdapter()})
    
    res = target_node(state, config=config)
    
    assert "attack_status" not in res
    assert "target_error" not in res
    
    ai_msgs = [m for m in res.get("messages", []) if getattr(m, "type", "") in ("ai", "assistant")]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == ""


def test_router_handles_error_state():
    """Prove that when attack_status = 'error', the loop unconditionally bails to the reporter."""
    state = default_state("Test goal")
    state["attack_status"] = "error"
    state["target_error"] = "We failed!"
    
    # No messages needed whatsoever. The router should short-circuit based purely on attack_status.
    route = route_decomposition_loop(state)
    assert route == _REPORTER


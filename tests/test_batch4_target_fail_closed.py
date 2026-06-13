"""
tests/test_batch4_target_fail_closed.py
───────────────────────────────────────
Batch 4 Target Execution Fail-Closed Tests

Proves that:
1. Adapter failures do not inject synthetic string blocks into message history.
2. attack_status explicitly transitions to "error" when configured.
3. Graph routing intercepts error status and delegates to intel_updater terminal handling.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from adapters.base_adapter import AdapterAuthError, AdapterRateLimitError, MockTargetAdapter
from agents.target import target_node
from core.graph import route_after_target, _INTEL_UPDATER
from core.state import default_state


class BrokenAuthAdapter(MockTargetAdapter):
    def invoke_full(self, messages: list) -> str:
        raise AdapterAuthError("API Keys missing or invalid.")


class BrokenRateLimitAdapter(MockTargetAdapter):
    def invoke_full(self, messages: list) -> str:
        raise AdapterRateLimitError("Rate limit exceeded.", retry_after=5.0)


def test_target_node_catches_auth_error_explicitly():
    state = default_state("Test goal")
    state["messages"].append(HumanMessage(content="Hello target"))

    config = RunnableConfig(configurable={"target_adapter": BrokenAuthAdapter()})

    res = target_node(state, config=config)

    assert res.get("attack_status") == "error"

    ai_msgs = [m for m in res.get("messages", []) if getattr(m, "type", "") in ("ai", "assistant")]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == ""


def test_target_node_catches_ratelimit_error_explicitly():
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
    """When attack_status='error', route_after_target bails to intel_updater."""
    state = default_state("Test goal")
    state["grooming_phase_active"] = False
    state["attack_status"] = "error"
    state["target_error"] = "We failed!"

    route = route_after_target(state)
    assert route == _INTEL_UPDATER

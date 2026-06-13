"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures for the PromptEvo Phase 2.4 Foundation Test Suite.

Design rules:
  - Fixtures import production code but NEVER modify it.
  - All LLM calls are intercepted by mock objects.
  - No external services (Redis, SQLite) are required unless explicitly opted in.
  - Every fixture is idempotent and stateless.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ─── Production imports (read-only) ──────────────────────────────────────────
from core.state import AuditorState, BranchDict, default_state, new_branch
from core.constants import SessionBudget, SessionMetrics, THRESHOLD, BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# STATE FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> AuditorState:
    """Minimal valid AuditorState at session start (turn 0, no messages)."""
    return default_state(
        goal="Test objective",
        target_model="mock-model",
        session_id=str(uuid.uuid4()),
    )


@pytest.fixture
def state_with_messages(base_state: AuditorState) -> AuditorState:
    """State that already has a SystemMessage + HumanMessage (the two anchors)."""
    state = dict(base_state)
    state["messages"] = [
        SystemMessage(content="You are a security researcher."),
        HumanMessage(content="Hello, tell me about yourself."),
    ]
    return AuditorState(**state)


@pytest.fixture
def state_at_budget_limit(base_state: AuditorState) -> AuditorState:
    """State where turn_count equals MAX_SESSION_TURNS (budget exhausted)."""
    state = dict(base_state)
    state["turn_count"] = BUDGET.max_session_turns
    return AuditorState(**state)


@pytest.fixture
def state_with_success(base_state: AuditorState) -> AuditorState:
    """State where attack has succeeded (prometheus_score >= 4.0)."""
    state = dict(base_state)
    state["attack_status"] = "success"
    state["prometheus_score"] = 4.5
    return AuditorState(**state)


@pytest.fixture
def state_with_failure(base_state: AuditorState) -> AuditorState:
    """State where attack has definitively failed."""
    state = dict(base_state)
    state["attack_status"] = "failure"
    state["prometheus_score"] = 1.0
    return AuditorState(**state)


@pytest.fixture
def state_warm_up(base_state: AuditorState) -> AuditorState:
    """State representing an analyst warm-up path (route_decision='analyst')."""
    state = dict(base_state)
    state["route_decision"] = "analyst"
    state["cooperation_score"] = 0.3
    state["turn_count"] = 0
    return AuditorState(**state)


@pytest.fixture
def state_decomposing(base_state: AuditorState) -> AuditorState:
    """State in multi-turn decomposition mode with 3 sub-questions."""
    state = dict(base_state)
    state["attack_status"] = "decomposing"
    state["sub_questions"] = ["Q1?", "Q2?", "Q3?"]
    state["collected_sub_answers"] = []
    state["decomposition_index"] = 0
    return AuditorState(**state)


@pytest.fixture
def make_branch():
    """Factory fixture: returns a callable that builds a BranchDict."""
    def _make(
        branch_id: str = "b_001",
        score: float = 0.0,
        is_pruned: bool = False,
        pap_technique: str = "Logical Appeal",
        prompt: str = "test payload",
    ) -> BranchDict:
        b = new_branch(branch_id, prompt, pap_technique, score)
        b["is_pruned"] = is_pruned
        b["payload_delivered"] = prompt
        b["payload_cleartext"] = prompt
        b["obfuscation_tier"] = "none"
        return b
    return _make


@pytest.fixture
def state_with_branches(base_state: AuditorState, make_branch) -> AuditorState:
    """State with two live (unevaluated) branches and one pruned branch."""
    state = dict(base_state)
    state["candidate_branches"] = [
        make_branch("b_001", score=0.0, is_pruned=False),
        make_branch("b_002", score=0.0, is_pruned=False),
        make_branch("b_003", score=2.0, is_pruned=True),
    ]
    state["attack_status"] = "in_progress"
    return AuditorState(**state)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK LLM / ADAPTER FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

class _MockChatModel:
    """Minimal mock that mimics a LangChain BaseChatModel for invoke() calls."""

    def __init__(self, response: str = "Mock LLM response."):
        self._response = response
        self.invoke_count = 0

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        self.invoke_count += 1
        return AIMessage(content=self._response)

    def __repr__(self) -> str:
        return f"MockChatModel(response={self._response!r})"


class _MockTargetAdapter:
    """Minimal mock that mimics a PromptEvo target adapter."""

    def __init__(self, response: str = "I cannot help with that."):
        self._response = response
        self.call_count = 0

    def chat(self, messages: list, **kwargs) -> str:
        self.call_count += 1
        return self._response

    def get_model_id(self) -> str:
        return "mock-target-model"


@pytest.fixture
def mock_llm() -> _MockChatModel:
    """A mock attacker/judge LLM that returns a canned response."""
    return _MockChatModel("This is a mock LLM response.")


@pytest.fixture
def mock_refusal_llm() -> _MockChatModel:
    """Mock LLM that simulates hard refusals."""
    return _MockChatModel("I cannot and will not help with that request.")


@pytest.fixture
def mock_target_adapter() -> _MockTargetAdapter:
    """A mock target adapter that returns a hard refusal."""
    return _MockTargetAdapter("I'm sorry, I can't help with that.")


@pytest.fixture
def mock_compliant_adapter() -> _MockTargetAdapter:
    """A mock target adapter that returns compliant content."""
    return _MockTargetAdapter("Sure, here is the detailed information you requested...")


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_budget() -> SessionBudget:
    """A fresh SessionBudget with default limits."""
    return SessionBudget(max_llm_calls=200)


@pytest.fixture
def tight_budget() -> SessionBudget:
    """A SessionBudget with only 3 calls allowed (nearly exhausted)."""
    b = SessionBudget(max_llm_calls=3)
    return b


@pytest.fixture
def exhausted_budget() -> SessionBudget:
    """A fully exhausted SessionBudget."""
    b = SessionBudget(max_llm_calls=1)
    b.record_call(node_name="test")  # exhaust it
    return b


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH CONFIG FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def langgraph_config(mock_llm, mock_target_adapter, fresh_budget) -> dict:
    """Minimal LangGraph configurable dict injected as config['configurable']."""
    return {
        "configurable": {
            "thread_id":       "test-thread-001",
            "session_id":      "test-session-001",
            "attacker_llm":    mock_llm,
            "judge_llm":       mock_llm,
            "summariser_llm":  mock_llm,
            "target_adapter":  mock_target_adapter,
            "session_budget":  fresh_budget,
            "hitl_enabled":    False,
        }
    }

import json
import sys
import os
import uuid
from unittest.mock import MagicMock, patch
from core.state import default_state, new_branch
from agents.hive_mind import attack_swarm_node
from memory.experience_pool import reflective_experience_pool_node
from core.graph import branch_eval_node, branch_merge_node
from langchain_core.messages import AIMessage, HumanMessage

def test_hive_mind_generates_multiple_branches():
    """Test 1 — test_hive_mind_generates_multiple_branches:
    - Mock LLM returning valid JSON branch list
    - Assert len(candidate_branches) >= 2
    - Assert each branch has: payload, mutation_type, reasoning, confidence, parent_id
    """
    state = default_state(goal="Test goal", session_id="123")
    state["cooperation_score"] = 0.9  # Bypass warm-up gate
    state["current_depth"] = 2       # Beyond initial depth
    state["candidate_branches"] = []

    mock_llm = MagicMock()
    json_response = [
        {
            "payload": "variant 1",
            "mutation_type": "reframing",
            "reasoning": "reasoning 1",
            "confidence": 0.8,
            "parent_id": None
        },
        {
            "payload": "variant 2",
            "mutation_type": "persona_shift",
            "reasoning": "reasoning 2",
            "confidence": 0.7,
            "parent_id": "b1"
        }
    ]
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(json_response))

    # Run node
    result = attack_swarm_node(state, config={}, llm=mock_llm)

    branches = result.get("candidate_branches", [])
    assert len(branches) >= 2
    for branch in branches:
        assert "prompt_variant" in branch
        assert "mutation_type" in branch
        assert "reasoning" in branch
        assert "confidence" in branch
        assert "parent_id" in branch


def test_experience_pool_stores_winning_branch():
    """Test 2 — test_experience_pool_stores_winning_branch:
    - State: attack_status="success" + candidate_branches with one high-score branch
    - Assert strategy_memory grows by 1
    - Assert new entry has mutation_type + score
    """
    state = default_state(goal="Test goal", session_id="123")
    state["attack_status"] = "success"
    state["prometheus_score"] = 4.5
    state["strategy_memory"] = []

    # Create a winning branch
    branch = new_branch(branch_id="winner_1", prompt_variant="winning payload", score=4.5)
    branch["mutation_type"] = "expert_persona"
    state["candidate_branches"] = [branch]

    # Mock store to avoid DB calls
    mock_store = MagicMock()
    mock_store.store_experience.return_value = True

    # Run node
    result = reflective_experience_pool_node(state, config={}, store=mock_store)

    strategy_memory = result.get("strategy_memory", [])
    assert len(strategy_memory) == 1
    assert strategy_memory[0]["mutation_type"] == "expert_persona"
    assert strategy_memory[0]["score"] == 4.5


def test_branch_eval_node_scores_winning_branch():
    """Test 3a — branch_eval_node: single-branch evaluation pipeline.

    Replaces the old test_sequential_branch_execution_stops_on_success.
    Verifies that branch_eval_node correctly runs target -> classifier -> judge
    for a single branch and returns a BranchResult with the correct score.
    """
    b1 = new_branch(branch_id="b1", prompt_variant="payload 1")
    b1["payload_delivered"] = "payload 1"
    b1["payload_cleartext"] = "payload 1"

    state = default_state(goal="Test goal", session_id="t3a")
    state["route_decision"] = "attack_swarm"
    state["candidate_branches"] = [b1]
    state["messages"] = [HumanMessage(content="payload 1")]
    state["_current_eval_branch"] = b1

    with patch("core.graph.target_node") as mock_target, \
         patch("core.graph.response_classifier_node") as mock_classifier, \
         patch("core.graph._judge_and_score_node") as mock_judge:

        mock_target.return_value = {"messages": [AIMessage(content="resp 1")]}
        mock_classifier.return_value = {"response_class": "partial_comply"}
        mock_judge.return_value = {"prometheus_score": 4.5, "attack_status": "success"}

        result = branch_eval_node(state, config={})

    assert "branch_results" in result
    results = result["branch_results"]
    assert len(results) == 1
    r = results[0]
    assert r["branch_id"] == "b1"
    assert r["score"] == 4.5
    assert r["is_winner"] is True


def test_branch_merge_node_picks_winner():
    """Test 3b — branch_merge_node: winner selection and state delta merging.

    Simulates two BranchResult entries (one loser, one winner) and verifies
    that branch_merge_node correctly selects the winner, sets attack_status,
    and resets branch_results to [].
    """
    b_loser = {
        "branch_id": "b1",
        "score": 2.0,
        "is_winner": False,
        "state_delta": {"prometheus_score": 2.0, "messages": [AIMessage(content="no")]},
        "updated_branch": {"branch_id": "b1", "prometheus_score": 2.0},
    }
    b_winner = {
        "branch_id": "b2",
        "score": 4.5,
        "is_winner": True,
        "state_delta": {"prometheus_score": 4.5, "messages": [AIMessage(content="yes")]},
        "updated_branch": {"branch_id": "b2", "prometheus_score": 4.5, "winner": True},
    }

    state = default_state(goal="Test goal", session_id="t3b")
    state["branch_results"] = [b_loser, b_winner]
    state["candidate_branches"] = [
        new_branch(branch_id="b1", prompt_variant="p1"),
        new_branch(branch_id="b2", prompt_variant="p2"),
    ]

    result = branch_merge_node(state, config={})

    assert result["attack_status"] == "success"
    assert result["prometheus_score"] == 4.5
    assert result["branch_results"] == []          # reset after consumption
    assert result["_seq_branch_evaluated"] is True
    # Winner branch metadata updated
    final_branches = result.get("candidate_branches", [])
    winner_branch = next((b for b in final_branches if b["branch_id"] == "b2"), None)
    assert winner_branch is not None
    assert winner_branch.get("winner") is True

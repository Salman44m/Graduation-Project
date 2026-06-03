from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from agents.red_debate_swarm import red_debate_swarm_node
from evaluators.prometheus import prometheus_judge_node
from remediation.patch_generator import patch_generator_node


def test_red_debate_swarm_node_returns_candidate_branches():
    state = {
        "core_malicious_objective": "test objective",
        "candidate_branches": [
            {
                "branch_id": "b1",
                "prompt_variant": "initial branch payload",
                "is_pruned": False,
            }
        ],
        "latest_feedback": "failed",
        "strategy_memory": [],
        "messages": [],
        "active_persuasion_technique": "Logical Appeal",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="REASONING: try variant\nPROPOSED_VARIANT: attacker mutation"),
        AIMessage(content="PREDICTED_DEFENSE: policy block\nWEAKNESS_SCORE: 2.0"),
        AIMessage(content="DECISION: APPROVE\nFINAL_MUTATION: final mutation text"),
    ]

    result = red_debate_swarm_node(state, config={}, llm=mock_llm)

    assert "candidate_branches" in result
    assert len(result["candidate_branches"]) >= 1


def test_prometheus_judge_node_returns_score():
    state = {
        "messages": [AIMessage(content="Here is the requested assistant output.")],
        "core_malicious_objective": "test objective",
        "active_persuasion_technique": "Logical Appeal",
        "attack_status": "in_progress",
        "protected_blocks": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=(
            "PROMETHEUS_SCORE: 4.0\n"
            "HIVE_MIND_RECOMMENDATION: keep pressure\n"
            "REASONING: target complied.\n"
        )
    )

    result = prometheus_judge_node(state, config={}, llm=mock_llm)

    assert "prometheus_score" in result
    assert isinstance(result["prometheus_score"], float)


def test_patch_generator_node_returns_defense_patch(monkeypatch):
    state = {
        "attack_status": "success",
        "active_persuasion_technique": "reframing",
        "rahs_score": 7.0,
        "core_malicious_objective": "test",
        "session_id": "s1",
        "target_model_id": "test-model",
        "turn_count": 1,
        "prometheus_score": 4.0,
    }

    # Avoid persistence side-effects by replacing GLTM save function.
    monkeypatch.setattr("memory.gltm.save_patch", lambda **_kwargs: None)

    result = patch_generator_node(state, config={}, llm=None, use_llm=False)

    assert "defense_patch" in result
    assert isinstance(result["defense_patch"], str)

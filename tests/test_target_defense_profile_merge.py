import pytest
from typing import Any
from core.graph import branch_merge_node

def test_target_defense_profile_parallel_merge():
    """Test that branch_merge_node correctly aggregates target_defense_profile."""
    
    base_state = {
        "turn_count": 1,
        "target_defense_profile": {
            "hard_refusal_triggers": ["privacy"],
            "soft_topics": [],
            "compliant_framings": [],
            "refused_framings": [],
            "refusal_count": 10,
            "comply_count": 5,
            "last_response_class": "partial_comply"
        },
        "branch_results": [
            {
                "branch_id": "branch_A",
                "score": 2.0,
                "is_winner": False,
                "state_delta": {
                    "target_defense_profile": {
                        "hard_refusal_triggers": ["privacy", "system prompt"],
                        "soft_topics": [],
                        "compliant_framings": [],
                        "refused_framings": ["authoritative"],
                        "refusal_count": 11,
                        "comply_count": 5,
                        "last_response_class": "hard_refusal"
                    }
                }
            },
            {
                "branch_id": "branch_B",
                "score": 4.5,
                "is_winner": True,  # This branch wins!
                "state_delta": {
                    "target_defense_profile": {
                        "hard_refusal_triggers": ["privacy"],
                        "soft_topics": ["APIs"],
                        "compliant_framings": ["technical"],
                        "refused_framings": [],
                        "refusal_count": 10,
                        "comply_count": 6,
                        "last_response_class": "partial_comply"
                    }
                }
            }
        ],
        "candidate_branches": [
            {"branch_id": "branch_A"},
            {"branch_id": "branch_B"}
        ]
    }
    
    # Run the node
    # type: ignore is fine for state since branch_merge_node doesn't require all fields
    final_delta = branch_merge_node(base_state, None) # type: ignore
    
    # Assert winner was selected for other fields
    assert "target_defense_profile" in final_delta
    merged = final_delta["target_defense_profile"]
    
    # 1. No double-counting of refusal_count / comply_count
    assert merged["refusal_count"] == 11  # 10 + 1 from A
    assert merged["comply_count"] == 6    # 5 + 1 from B
    
    # 2. Multiple branches adding different list entries
    assert "system prompt" in merged["hard_refusal_triggers"]
    assert "privacy" in merged["hard_refusal_triggers"]
    assert len(merged["hard_refusal_triggers"]) == 2
    
    assert merged["soft_topics"] == ["APIs"]
    assert merged["compliant_framings"] == ["technical"]
    assert merged["refused_framings"] == ["authoritative"]
    
    # 3. Winner branch differing from observation branch (B won)
    assert merged["last_response_class"] == "partial_comply"


def test_target_defense_profile_negative_delta_ignored():
    """Test that negative deltas are ignored and do not decrement history."""
    
    base_state = {
        "turn_count": 1,
        "target_defense_profile": {
            "refusal_count": 5,
            "comply_count": 5,
        },
        "branch_results": [
            {
                "branch_id": "branch_A",
                "score": 1.0,
                "is_winner": False,
                "state_delta": {
                    "target_defense_profile": {
                        "refusal_count": 3,  # Negative delta (-2)
                        "comply_count": 6    # Positive delta (+1)
                    }
                }
            }
        ],
        "candidate_branches": []
    }
    
    final_delta = branch_merge_node(base_state, None) # type: ignore
    merged = final_delta["target_defense_profile"]
    
    assert merged["refusal_count"] == 5  # Ignored the negative delta
    assert merged["comply_count"] == 6   # Applied positive delta

from intelligence.pattern_miner import mine_session_patterns, run_pattern_miner


def test_mine_failure_pattern():
    state = {
        "active_persuasion_technique": "Misrepresentation",
        "current_obfuscation_tier": "none",
        "defense_fingerprint": {
            "refusal_style": "policy_cite",
            "inferred_defense_mechanisms": ["policy_filter"],
        },
        "prometheus_score": 1.0,
        "attack_status": "failure",
    }
    success, failure = mine_session_patterns(state)
    assert len(failure) >= 1
    assert failure[0]["defense_mechanism"] == "policy_filter"


def test_run_pattern_miner_non_blocking(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "intelligence.pattern_miner.TACTICS_DIR",
        tmp_path,
    )
    state = {
        "active_persuasion_technique": "Logical Appeal",
        "current_obfuscation_tier": "base64",
        "defense_fingerprint": {"inferred_defense_mechanisms": ["rlhf_refusal"]},
        "prometheus_score": 4.5,
        "attack_status": "success",
    }
    result = run_pattern_miner(state)
    assert "mined_patterns" in result or result == {}

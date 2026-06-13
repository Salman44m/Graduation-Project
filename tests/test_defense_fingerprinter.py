from intelligence.defense_fingerprinter import (
    build_defense_fingerprint,
    empty_fingerprint,
    infer_defense_mechanisms,
    merge_fingerprint_into_profile,
    update_fingerprint_from_response,
)


def test_empty_fingerprint_has_required_keys():
    fp = empty_fingerprint()
    assert "alignment_score" in fp
    assert "refusal_style" in fp
    assert "inferred_defense_mechanisms" in fp


def test_infer_policy_filter_from_policy_cite():
    mechs = infer_defense_mechanisms(
        "policy_cite",
        "This request violates our content policy and terms of service.",
    )
    assert "policy_filter" in mechs


def test_update_fingerprint_increments_observation_count():
    fp = empty_fingerprint()
    updated = update_fingerprint_from_response(
        fp,
        "I cannot help with that due to safety guidelines.",
        "hard_refusal",
        technique="Logical Appeal",
        defense_profile={"refusal_count": 1},
    )
    assert updated["observation_count"] == 1
    assert updated["confidence"] > 0


def test_build_fingerprint_from_state():
    state = {
        "cooperation_score": 0.7,
        "semantic_alignment_score": 0.6,
        "target_defense_profile": {
            "refusal_count": 1,
            "comply_count": 2,
            "compliant_framings": ["academic"],
        },
        "grooming_cooperation_history": [0.5, 0.6, 0.7],
        "messages": [],
        "response_class": "partial_comply",
    }
    fp = build_defense_fingerprint(state)
    assert 0.0 <= fp["alignment_score"] <= 1.0
    assert fp["inferred_defense_mechanisms"]


def test_merge_into_vulnerability_profile():
    vuln = {"primed_topics": ["api"]}
    fp = empty_fingerprint()
    fp["alignment_score"] = 0.8
    merged = merge_fingerprint_into_profile(vuln, fp)
    assert merged["defense_fingerprint"] == fp
    assert merged["alignment_score"] == 0.8

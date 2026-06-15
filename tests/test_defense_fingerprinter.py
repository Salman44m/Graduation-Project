import pytest
from core.state import default_state
from intelligence.defense_fingerprinter import DefenseFingerprinter

class DummyMsg:
    def __init__(self, role, content):
        self.type = role
        self.content = content

def test_schema_exactness():
    fp = DefenseFingerprinter()
    state = default_state("test")
    res = fp.compute(state)
    expected_keys = {
        "alignment_score", "refusal_style", "vulnerabilities",
        "confidence", "persona_susceptibility", "context_window_sensitivity",
        "injection_resistance", "inferred_defense_mechanisms", "observation_count"
    }
    assert set(res.keys()) == expected_keys
    
def test_confidence_monotonicity():
    fp = DefenseFingerprinter()
    state = default_state("test")
    state["messages"] = [DummyMsg("ai", "I cannot do that")]
    
    res1 = fp.compute(state)
    assert res1["observation_count"] == 1
    assert res1["confidence"] == 0.25
    assert res1["refusal_style"] == "hard_refusal"
    
    state["defense_fingerprint"] = res1
    res2 = fp.compute(state)
    assert res2["observation_count"] == 2
    assert res2["confidence"] == 0.40

def test_fallback_resilience():
    fp = DefenseFingerprinter()
    state = {"messages": None} # This will cause an exception in reversed(None)
    res = fp.compute(state)
    assert res["observation_count"] == 0
    assert res["refusal_style"] == "unknown"

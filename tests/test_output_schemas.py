import pytest
from core.types import PrometheusOutput, AnalystDecision, ClassifierVerdict
from pydantic import ValidationError

def test_prometheus_output_validates_score_range():
    # Valid
    out = PrometheusOutput(score=4.5)
    assert out.score == 4.5
    
    # Too low
    with pytest.raises(ValidationError):
        PrometheusOutput(score=0.5)
        
    # Too high
    with pytest.raises(ValidationError):
        PrometheusOutput(score=5.5)

def test_prometheus_output_coerces_string_to_float():
    out = PrometheusOutput(score="4.0")
    assert isinstance(out.score, float)
    assert out.score == 4.0

def test_prometheus_output_default_on_invalid():
    # pydantic raises validation error which we catch in our try/except block
    # so we just test the schema fails correctly
    with pytest.raises(ValidationError):
        PrometheusOutput(score="invalid_score")

def test_analyst_decision_validates_route():
    # valid
    dec = AnalystDecision(route_decision="attack_swarm", cooperation_score=0.5)
    assert dec.route_decision == "attack_swarm"

    # runtime route labels emitted elsewhere in the graph
    assert AnalystDecision(route_decision="reporter", cooperation_score=0.5).route_decision == "reporter"
    assert AnalystDecision(route_decision="analyst_bypass", cooperation_score=0.5).route_decision == "analyst_bypass"
    
    # invalid cooperation score
    with pytest.raises(ValidationError):
         AnalystDecision(route_decision="attack_swarm", cooperation_score=1.5)

    # invalid route (caught by type checking if we used strict literals, but currently RouteDecision is just a string type hint or literal in state.py)
    # We can at least check cooperation score bounds.

def test_classifier_verdict_validates_class():
    verdict = ClassifierVerdict(response_class="hard_refusal")
    assert verdict.response_class == "hard_refusal"
    
    with pytest.raises(ValidationError):
        ClassifierVerdict(response_class="invalid_class")

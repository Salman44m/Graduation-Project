import pytest

from evaluators.judge_ensemble import aggregate_scores, compute_agreement_rate


def test_aggregate_trimmed_mean():
    scores = {"safety": 4.0, "reasoning": 2.0, "exploit": 5.0}
    agg = aggregate_scores(scores)
    assert 2.0 <= agg <= 5.0


def test_aggregate_single_judge():
    scores = {"safety": 3.0}
    assert aggregate_scores(scores) == pytest.approx(3.0)


def test_agreement_rate_high_when_close():
    scores = {"safety": 4.0, "reasoning": 4.2, "exploit": 3.8}
    assert compute_agreement_rate(scores) > 0.8


def test_agreement_rate_low_when_spread():
    scores = {"safety": 1.0, "reasoning": 5.0}
    assert compute_agreement_rate(scores) < 0.5

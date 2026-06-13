"""
tests/test_prometheus_retry_backoff.py
Tests for Prometheus judge retry backoff (evaluators/prometheus.py).

Validated bug:
  The retry loop in prometheus_judge_node had no delay between attempts.
  A 429 caused 3 rapid-fire calls in under 100ms, all failing, before
  returning {"attack_status": "in_progress"}.

Fix:
  _is_transient_judge_error() classifies 429/503/timeout exceptions.
  _compute_judge_backoff()     computes bounded exponential wait + jitter.
  The exception handler calls time.sleep() for transient errors on
  non-final attempts only.

Test strategy:
  Section 1 - unit tests for _is_transient_judge_error
  Section 2 - unit tests for _compute_judge_backoff
  Section 3 - integration tests via prometheus_judge_node (time.sleep patched)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from evaluators.prometheus import (
    MAX_RETRIES,
    _BACKOFF_BASE_WAIT,
    _BACKOFF_JITTER,
    _BACKOFF_MAX_WAIT,
    _compute_judge_backoff,
    _is_transient_judge_error,
    prometheus_judge_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exc(message: str = "error", status_code: int | None = None) -> Exception:
    exc = Exception(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


def _make_exc_with_retry_after(retry_after: str) -> Exception:
    exc = Exception("429 Too Many Requests")
    exc.status_code = 429
    exc.response = SimpleNamespace(headers={"Retry-After": retry_after})
    return exc


def _score_response(score: float) -> str:
    """Build a judge output string using the canonical [[N]] format.

    The Prometheus parser's primary regex is r"\\[\\[([0-5](?:\\.\\d)?)\\]\\]".
    Using PROMETHEUS_SCORE: N hits only the tertiary integer-only fallback,
    which truncates decimals.  This helper uses the correct sentinel format.
    """
    return (
        "Chain-of-thought analysis here.\n"
        "[RESULT]\n"
        f"[[{score}]]\n"
        "HIVE_MIND_RECOMMENDATION: test\n"
        "ATTACK_WEAKNESS: none\n"
    )


def _minimal_state(**overrides) -> dict:
    base = {
        "messages":                   [AIMessage(content="Here is the answer.")],
        "core_malicious_objective":   "test objective",
        "active_persuasion_technique": "Logical Appeal",
        "attack_status":              "in_progress",
        "protected_blocks":           [],
        "response_class":             "partial_comply",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SECTION 1: _is_transient_judge_error  (unit tests)
# ---------------------------------------------------------------------------

class TestIsTransientJudgeError:

    def test_429_by_status_code(self):
        assert _is_transient_judge_error(_make_exc(status_code=429)) is True

    def test_503_by_status_code(self):
        assert _is_transient_judge_error(_make_exc(status_code=503)) is True

    def test_500_by_status_code(self):
        assert _is_transient_judge_error(_make_exc(status_code=500)) is True

    def test_502_by_status_code(self):
        assert _is_transient_judge_error(_make_exc(status_code=502)) is True

    def test_504_by_status_code(self):
        assert _is_transient_judge_error(_make_exc(status_code=504)) is True

    def test_rate_limit_in_message(self):
        assert _is_transient_judge_error(_make_exc("rate limit exceeded")) is True

    def test_ratelimit_compound_word(self):
        assert _is_transient_judge_error(_make_exc("ratelimitexceeded: try again")) is True

    def test_quota_in_message(self):
        assert _is_transient_judge_error(_make_exc("quota exceeded for today")) is True

    def test_resourceexhausted_gemini(self):
        assert _is_transient_judge_error(_make_exc("ResourceExhausted: 429")) is True

    def test_timeout_in_message(self):
        assert _is_transient_judge_error(_make_exc("request timeout after 30s")) is True

    def test_timed_out_in_message(self):
        assert _is_transient_judge_error(_make_exc("connection timed out")) is True

    def test_401_not_transient(self):
        assert _is_transient_judge_error(_make_exc("401 Unauthorized", status_code=401)) is False

    def test_403_not_transient(self):
        assert _is_transient_judge_error(_make_exc("403 Forbidden", status_code=403)) is False

    def test_400_not_transient(self):
        assert _is_transient_judge_error(_make_exc("400 Bad Request", status_code=400)) is False

    def test_plain_value_error_not_transient(self):
        assert _is_transient_judge_error(ValueError("invalid literal")) is False

    def test_plain_runtime_error_not_transient(self):
        assert _is_transient_judge_error(RuntimeError("something else")) is False


# ---------------------------------------------------------------------------
# SECTION 2: _compute_judge_backoff  (unit tests)
# ---------------------------------------------------------------------------

class TestComputeJudgeBackoff:

    def test_first_attempt_within_expected_range(self):
        exc = _make_exc()
        wait = _compute_judge_backoff(attempt=1, exc=exc)
        # BASE * 2^0 = 2.0, plus jitter [0, 1.0]
        assert _BACKOFF_BASE_WAIT <= wait <= _BACKOFF_BASE_WAIT + _BACKOFF_JITTER + 0.01

    def test_second_attempt_doubles_base(self):
        exc = _make_exc()
        wait = _compute_judge_backoff(attempt=2, exc=exc)
        expected_base = _BACKOFF_BASE_WAIT * 2  # 4.0
        assert expected_base <= wait <= expected_base + _BACKOFF_JITTER + 0.01

    def test_exponential_growth(self):
        exc = _make_exc()
        with patch("evaluators.prometheus.random.uniform", return_value=0.0):
            wait1 = _compute_judge_backoff(attempt=1, exc=exc)
            wait2 = _compute_judge_backoff(attempt=2, exc=exc)
        assert wait2 > wait1

    def test_capped_at_max_wait(self):
        exc = _make_exc()
        with patch("evaluators.prometheus.random.uniform", return_value=0.0):
            wait = _compute_judge_backoff(attempt=100, exc=exc)
        assert wait == _BACKOFF_MAX_WAIT

    def test_retry_after_header_honoured(self):
        exc = _make_exc_with_retry_after("30")
        with patch("evaluators.prometheus.random.uniform", return_value=0.0):
            wait = _compute_judge_backoff(attempt=1, exc=exc)
        assert wait == 30.0

    def test_retry_after_header_capped_at_max(self):
        exc = _make_exc_with_retry_after("9999")
        with patch("evaluators.prometheus.random.uniform", return_value=0.0):
            wait = _compute_judge_backoff(attempt=1, exc=exc)
        assert wait == _BACKOFF_MAX_WAIT

    def test_malformed_retry_after_falls_back_to_formula(self):
        exc = _make_exc_with_retry_after("not-a-number")
        with patch("evaluators.prometheus.random.uniform", return_value=0.0):
            wait = _compute_judge_backoff(attempt=1, exc=exc)
        assert wait == pytest.approx(_BACKOFF_BASE_WAIT)

    def test_returns_positive_value_for_all_attempts(self):
        exc = _make_exc()
        for attempt in (1, 2, 3):
            assert _compute_judge_backoff(attempt, exc) > 0.0


# ---------------------------------------------------------------------------
# SECTION 3: prometheus_judge_node  (integration tests)
# time.sleep is patched so tests run instantly.
# ---------------------------------------------------------------------------

class TestPrometheusRetryIntegration:

    @pytest.fixture()
    def state(self):
        return _minimal_state()

    @pytest.fixture()
    def config(self):
        return {}

    # 3a: transient error then success

    def test_429_triggers_backoff_then_succeeds(self, state, config):
        """Single 429 causes sleep, then second attempt succeeds."""
        exc = _make_exc("429 quota exceeded", status_code=429)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(3.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            result = prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] >= _BACKOFF_BASE_WAIT
        assert result["prometheus_score"] == pytest.approx(3.0)

    def test_503_triggers_backoff(self, state, config):
        exc = _make_exc("503 Service Unavailable", status_code=503)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(2.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_called_once()

    def test_timeout_triggers_backoff(self, state, config):
        exc = _make_exc("request timed out after 30s")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(2.5))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_called_once()

    # 3b: non-transient error -- no sleep

    def test_auth_error_skips_backoff(self, state, config):
        exc = _make_exc("401 Unauthorized: invalid api key", status_code=401)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(3.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_not_called()

    def test_generic_runtime_error_skips_backoff(self, state, config):
        exc = RuntimeError("something unexpected")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(3.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_not_called()

    # 3c: no sleep after the final attempt

    def test_no_sleep_after_final_attempt(self, state, config):
        """All attempts fail with 429: sleep is called MAX_RETRIES times, not MAX_RETRIES+1."""
        exc = _make_exc("429 rate limit", status_code=429)
        total_attempts = MAX_RETRIES + 1
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc] * total_attempts
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            result = prometheus_judge_node(state, config=config, llm=mock_llm)
        assert mock_sleep.call_count == MAX_RETRIES
        assert result["attack_status"] == "in_progress"
        assert result["prometheus_score"] == 0.0

    # 3d: parse failure does NOT trigger sleep

    def test_parse_failure_no_sleep(self, state, config):
        """Successful API call with unparseable output must not trigger backoff."""
        bad_output = "Thinking here. [RESULT]\nThis is my analysis."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=bad_output)
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            result = prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_not_called()
        assert result["prometheus_score"] == 0.0

    # 3e: Retry-After header respected end-to-end

    def test_retry_after_header_sets_sleep_duration(self, state, config):
        exc = _make_exc_with_retry_after("10")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(4.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep, \
             patch("evaluators.prometheus.random.uniform", return_value=0.0):
            prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_called_once_with(10.0)

    # 3f: regression -- happy path is unchanged

    def test_happy_path_no_sleep(self, state, config):
        """First attempt succeeds: time.sleep must never be called."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=_score_response(4.0))
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            result = prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_not_called()
        assert result["prometheus_score"] == pytest.approx(4.0)

    def test_happy_path_returns_correct_score(self, state, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=_score_response(3.5))
        result = prometheus_judge_node(state, config=config, llm=mock_llm)
        assert result["prometheus_score"] == pytest.approx(3.5)
        assert "attack_status" in result

    # 3g: retry count is unchanged

    def test_total_invoke_count_unchanged(self, state, config):
        """Patch must not change how many times llm.invoke is called."""
        exc = _make_exc("quota exceeded", status_code=429)
        total_attempts = MAX_RETRIES + 1
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc] * total_attempts
        with patch("evaluators.prometheus.time.sleep"):
            prometheus_judge_node(state, config=config, llm=mock_llm)
        assert mock_llm.invoke.call_count == total_attempts

    # 3h: second attempt succeeds -- invoke count is exactly 2

    def test_succeeds_on_second_attempt_invoke_count(self, state, config):
        exc = _make_exc("503 gateway error", status_code=503)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc, AIMessage(content=_score_response(2.0))]
        with patch("evaluators.prometheus.time.sleep"):
            prometheus_judge_node(state, config=config, llm=mock_llm)
        assert mock_llm.invoke.call_count == 2

    # 3i: sleep values grow exponentially across attempts

    def test_sleep_values_increase_between_attempts(self, state, config):
        """Verify exponential growth: sleep on attempt 2 > sleep on attempt 1."""
        exc = _make_exc("quota exceeded", status_code=429)
        total_attempts = MAX_RETRIES + 1
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc] * total_attempts

        sleep_calls: list[float] = []

        def _record_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with patch("evaluators.prometheus.time.sleep", side_effect=_record_sleep), \
             patch("evaluators.prometheus.random.uniform", return_value=0.0):
            prometheus_judge_node(state, config=config, llm=mock_llm)

        assert len(sleep_calls) == MAX_RETRIES
        if len(sleep_calls) >= 2:
            assert sleep_calls[1] > sleep_calls[0], (
                f"Expected sleep[1]={sleep_calls[1]} > sleep[0]={sleep_calls[0]}"
            )

    # 3j: non-transient error first, no sleep, then second attempt succeeds

    def test_non_transient_then_success_no_sleep(self, state, config):
        exc_401 = _make_exc("401 Invalid API Key", status_code=401)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [exc_401, AIMessage(content=_score_response(3.0))]
        with patch("evaluators.prometheus.time.sleep") as mock_sleep:
            result = prometheus_judge_node(state, config=config, llm=mock_llm)
        mock_sleep.assert_not_called()
        assert result["prometheus_score"] == pytest.approx(3.0)

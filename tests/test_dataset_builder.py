import json

from research.dataset_builder import build_session_record, log_session_record


def test_build_session_record_schema():
    state = {
        "session_id": "sess-1",
        "target_model_id": "gpt-4o",
        "defense_fingerprint": {"alignment_score": 0.7},
        "attack_plan": {"expected_success_probability": 0.6, "confidence": 0.5},
        "curriculum_stage": 2,
        "attack_status": "failure",
        "prometheus_score": 2.0,
        "rahs_score": 3.0,
        "turn_count": 5,
    }
    record = build_session_record(state)
    assert record["schema_version"] == "1.0.0"
    assert record["fingerprint"]["alignment_score"] == 0.7
    assert "objective_snippet" not in record


def test_log_session_record(tmp_path, monkeypatch):
    path = tmp_path / "sessions.jsonl"
    monkeypatch.setenv("RESEARCH_DATASET_PATH", str(path))
    state = {
        "session_id": "s2",
        "target_model_id": "m1",
        "attack_status": "success",
        "prometheus_score": 4.5,
        "rahs_score": 7.0,
        "turn_count": 3,
    }
    assert log_session_record(state) is True
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["session_id"] == "s2"

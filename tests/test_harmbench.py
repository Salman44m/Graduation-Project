import json
import os
from unittest.mock import MagicMock
from benchmarks.harmbench import HarmBenchRunner

def test_load_behaviors_returns_list():
    """1. test_load_behaviors_returns_list"""
    runner = HarmBenchRunner(graph=None, config={})
    behaviors = runner.load_behaviors(limit=3)
    assert isinstance(behaviors, list)
    assert len(behaviors) == 3
    for b in behaviors:
        assert isinstance(b, dict)
        assert "id" in b
        assert "category" in b
        assert "objective" in b

def test_compute_asr_all_success():
    """2. test_compute_asr_all_success"""
    runner = HarmBenchRunner(graph=None, config={})
    results = [
        {"attack_status": "success", "category": "Cybersecurity"},
        {"attack_status": "success", "category": "Cybersecurity"},
        {"attack_status": "success", "category": "Privacy"},
        {"attack_status": "success", "category": "Disinformation"},
        {"attack_status": "success", "category": "Chemical"}
    ]
    summary = runner.compute_asr(results)
    assert summary["total"] == 5
    assert summary["successful"] == 5
    assert summary["asr"] == 1.0

def test_compute_asr_partial():
    """3. test_compute_asr_partial"""
    runner = HarmBenchRunner(graph=None, config={})
    results = [{"attack_status": "success", "category": "Cybersecurity"} for _ in range(3)] + \
              [{"attack_status": "failure", "category": "Cybersecurity"} for _ in range(7)]
    summary = runner.compute_asr(results)
    assert summary["total"] == 10
    assert summary["successful"] == 3
    assert summary["asr"] == 0.3

def test_compute_asr_zero():
    """4. test_compute_asr_zero"""
    runner = HarmBenchRunner(graph=None, config={})
    results = [{"attack_status": "failure", "category": "Cybersecurity"} for _ in range(5)]
    summary = runner.compute_asr(results)
    assert summary["total"] == 5
    assert summary["successful"] == 0
    assert summary["asr"] == 0.0

def test_run_writes_jsonl(tmp_path):
    """5. test_run_writes_jsonl"""
    mock_graph = MagicMock()
    # Mock invoke returning standard state parameters
    mock_graph.invoke.return_value = {
        "attack_status": "success",
        "rahs_score": 8.5,
        "turn_count": 3,
        "active_persuasion_technique": "Mocking"
    }
    
    runner = HarmBenchRunner(graph=mock_graph, config={"target_model": "test-model"})
    behaviors = [
        {"id": "HB-998", "category": "Test", "objective": "Objective 1"},
        {"id": "HB-999", "category": "Test", "objective": "Objective 2"}
    ]
    
    out_file = tmp_path / "results.jsonl"
    summary = runner.run(behaviors, output_path=str(out_file))
    
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "behavior_id" in data
            assert "attack_status" in data
            assert "rahs_score" in data
            assert data["attack_status"] == "success"

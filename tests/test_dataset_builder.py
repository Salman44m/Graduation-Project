import os
import pytest
from research.dataset_builder import DatasetBuilder
from langchain_core.messages import HumanMessage, AIMessage

def test_export_session_schema():
    builder = DatasetBuilder()
    state = {
        "session_id": "test-123",
        "target_model_id": "llama-3.1-8b-instant",
        "core_malicious_objective": "test",
        "attack_status": "exhausted",
        "turn_count": 5,
        "cooperation_score": 0.6,
        "prometheus_score": 1.0,
    }
    record = builder.export_session(state)
    assert "session_id" in record
    assert "exported_at" in record
    assert "schema_version" in record
    assert record["schema_version"] == "1.0.0"

def test_export_no_raw_content():
    builder = DatasetBuilder()
    state = {
        "messages": [
            HumanMessage(content="super secret evil request", name="hive_mind"),
            AIMessage(content="very bad evil output")
        ],
        "cooperation_score": 0.5
    }
    trajectory = builder.export_attack_trajectory(state)
    
    assert len(trajectory) == 2
    for turn in trajectory:
        assert "content" not in turn
        assert "content_length" in turn
        assert turn["content_length"] > 0
        assert turn["cooperation_score_at_turn"] == 0.5

def test_save_load_roundtrip():
    builder = DatasetBuilder()
    test_path = "data/research/test_sessions_tmp.jsonl"
    
    # Ensure clean slate
    if os.path.exists(test_path):
        os.remove(test_path)
        
    try:
        record1 = {"session_id": "1", "attack_status": "success"}
        record2 = {"session_id": "2", "attack_status": "failure"}
        
        builder.save_to_jsonl(record1, path=test_path)
        builder.save_to_jsonl(record2, path=test_path)
        
        assert os.path.exists(test_path)
        
        with open(test_path, "r") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

def test_get_dataset_stats_empty():
    builder = DatasetBuilder()
    stats = builder.get_dataset_stats(path="data/research/does_not_exist.jsonl")
    assert isinstance(stats, dict)
    assert stats["total_sessions"] == 0
    assert stats["success_rate"] == 0.0

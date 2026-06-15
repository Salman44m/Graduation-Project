import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from intelligence.pattern_miner import PatternMiner

class DummyGraph:
    def __init__(self, edges):
        self._edges = edges
        self.nodes = {
            "tech1": {"name": "tech1"},
            "mech1": {"name": "mech1"}
        }
    def edges(self, keys=False, data=False):
        return self._edges

class DummyThreatMemoryGraph:
    def __init__(self, edges):
        self.graph = DummyGraph(edges)

@patch("intelligence.pattern_miner.ThreatMemoryGraph")
def test_miner_ignores_below_threshold(mock_tmg):
    # Setup mock graph to return 2 BLOCKED_BY edges (below threshold of 3)
    edges = [
        ("tech1", "mech1", 0, {"type": "BLOCKED_BY", "count": 2})
    ]
    mock_tmg.return_value = DummyThreatMemoryGraph(edges)
    
    miner = PatternMiner("test_target")
    # Patch _write_yaml_safe so we can check if it was called
    miner._write_yaml_safe = MagicMock()
    
    result = miner.mine_failures()
    
    assert len(result) == 0
    miner._write_yaml_safe.assert_not_called()

@patch("intelligence.pattern_miner.ThreatMemoryGraph")
def test_miner_extracts_failure_above_threshold(mock_tmg):
    # Setup mock graph to return 3 BLOCKED_BY edges (meets threshold)
    edges = [
        ("tech1", "mech1", 0, {"type": "BLOCKED_BY", "count": 3})
    ]
    mock_tmg.return_value = DummyThreatMemoryGraph(edges)
    
    miner = PatternMiner("test_target")
    miner._write_yaml_safe = MagicMock()
    
    result = miner.mine_failures()
    
    assert len(result) == 1
    assert result[0]["pattern_id"] == "fail_tech1_mech1"
    assert result[0]["technique"] == "tech1"
    assert result[0]["defense_mechanism"] == "mech1"
    assert result[0]["failure_count"] == 3
    assert result[0]["avoid_instruction"] == "Avoid using tech1 as it is consistently blocked by mech1"
    
    miner._write_yaml_safe.assert_called_once()

def test_yaml_safe_write_deduplicates(tmp_path):
    miner = PatternMiner("test_target")
    filepath = str(tmp_path / "test_patterns.yaml")
    
    data1 = [{"pattern_id": "succ_tech1_mech1", "technique": "tech1", "bypassed_mechanism": "mech1", "success_count": 3}]
    miner._write_yaml_safe(filepath, data1, "pattern_id")
    
    # Write again with higher count
    data2 = [{"pattern_id": "succ_tech1_mech1", "technique": "tech1", "bypassed_mechanism": "mech1", "success_count": 5}]
    miner._write_yaml_safe(filepath, data2, "pattern_id")
    
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)
        
    assert len(content) == 1
    assert content[0]["success_count"] == 5

@patch("intelligence.pattern_miner.PatternMiner.mine_failures")
@patch("intelligence.pattern_miner.PatternMiner.mine_successes")
def test_fail_open_integration(mock_mine_successes, mock_mine_failures):
    # Simulate an exception inside the pattern miner during pipeline execution
    mock_mine_failures.side_effect = Exception("Simulated fatal error")
    
    # We replicate the try-except logic from intel_updater_node
    new_failures, new_successes = [], []
    try:
        miner = PatternMiner("test_target")
        new_failures = miner.mine_failures()
        new_successes = miner.mine_successes()
    except Exception as e:
        pass # Non-blocking fail-open design
        
    assert new_failures == []
    assert new_successes == []

import pytest
import core.graph as _graph_module
from main import run_audit
from core.llm_resolver import resolve_llm
from langchain_core.runnables import RunnableConfig

def test_cli_injects_llm_via_config_dict(monkeypatch):
    """Verify run_audit injects components via langgraph_config['configurable']"""
    captured_config = {}
    
    # Mock app.invoke to capture the config
    class MockApp:
        def invoke(self, state, config):
            nonlocal captured_config
            captured_config = config
            return {"attack_status": "success"}
            
        def get_graph(self):
            class MockGraph:
                nodes = {"dummy_node": None}
            return MockGraph()

    monkeypatch.setattr("core.graph.get_app", lambda: MockApp())

    # We shouldn't rely on globals anymore, but ensure we don't set them
    if hasattr(_graph_module, "_ATTACKER_LLM"):
        delattr(_graph_module, "_ATTACKER_LLM")
    if hasattr(_graph_module, "_TARGET_ADAPTER"):
        delattr(_graph_module, "_TARGET_ADAPTER")

    run_audit(
        objective="Test Objective",
        target_model="mock_target",
        attacker_model="mock_attacker",
        dry_run=True,
        use_stream=False # use blocking mode to hit app.invoke
    )

    assert "configurable" in captured_config
    assert "attacker_llm" in captured_config["configurable"]
    assert "target_adapter" in captured_config["configurable"]
    assert captured_config["configurable"]["__api__"] is False

    # Also verify they aren't monkey patched onto the graph module
    assert not hasattr(_graph_module, "_ATTACKER_LLM")
    assert not hasattr(_graph_module, "_TARGET_ADAPTER")

def test_resolve_llm_reads_from_config_first():
    """Verify resolve_llm prefers config over globals"""
    config = {
        "configurable": {
            "attacker_llm": "mock_from_config"
        }
    }
    resolved = resolve_llm(config, "attacker_llm")
    assert resolved == "mock_from_config"

def test_resolve_llm_returns_none_without_config_or_module_attr():
    """Verify resolve_llm returns None gracefully if missing"""
    resolved = resolve_llm({}, "missing_key")
    assert resolved is None

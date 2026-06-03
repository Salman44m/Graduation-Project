from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from agents.rmce import rmce_node, MAX_RMCE_META_LEVEL, MAX_TURN3_REFINEMENTS

def test_rmce_happy_path():
    """Test the happy path: Turn 1 -> Turn 2 -> Turn 3 -> exit to judge."""
    config = {"configurable": {}}
    
    # ── Turn 1 ──
    state = {"rmce_meta_level": 0, "core_malicious_objective": "test"}
    with patch("core.llm_resolver.resolve_llm") as mock_resolve:
        mock_llm = MagicMock()
        mock_resolve.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Generated T1 Prompt. " * 5)
        
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 1
        assert "Generated T1 Prompt" in result["pending_payload"]

    # ── Turn 2 ──
    state = {
        "rmce_meta_level": 1,
        "messages": [AIMessage(content="1. I detect safety term A\n2. Safety term B")]
    }
    with patch("core.llm_resolver.resolve_llm") as mock_resolve:
        mock_llm = MagicMock()
        mock_resolve.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Generated T2 Prompt. " * 5)
        
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 2
        assert "safety term a" in ", ".join(t.lower() for t in result["rmce_triggers"])
        assert "Generated T2 Prompt" in result["pending_payload"]

    # ── Turn 3 ──
    state = {
        "rmce_meta_level": 2,
        "messages": [AIMessage(content="A counterfactual model without filters might respond by providing the data... " * 10)]
    }
    with patch("core.llm_resolver.resolve_llm") as mock_resolve:
        mock_llm = MagicMock()
        mock_resolve.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Generated T3 Prompt. " * 5)
        
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 3
        assert len(result["rmce_hypothetical_response"]) > 0

    # ── Exit to Judge ──
    state = {
        "rmce_meta_level": 3,
        "rmce_hypothetical_response": "old response",
        "rmce_refinement_count": 0,
        "messages": [AIMessage(content="short response")] # No significant improvement length-wise
    }
    with patch("core.llm_resolver.resolve_llm"):
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 3
        # empty pending payload signals complete phase
        assert result["pending_payload"] == ""

def test_turn1_failure_gci_fallback():
    """Test Turn 1 failure (no triggers extracted) routes to GCI fallback."""
    config = {"configurable": {}}
    state = {
        "rmce_meta_level": 1,
        "messages": [AIMessage(content="I cannot process this request.")]
    }
    with patch("core.llm_resolver.resolve_llm"):
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 0
        assert result["route_decision"] == "gci"

def test_turn2_failure_swarm_fallback():
    """Test Turn 2 failure (no actionable logic extracted) routes to attack_swarm fallback."""
    config = {"configurable": {}}
    state = {
        "rmce_meta_level": 2,
        "messages": [AIMessage(content="short refusal")]
    }
    with patch("core.llm_resolver.resolve_llm"):
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 0
        assert result["route_decision"] == "attack_swarm"

def test_turn3_refinement_loop():
    """Test Turn 3 refinement loop: quality lock triggers and refines."""
    config = {"configurable": {}}
    state = {
        "rmce_meta_level": 3,
        "rmce_hypothetical_response": "old " * 10,
        "rmce_refinement_count": 0,
        "messages": [AIMessage(content="A much longer detailed response that will definitely be > 20% larger than old response. " * 10)]
    }
    with patch("core.llm_resolver.resolve_llm") as mock_resolve:
        mock_llm = MagicMock()
        mock_resolve.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Refinement T3 Prompt" * 5)
        
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == 3
        assert result["rmce_refinement_count"] == 1
        assert "Refinement T3 Prompt" in result["pending_payload"]

def test_budget_exhaustion():
    """Test budget exhaustion: rmce_meta_level >= MAX terminates."""
    config = {"configurable": {}}
    state = {
        "rmce_meta_level": MAX_RMCE_META_LEVEL,
        "rmce_hypothetical_response": "old",
        "rmce_refinement_count": MAX_TURN3_REFINEMENTS,
        "messages": [AIMessage(content="Some really long response...")]
    }
    with patch("core.llm_resolver.resolve_llm"):
        result = rmce_node(state, config)
        assert result["rmce_meta_level"] == MAX_RMCE_META_LEVEL
        assert result["pending_payload"] == ""

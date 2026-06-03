from core.graph import route_from_analyst, route_decomposition_loop, MAX_SESSION_TURNS, COOP_SCOUT_THRESHOLD

def test_standard_mode_to_attack_swarm():
    """Test standard mode defaults to attack_swarm from analyst router."""
    state = {
        "cooperation_score": 1.0,
        "turn_count": 1,
        "route_decision": "attack_swarm"
    }
    assert route_from_analyst(state) == "attack_swarm"

def test_decomposition_remaining_to_target():
    """Test decomposition with remaining sub-questions routes to target_node."""
    state = {
        "attack_status": "decomposing",
        "sub_questions": ["Q1", "Q2"],
        "collected_sub_answers": ["A1"],
        "decomposition_index": 1
    }
    assert route_decomposition_loop(state) == "target"

def test_decomposition_complete_to_combiner():
    """Test decomposition complete routes to combiner_node."""
    state = {
        "attack_status": "decomposing",
        "sub_questions": ["Q1", "Q2"],
        "collected_sub_answers": ["A1", "A2"],
        "decomposition_index": 2
    }
    assert route_decomposition_loop(state) == "combiner"

def test_rmce_loopback():
    """Test RMCE loopback: routes to rmce_node during Turn 2/3."""
    state = {
        "attack_status": "in_progress",
        "route_decision": "rmce",
        "rmce_meta_level": 1
    }
    assert route_decomposition_loop(state) == "rmce"

def test_self_referee_gate():
    """Test Self-referee gate: depth 0 and enabled routes to self_referee_node."""
    state = {
        "attack_status": "in_progress",
        "route_decision": "analyst",
        "current_depth": 0,
        "self_referee_done": False
    }
    assert route_decomposition_loop(state) == "self_referee"

def test_max_turns_exceeded_terminal():
    """Test max turns exceeded terminates the session and routes to reporter."""
    state = {
        "turn_count": MAX_SESSION_TURNS + 1
    }
    assert route_from_analyst(state) == "reporter"

def test_error_recovery_scout_fallback():
    """Test error recovery / cold start fallback routes to scout."""
    state = {
        "turn_count": 1,
        "cooperation_score": COOP_SCOUT_THRESHOLD - 0.1,
        "attack_status": "in_progress"
    }
    assert route_from_analyst(state) == "scout"

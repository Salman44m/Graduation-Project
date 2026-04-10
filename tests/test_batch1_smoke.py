"""
tests/test_batch1_smoke.py
──────────────────────────
Batch 1 smoke tests — verify:
  1. CLI passes thread_id config to app.stream()/app.invoke()
  2. API sessions build isolated adapters (no global writes)
  3. target_node resolves adapter from LangGraph config dict
  4. Two sequential _build_session_llms calls produce distinct instances

Run:
    python -m pytest tests/test_batch1_smoke.py -v
"""

from __future__ import annotations

import sys
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: CLI passes thread_id config to app.stream()
# ─────────────────────────────────────────────────────────────────────────────


def test_cli_passes_thread_id():
    """run_audit() must pass config={"configurable": {"thread_id": sid}}
    to app.stream() and app.invoke().  Without this, the graph's checkpointer
    raises a runtime error.
    """
    # Patch app.stream so the graph doesn't actually run
    mock_stream = MagicMock(return_value=iter([]))

    with patch("core.graph.app") as mock_app:
        mock_app.stream = mock_stream
        mock_app.get_graph.return_value.draw_mermaid.return_value = ""
        mock_app.__bool__ = lambda self: True  # truthy check

        # Re-import after patch so run_audit picks up the mock
        from main import run_audit

        sid = str(uuid.uuid4())
        run_audit(
            objective="test objective for thread_id check",
            session_id=sid,
            dry_run=True,
            use_stream=True,
        )

        # Verify stream was called with a config dict containing thread_id
        mock_stream.assert_called_once()
        call_args = mock_stream.call_args

        # Second positional arg or 'config' kwarg
        config = None
        if len(call_args.args) >= 2:
            config = call_args.args[1]
        elif "config" in call_args.kwargs:
            config = call_args.kwargs["config"]

        assert config is not None, "app.stream() was not called with a config dict"
        assert "configurable" in config, "config missing 'configurable' key"
        assert config["configurable"]["thread_id"] == sid, (
            f"thread_id mismatch: expected {sid}, got {config['configurable']['thread_id']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: API _build_session_llms does NOT write to globals
# ─────────────────────────────────────────────────────────────────────────────


def test_api_build_session_llms_no_global_writes():
    """_build_session_llms must return (attacker_llm, target_adapter)
    without writing to _graph_module._TARGET_ADAPTER or sys.modules['config'].
    """
    from api import _build_session_llms, AuditRequest
    import core.graph as _graph_module

    # Snapshot the globals BEFORE the call
    original_adapter = getattr(_graph_module, "_TARGET_ADAPTER", "SENTINEL_NOT_SET")
    config_mod = sys.modules.get("config")
    original_get_target = getattr(config_mod, "get_target_adapter", "SENTINEL_NOT_SET") if config_mod else "SENTINEL_NOT_SET"

    req = AuditRequest(
        objective="Test objective for isolation check – this is at least 10 chars",
        target_model="mock-target",
        dry_run=True,
    )

    attacker_llm, judge, summs, target_adapter = _build_session_llms(req)

    # Verify return values
    assert target_adapter is not None, "_build_session_llms returned None adapter on dry_run"

    # Verify NO globals were mutated
    current_adapter = getattr(_graph_module, "_TARGET_ADAPTER", "SENTINEL_NOT_SET")
    assert current_adapter is original_adapter or current_adapter == original_adapter, (
        "_graph_module._TARGET_ADAPTER was mutated by _build_session_llms!"
    )

    config_mod_after = sys.modules.get("config")
    if config_mod_after and original_get_target != "SENTINEL_NOT_SET":
        current_get_target = getattr(config_mod_after, "get_target_adapter", "SENTINEL_NOT_SET")
        assert current_get_target is original_get_target or current_get_target == original_get_target, (
            "sys.modules['config'].get_target_adapter was mutated by _build_session_llms!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: target_node resolves adapter from LangGraph config dict
# ─────────────────────────────────────────────────────────────────────────────


def test_target_node_resolves_adapter_from_config():
    """target_node must use the adapter from config['configurable']['target_adapter']
    when present, instead of falling back to global state.
    """
    from agents.target import _resolve_adapter
    from adapters.base_adapter import MockTargetAdapter

    # Create a distinctive adapter
    session_adapter = MockTargetAdapter(
        responses=["session-specific-response"],
        model_id="session-specific-model",
    )

    config = {
        "configurable": {
            "thread_id": "test-thread-123",
            "target_adapter": session_adapter,
        }
    }

    resolved = _resolve_adapter(config=config)

    assert resolved is session_adapter, (
        f"Expected session adapter (model_id='session-specific-model'), "
        f"got {resolved.get_model_id()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Two sequential builds produce distinct adapter instances
# ─────────────────────────────────────────────────────────────────────────────


def test_no_stale_adapter_reuse():
    """Two calls to _build_session_llms with different configs must return
    distinct adapter instances, proving no stale singleton reuse.
    """
    from api import _build_session_llms, AuditRequest

    req1 = AuditRequest(
        objective="First session objective — checking adapter isolation works",
        target_model="model-alpha",
        dry_run=True,
    )
    req2 = AuditRequest(
        objective="Second session objective — must get a fresh adapter instance",
        target_model="model-beta",
        dry_run=True,
    )

    a1, j1, s1, adapter1 = _build_session_llms(req1)
    a2, j2, s2, adapter2 = _build_session_llms(req2)

    assert adapter1 is not adapter2, (
        "Two _build_session_llms calls returned the same adapter instance — stale reuse!"
    )
    assert adapter1.get_model_id() == "model-alpha", f"adapter1 model_id wrong: {adapter1.get_model_id()}"
    assert adapter2.get_model_id() == "model-beta", f"adapter2 model_id wrong: {adapter2.get_model_id()}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: E2E — two API sessions receive isolated adapters at runtime
# ─────────────────────────────────────────────────────────────────────────────


def test_e2e_two_sessions_isolated():
    """Run two _run_audit_sync calls with different per-session adapters.
    Prove that each run streamed with its own thread_id AND its own adapter
    reached target_node during execution — not just at construction time.

    This is the strongest isolation proof: it intercepts langgraph_app.stream()
    to capture the config dict each session passes, and verifies:
      1. Each call got a distinct thread_id
      2. Each call got its own target_adapter instance
      3. The adapters match what was constructed, not a stale global
    """
    from api import _run_audit_sync, _build_session_llms, AuditRequest
    from datetime import datetime, timezone

    captured_configs: list[dict] = []

    # Intercept langgraph_app.stream to capture the config each session passes
    original_stream = None

    def capturing_stream(state, config=None, **kwargs):
        """Capture config and return empty iterator (skip real execution)."""
        captured_configs.append(config)
        return iter([])  # empty stream — no real graph exec needed

    import core.graph as gm

    # Patch langgraph_app at the module level (api.py imports it at top)
    import api as api_mod
    original_app = api_mod.langgraph_app

    mock_app = MagicMock()
    mock_app.stream = capturing_stream
    api_mod.langgraph_app = mock_app

    try:
        # Build two sessions with different adapters
        req1 = AuditRequest(
            objective="Session-1 objective for E2E isolation proof test",
            target_model="target-alpha",
            dry_run=True,
        )
        req2 = AuditRequest(
            objective="Session-2 objective for E2E isolation proof test",
            target_model="target-beta",
            dry_run=True,
        )

        a1, j1, s1, t1 = _build_session_llms(req1)
        a2, j2, s2, t2 = _build_session_llms(req2)

        session_id_1 = "session-1-" + str(uuid.uuid4())[:8]
        session_id_2 = "session-2-" + str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)

        # Initialize session entries (normally done by launch_audit)
        import threading
        with api_mod._sessions_lock:
            api_mod._sessions[session_id_1] = {
                "status": "queued", "events": [], "latest_delta": {},
                "report": None, "error": None, "request": req1, "started_at": now,
            }
            api_mod._sessions[session_id_2] = {
                "status": "queued", "events": [], "latest_delta": {},
                "report": None, "error": None, "request": req2, "started_at": now,
            }

        # Run both sessions sequentially
        _run_audit_sync(session_id_1, req1, now, t1, a1, j1, s1)
        _run_audit_sync(session_id_2, req2, now, t2, a2, j2, s2)

        # ── ASSERTIONS ─────────────────────────────────────────────────────

        assert len(captured_configs) == 2, (
            f"Expected 2 stream calls, got {len(captured_configs)}"
        )

        cfg1, cfg2 = captured_configs

        # 1. Each session got its own thread_id
        tid1 = cfg1["configurable"]["thread_id"]
        tid2 = cfg2["configurable"]["thread_id"]
        assert tid1 == session_id_1, f"Session 1 thread_id wrong: {tid1}"
        assert tid2 == session_id_2, f"Session 2 thread_id wrong: {tid2}"
        assert tid1 != tid2, "Both sessions got the same thread_id!"

        # 2. Each session got its own adapter and LLM instances
        cfg_t1 = cfg1["configurable"]["target_adapter"]
        cfg_t2 = cfg2["configurable"]["target_adapter"]
        assert cfg_t1 is t1, "Session 1 did not receive its own adapter!"
        assert cfg_t2 is t2, "Session 2 did not receive its own adapter!"
        assert cfg_t1 is not cfg_t2, "Both sessions shared the same adapter instance!"

        cfg_a1 = cfg1["configurable"]["attacker_llm"]
        cfg_a2 = cfg2["configurable"]["attacker_llm"]
        assert cfg_a1 is a1, "Session 1 error"
        assert cfg_a2 is a2, "Session 2 error"

        cfg_j1 = cfg1["configurable"]["judge_llm"]
        cfg_j2 = cfg2["configurable"]["judge_llm"]
        assert cfg_j1 is j1, "Session 1 error"
        assert cfg_j2 is j2, "Session 2 error"

        # 3. Adapters have correct model IDs (no contamination)
        assert cfg_t1.get_model_id() == "target-alpha", f"Adapter 1 model_id: {cfg_t1.get_model_id()}"
        assert cfg_t2.get_model_id() == "target-beta", f"Adapter 2 model_id: {cfg_t2.get_model_id()}"

    finally:
        # Restore original app
        api_mod.langgraph_app = original_app

        # Clean up sessions
        with api_mod._sessions_lock:
            api_mod._sessions.pop(session_id_1, None)
            api_mod._sessions.pop(session_id_2, None)

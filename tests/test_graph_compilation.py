"""
tests/test_graph_compilation.py
───────────────────────────────
Startup and graph compilation regression tests (Phase 0).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

import core.graph as graph_module
from core.graph import build_graph, get_app


@pytest.fixture(autouse=True)
def _reset_graph_singleton():
    """Isolate graph singleton between tests."""
    graph_module._app = None
    graph_module._app_built = False
    yield
    graph_module._app = None
    graph_module._app_built = False


@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


class TestGraphCompilation:

    def test_build_graph_returns_compiled_state_graph(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            app = build_graph()
        assert app is not None
        assert isinstance(app, CompiledStateGraph)

    def test_build_graph_node_count(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            app = build_graph()
        user_nodes = {n for n in app.get_graph().nodes.keys() if not n.startswith("__")}
        assert len(user_nodes) == 19

    def test_get_app_singleton(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            first = get_app()
            second = get_app()
        assert first is not None
        assert second is first

    def test_get_app_compiles_once(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            get_app()
            assert graph_module._app_built is True
            built_app = graph_module._app
            get_app()
            assert graph_module._app is built_app

    def test_graph_entry_point(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            app = build_graph()
        edges_from_start = [
            e for e in app.get_graph().edges
            if e.source == "__start__"
        ]
        assert len(edges_from_start) == 1
        assert edges_from_start[0].target == "intel_retriever"

    def test_graph_has_checkpointer(self, memory_checkpointer):
        with patch("infra.persistence.build_checkpointer", return_value=memory_checkpointer):
            app = build_graph()
        config = {"configurable": {"thread_id": "test-compilation-thread"}}
        # get_state on empty thread should not raise when checkpointer is wired
        snapshot = app.get_state(config)
        assert snapshot is not None

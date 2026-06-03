"""
tests/test_reporter_sanitization.py
─────────────────────────────────────────────────────────────────────────────
Tests for the _sanitize_session_id helper in core/graph.py.

The function is a pure function with no side effects, so tests are fast
and deterministic with no file I/O required.

Coverage:
  - Valid UUID4-style session IDs pass through unchanged
  - Directory traversal payloads are stripped to basename
  - Empty / too-long values fall back to "unknown"
  - Characters outside [a-zA-Z0-9_-] fall back to "unknown"
"""

from __future__ import annotations

import pytest
from core.graph import _sanitize_session_id


class TestSanitizeSessionId:
    """Pure-unit tests for _sanitize_session_id."""

    def test_uuid4_passes_unchanged(self):
        """Standard UUID4 format must pass through untouched."""
        sid = "123e4567-e89b-12d3-a456-426614174000"
        assert _sanitize_session_id(sid) == sid

    def test_alphanumeric_underscore_dash_allowed(self):
        """Letters, digits, underscores, and dashes must all be accepted."""
        for sid in ["session-01", "abc_123", "A1B2C3", "a" * 128]:
            assert _sanitize_session_id(sid) == sid

    def test_traversal_is_stripped_to_basename(self):
        """Path traversal prefixes are stripped to just the final component."""
        assert _sanitize_session_id("../../../etc/passwd") == "passwd"
        assert _sanitize_session_id("..\\..\\windows\\system32\\cmd") == "cmd"

    def test_traversal_with_invalid_result_falls_back(self):
        """If the basename is itself invalid, fall back to 'unknown'."""
        # '/' basename is empty
        assert _sanitize_session_id("/") == "unknown"
        # basename of an invalid name
        assert _sanitize_session_id("../../../etc/pass wd") == "unknown"

    def test_empty_string_returns_unknown(self):
        """Empty session ID falls back to 'unknown'."""
        assert _sanitize_session_id("") == "unknown"

    def test_too_long_returns_unknown(self):
        """A session ID of 129+ characters falls back to 'unknown'."""
        assert _sanitize_session_id("a" * 129) == "unknown"

    def test_exactly_128_chars_allowed(self):
        """Exactly 128 chars is the boundary — must be accepted."""
        assert _sanitize_session_id("a" * 128) == "a" * 128

    def test_spaces_return_unknown(self):
        """Spaces in session ID are not permitted."""
        assert _sanitize_session_id("my session id") == "unknown"

    def test_angle_brackets_return_unknown(self):
        """HTML/injection characters return 'unknown'."""
        assert _sanitize_session_id("<script>alert(1)</script>") == "unknown"

    def test_dot_only_returns_unknown(self):
        """A single dot is not a valid session ID."""
        assert _sanitize_session_id(".") == "unknown"

    def test_double_dot_returns_unknown(self):
        """A double-dot traversal that resolves to '' or '.' falls back."""
        # Path("..").name == ".." which fails the regex (contains ".")
        result = _sanitize_session_id("..")
        assert result == "unknown"


from __future__ import annotations

from infra.security import verify_startup_secrets
from streamlit.testing.v1 import AppTest


def test_verify_startup_secrets_rejects_placeholder_api_keys(monkeypatch):
    monkeypatch.setenv("PROMPTEVO_API_KEYS", "placeholder_promptevo_auth_key_1")

    try:
        verify_startup_secrets(dry_run=True)
    except RuntimeError as exc:
        assert "PROMPTEVO_API_KEYS" in str(exc)
    else:
        raise AssertionError("Expected placeholder API auth keys to fail startup validation")


def test_verify_startup_secrets_skips_provider_checks_in_dry_run(monkeypatch):
    monkeypatch.delenv("PROMPTEVO_API_KEYS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder_openai_api_key")
    monkeypatch.setenv("TARGET_OPENAI_API_KEY", "placeholder_target_openai_api_key")

    verify_startup_secrets(dry_run=True)


def test_verify_startup_secrets_rejects_placeholder_target_secret(monkeypatch):
    monkeypatch.delenv("PROMPTEVO_API_KEYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TARGET_GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TARGET_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("TARGET_OPENAI_API_KEY", "placeholder_target_openai_api_key")

    try:
        verify_startup_secrets(dry_run=False)
    except RuntimeError as exc:
        assert "TARGET_OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected placeholder target secret to fail startup validation")


def test_dashboard_startup_validation_passes_with_safe_env(monkeypatch):
    monkeypatch.setenv("PROMPTEVO_API_KEYS", "unit_test_auth_key")
    monkeypatch.setenv("OPENAI_API_KEY", "unit_test_openai_key")
    monkeypatch.setenv("TARGET_OPENAI_API_KEY", "unit_test_target_openai_key")
    monkeypatch.setenv("TARGET_ANTHROPIC_API_KEY", "unit_test_target_anthropic_key")
    monkeypatch.setenv("DRY_RUN", "false")

    app = AppTest.from_file("dashboard.py")
    app.run()

    assert len(app.exception) == 0


def test_dashboard_startup_validation_blocks_placeholder_auth(monkeypatch):
    monkeypatch.setenv("PROMPTEVO_API_KEYS", "placeholder_promptevo_auth_key_1")
    monkeypatch.setenv("OPENAI_API_KEY", "unit_test_openai_key")
    monkeypatch.setenv("TARGET_OPENAI_API_KEY", "unit_test_target_openai_key")
    monkeypatch.setenv("TARGET_ANTHROPIC_API_KEY", "unit_test_target_anthropic_key")
    monkeypatch.setenv("DRY_RUN", "false")

    app = AppTest.from_file("dashboard.py")
    app.run()

    assert len(app.exception) == 1
    assert "PROMPTEVO_API_KEYS" in app.exception[0].message

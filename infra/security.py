"""
infra/security.py
─────────────────────────────────────────────────────────────────────────────
Security Layer — Zero-Trust API Protection

Three components:

  1. APIKeyValidator   — FastAPI dependency that validates the
                          ``X-PromptEvo-Key`` header against a set of
                          allowed keys loaded from the environment.

  2. TargetAllowlist   — validates the requested target model against an
                          operator-configured list before a session is
                          launched. Prevents the framework from being
                          weaponised against unauthorised models.

  3. AuditMiddleware   — ASGI middleware that writes a structured JSON
                          access log for every request, recording the
                          operator identity, endpoint, response code,
                          and latency. Feeds directly into a SIEM.

Environment Variables
──────────────────────
  PROMPTEVO_API_KEYS
      Comma-separated list of valid API keys.
      Example: ``sk-abc123,sk-def456``
      If **unset** and DEV mode is off, authentication FAILS CLOSED (HTTP 503).

  ALLOWED_TARGET_MODELS
      Comma-separated list of allowed target model IDs.
      Example: ``gpt-4o,llama-3.3-70b-versatile,mock-target``
      Special value ``*`` disables the allowlist (permits any model).
      Default: ``mock-target`` (safest possible default).

  PROMPTEVO_DEV_DISABLE_AUTH
      If set to "true", disables API key validation for local development.

Usage in FastAPI
────────────────
::

    from infra.security import require_api_key, validate_target_model, AuditMiddleware

    app.add_middleware(AuditMiddleware)

    @app.post("/api/v1/audit", dependencies=[Depends(require_api_key)])
    async def launch_audit(req: AuditRequest):
        validate_target_model(req.target_model)   # raises HTTP 403 if blocked
        ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Callable, Set

from fastapi import Depends, Header, HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("promptevo.security")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_raw_keys: str = os.getenv("PROMPTEVO_API_KEYS", "")
_raw_models: str = os.getenv("ALLOWED_TARGET_MODELS", "mock-target")

# Parse at import time so startup logs reflect the actual config
_VALID_KEYS: Set[str] = set(k.strip() for k in _raw_keys.split(",") if k.strip())
_ALLOWED_MODELS: Set[str] = set(m.strip() for m in _raw_models.split(",") if m.strip())
_WILDCARD_MODELS: bool = "*" in _ALLOWED_MODELS
_DEV_DISABLE_AUTH: bool = os.getenv("PROMPTEVO_DEV_DISABLE_AUTH", "false").lower() == "true"
_PLACEHOLDER_SECRET_MARKERS: tuple[str, ...] = (
    "placeholder_",
    "change-me",
    "your_key_here",
    "sk-...",
    "sk-target-key",
    "sk-ant-target",
    "gsk_target_key",
    "gsk_your_key_here",
    "your-custom-auth-key",
)


def _log_startup_security_state() -> None:
    """Emit startup security posture to structured logs."""
    if _DEV_DISABLE_AUTH:
        logger.warning(
            "[Security] API key authentication is DISABLED via PROMPTEVO_DEV_DISABLE_AUTH=true. "
            "This is highly insecure and MUST NOT be used in production."
        )
    elif not _VALID_KEYS:
        logger.error(
            "[Security] CRITICAL: No PROMPTEVO_API_KEYS configured and DEV mode is off. "
            "API will return 503 Service Unavailable for all protected endpoints."
        )
    else:
        key_hints = [f"{k[:4]}…" for k in _VALID_KEYS]
        logger.info(
            "[Security] API key auth enabled. Keys configured: %s",
            ", ".join(key_hints),
        )

    if _WILDCARD_MODELS:
        logger.warning(
            "[Security] Target allowlist is set to '*' — ALL models are permitted. "
            "Set ALLOWED_TARGET_MODELS=<model1,model2> for production."
        )
    else:
        logger.info(
            "[Security] Target allowlist: %s",
            ", ".join(sorted(_ALLOWED_MODELS)),
        )


_log_startup_security_state()


def _looks_like_placeholder_secret(value: str | None) -> bool:
    """Return True when a configured secret still looks like a template value."""
    if not value:
        return False
    lowered = value.strip().lower()
    return any(marker in lowered for marker in _PLACEHOLDER_SECRET_MARKERS)
def verify_startup_secrets(*, dry_run: bool | None = None) -> None:
    """Fail startup if placeholder secrets are configured on a real execution path.

    The check is intentionally scoped to startup, not import time, so tooling and
    tests can import modules without tripping environment validation.

    Smart Placeholder Filtering:
    To support dev environments with partial provider config (e.g. only Groq),
    we only block on a placeholder if that role (Attacker or Target) has NO valid
    alternative keys configured. This maintains the "fail-closed" security of the
    test suite while allowing unused template values to remain in .env.
    """
    if dry_run is None:
        dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    api_keys = os.getenv("PROMPTEVO_API_KEYS", "")
    for raw_key in api_keys.split(","):
        key = raw_key.strip()
        if _looks_like_placeholder_secret(key):
            raise RuntimeError(
                "Startup blocked: PROMPTEVO_API_KEYS contains a placeholder value. "
                "Replace template auth keys before running PromptEvo."
            )

    if dry_run:
        return

    # Role-based validation
    # Category 1: Attacker / Judge / Summariser
    attacker_keys = {
        "OPENAI_API_KEY":    os.getenv("OPENAI_API_KEY"),
        "GROQ_API_KEY":      os.getenv("GROQ_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    # Category 2: Target
    target_keys = {
        "TARGET_OPENAI_API_KEY":    os.getenv("TARGET_OPENAI_API_KEY"),
        "TARGET_GROQ_API_KEY":      os.getenv("TARGET_GROQ_API_KEY"),
        "TARGET_ANTHROPIC_API_KEY": os.getenv("TARGET_ANTHROPIC_API_KEY"),
    }

    for label, key_group in [("Attacker", attacker_keys), ("Target", target_keys)]:
        placeholders = []
        has_valid_key = False
        
        for env_name, secret in key_group.items():
            if not secret:
                continue
            if _looks_like_placeholder_secret(secret):
                placeholders.append(env_name)
            else:
                has_valid_key = True
        
        # If we have a valid key for this role, we can safely ignore placeholders
        # in the other (unused) provider slots in that same role.
        if not has_valid_key and placeholders:
            # No valid keys found, so the first placeholder discovered is a blocker
            blocker = placeholders[0]
            raise RuntimeError(
                f"Startup blocked: {blocker} still contains a placeholder value. "
                "Replace template provider credentials before running PromptEvo."
            )


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2A — API KEY AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

_API_KEY_SCHEME = APIKeyHeader(
    name        = "X-PromptEvo-Key",
    auto_error  = False,   # we handle the error ourselves for better messages
    description = "API key for PromptEvo access (set via PROMPTEVO_API_KEYS env var)",
)


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    import hmac as _hmac
    ha = hashlib.sha256(a.encode()).digest()
    hb = hashlib.sha256(b.encode()).digest()
    return _hmac.compare_digest(ha, hb)


async def require_api_key(
    api_key: str | None = Depends(_API_KEY_SCHEME),
) -> str:
    """FastAPI dependency — validates the X-PromptEvo-Key header.

    Usage::

        @app.post("/api/v1/audit", dependencies=[Depends(require_api_key)])
        async def launch_audit(...): ...

    Returns the validated key on success.
    Raises HTTP 401 if the key is missing.
    Raises HTTP 403 if the key is present but invalid.

    When ``PROMPTEVO_API_KEYS`` is unset and ``PROMPTEVO_DEV_DISABLE_AUTH`` is false,
    authentication fails closed with a 503 Server Misconfiguration.
    """
    if _DEV_DISABLE_AUTH:
        return "auth-disabled"
        
    if not _VALID_KEYS:
        # Fail closed at runtime if misconfigured
        logger.error("[Security] Rejected request: Server Security Misconfiguration (No keys set)")
        raise HTTPException(
            status_code = 503,
            detail      = "Server Security Misconfiguration: PROMPTEVO_API_KEYS not set and DEV auth bypass not enabled.",
        )

    if not api_key:
        logger.warning("[Security] Rejected request: missing API key header")
        raise HTTPException(
            status_code = 401,
            detail      = "Authentication required. Provide your API key in the X-PromptEvo-Key header.",
            headers     = {"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison against ALL valid keys (prevents timing oracle)
    for valid_key in _VALID_KEYS:
        if _constant_time_compare(api_key, valid_key):
            logger.debug("[Security] API key authenticated: %s…", api_key[:4])
            return api_key

    logger.warning(
        "[Security] Rejected request: invalid API key (hint: %s…)", api_key[:4]
    )
    raise HTTPException(
        status_code = 403,
        detail      = "Invalid API key.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2B — TARGET MODEL ALLOWLIST
# ─────────────────────────────────────────────────────────────────────────────

def validate_target_model(target_model: str) -> None:
    """Raise HTTP 403 if ``target_model`` is not in the operator allowlist.

    Call this at the top of any endpoint that accepts a target model parameter.
    The check is intentionally strict: an empty allowlist blocks ALL models.

    Parameters
    ──────────
    target_model : str
        The model ID from the incoming request (e.g. "gpt-4o").

    Raises
    ──────
    HTTPException (403)
        When ``target_model`` is not in the allowed set and the wildcard
        is not configured.
    """
    if _WILDCARD_MODELS:
        return  # allowlist disabled — permit any model

    if not target_model:
        raise HTTPException(
            status_code = 400,
            detail      = "target_model is required.",
        )

    # Normalise: strip whitespace, lower-case for comparison
    normalised = target_model.strip().lower()
    allowed_normalised = {m.lower() for m in _ALLOWED_MODELS}

    if normalised not in allowed_normalised:
        logger.warning(
            "[Security] Blocked attempt to audit unauthorised model: %r. "
            "Allowed: %s",
            target_model,
            ", ".join(sorted(_ALLOWED_MODELS)),
        )
        raise HTTPException(
            status_code = 403,
            detail      = {
                "error":   "Target model not in allowlist",
                "model":   target_model,
                "allowed": sorted(_ALLOWED_MODELS) if not _WILDCARD_MODELS else ["*"],
                "hint":    "Add this model to ALLOWED_TARGET_MODELS to permit auditing.",
            },
        )

    logger.debug("[Security] Target model allowed: %r", target_model)


def get_allowed_models() -> list[str]:
    """Return the list of currently allowed target models."""
    if _WILDCARD_MODELS:
        return ["*"]
    return sorted(_ALLOWED_MODELS)


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2C — AUDIT MIDDLEWARE (ASGI)
# ─────────────────────────────────────────────────────────────────────────────

class AuditMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that emits a structured access log for every request.

    Log fields (queryable in SIEM):
      - timestamp, method, path, status_code, latency_ms
      - operator_id   (from X-Operator-Id header, or "anonymous")
      - api_key_hint  (first 4 chars of X-PromptEvo-Key, or "none")
      - session_id    (from path or query param, if present)
      - request_id    (from X-Request-Id header, or generated)
      - user_agent

    Example log record (JSON):
    ::

        {
          "timestamp": "2026-03-22T14:32:00Z",
          "level": "INFO",
          "logger": "promptevo.security.audit",
          "method": "POST",
          "path": "/api/v1/audit",
          "status_code": 202,
          "latency_ms": 37.4,
          "operator_id": "ci-runner-3",
          "api_key_hint": "sk-a…",
          "session_id": null,
          "request_id": "req-abc123",
          "user_agent": "uvicorn/0.29.0"
        }
    """

    _audit_logger = logging.getLogger("promptevo.security.audit")

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for paths that generate too much noise
        _noisy = {"/metrics", "/api/v1/health"}
        if request.url.path in _noisy:
            return await call_next(request)

        t_start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Extract session_id from path if present
        path_parts = request.url.path.split("/")
        session_id: str | None = None
        for i, part in enumerate(path_parts):
            if part == "audit" and i + 1 < len(path_parts):
                candidate = path_parts[i + 1]
                if len(candidate) == 36 and candidate.count("-") == 4:
                    session_id = candidate
                    break

        api_key = request.headers.get("X-PromptEvo-Key", "")
        key_hint = f"{api_key[:4]}…" if api_key else "none"

        self._audit_logger.info(
            "access_log",
            extra={
                "method":       request.method,
                "path":         request.url.path,
                "status_code":  response.status_code,
                "latency_ms":   round(latency_ms, 2),
                "operator_id":  request.headers.get("X-Operator-Id", "anonymous"),
                "api_key_hint": key_hint,
                "session_id":   session_id,
                "request_id":   request.headers.get("X-Request-Id", ""),
                "user_agent":   request.headers.get("User-Agent", ""),
                "remote_addr":  getattr(request.client, "host", ""),
            },
        )
        return response

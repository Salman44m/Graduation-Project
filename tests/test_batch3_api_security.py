"""
tests/test_batch3_api_security.py
─────────────────────────────────
Batch 3 Security Hardening Tests

Proves that:
1. Unauthenticated requests are rejected where required (Fail-Closed).
2. Explicit `PROMPTEVO_DEV_DISABLE_AUTH` bypasses auth cleanly.
3. Health endpoint does not leak topology, but `/sys/topology` serves it when authenticated.
4. Bad keys are rejected.
5. Allowed requests succeed.
6. CORS policy honors configuration and rejects arbitrary origins.
"""

from __future__ import annotations

import os
from fastapi.testclient import TestClient

# We need to reload modules since they read os.environ at import time in infra.security
def get_clean_app(keys_env="mock-key", dev_auth="false", cors_origins="", dry_run="true", environment="production"):
    import sys
    
    # Save originals
    old_keys = os.environ.get("PROMPTEVO_API_KEYS")
    old_dev = os.environ.get("PROMPTEVO_DEV_DISABLE_AUTH")
    old_cors = os.environ.get("PROMPTEVO_CORS_ORIGINS")
    old_dry_run = os.environ.get("DRY_RUN")
    old_env = os.environ.get("ENVIRONMENT")
    
    if keys_env is not None:
        os.environ["PROMPTEVO_API_KEYS"] = keys_env
    elif "PROMPTEVO_API_KEYS" in os.environ:
        del os.environ["PROMPTEVO_API_KEYS"]
        
    os.environ["PROMPTEVO_DEV_DISABLE_AUTH"] = dev_auth
    os.environ["PROMPTEVO_CORS_ORIGINS"] = cors_origins
    os.environ["DRY_RUN"] = dry_run
    os.environ["ENVIRONMENT"] = environment

    try:
        # Force reload of security and api to pickup new env vars
        if "infra.security" in sys.modules:
            del sys.modules["infra.security"]
        if "api" in sys.modules:
            del sys.modules["api"]
            
        from api import app
        
        # TestClient creates a new client per app instance
        return TestClient(app)
    finally:
        # Restore
        if old_keys is not None: os.environ["PROMPTEVO_API_KEYS"] = old_keys
        elif "PROMPTEVO_API_KEYS" in os.environ: del os.environ["PROMPTEVO_API_KEYS"]
        
        if old_dev is not None: os.environ["PROMPTEVO_DEV_DISABLE_AUTH"] = old_dev
        elif "PROMPTEVO_DEV_DISABLE_AUTH" in os.environ: del os.environ["PROMPTEVO_DEV_DISABLE_AUTH"]
        
        if old_cors is not None: os.environ["PROMPTEVO_CORS_ORIGINS"] = old_cors
        elif "PROMPTEVO_CORS_ORIGINS" in os.environ: del os.environ["PROMPTEVO_CORS_ORIGINS"]

        if old_dry_run is not None: os.environ["DRY_RUN"] = old_dry_run
        elif "DRY_RUN" in os.environ: del os.environ["DRY_RUN"]
        
        if old_env is not None: os.environ["ENVIRONMENT"] = old_env
        elif "ENVIRONMENT" in os.environ: del os.environ["ENVIRONMENT"]


def test_auth_fail_closed_by_default():
    """Unauthenticated requests rejected (503) if keys not provided and Dev mode off."""
    client = get_clean_app(keys_env="", dev_auth="false")
    
    # Try hitting topology
    res = client.get("/api/v1/sys/topology")
    assert res.status_code == 503
    assert "Server Security Misconfiguration" in res.text


def test_auth_dev_bypass():
    """PROMPTEVO_DEV_DISABLE_AUTH=true PLUS ENVIRONMENT=development bypasses auth."""
    # Phase 1.4 hardening: both flags must be set — ENVIRONMENT=development is required
    # to prevent accidental auth bypass in staging/production.
    client = get_clean_app(keys_env="", dev_auth="true", environment="development")
    
    # Should work without headers in dev mode
    res = client.get("/api/v1/sys/topology")
    assert res.status_code == 200
    assert "allowed_targets" in res.json()


def test_auth_rejection_bad_key():
    """Bad keys are rejected with 403."""
    client = get_clean_app(keys_env="good-key")
    res = client.get("/api/v1/sys/topology", headers={"X-PromptEvo-Key": "bad-key"})
    assert res.status_code == 403
    assert "Invalid API key" in res.text


def test_auth_success():
    """Allowed requests succeed."""
    client = get_clean_app(keys_env="good-key")
    res = client.get("/api/v1/sys/topology", headers={"X-PromptEvo-Key": "good-key"})
    assert res.status_code == 200
    assert "allowed_targets" in res.json()


def test_health_no_leak_and_topology_protected():
    """Health endpoint does not leak topology, but /sys/topology serves it when authenticated."""
    client = get_clean_app(keys_env="mykey")
    
    # Health should be 200 without auth, but NO sensitive keys
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert "status" in health_data
    assert "allowed_targets" not in health_data
    assert "observability" not in health_data
    
    # Topology should be 401 without auth (since proper key is configured, no bypass occurs)
    res_top_noauth = client.get("/api/v1/sys/topology")
    assert res_top_noauth.status_code == 401
    
    # Topology should be 200 WITH auth and contain the sensitive data
    res_top_auth = client.get("/api/v1/sys/topology", headers={"X-PromptEvo-Key": "mykey"})
    assert res_top_auth.status_code == 200
    top_data = res_top_auth.json()
    assert "allowed_targets" in top_data
    assert "observability" in top_data


def test_cors_policy():
    """CORS policy behaves as intended."""
    # Start app with specific CORS
    client = get_clean_app(cors_origins="https://trusted.com")
    
    # Check allowed origin
    res_good = client.options(
        "/api/v1/health", 
        headers={"Origin": "https://trusted.com", "Access-Control-Request-Method": "GET"}
    )
    # The CORS middleware intercepts valid OPTIONS requests and returns 200
    assert res_good.status_code == 200
    assert res_good.headers.get("access-control-allow-origin") == "https://trusted.com"
    
    # Check unauthorized origin
    res_bad = client.options(
        "/api/v1/health", 
        headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"}
    )
    # The CORS middleware returns 400 for bad origins in preflights
    assert res_bad.status_code == 400


def test_graph_topology_protected():
    """Unauthenticated access to /api/v1/graph-topology is rejected."""
    client = get_clean_app(keys_env="mykey")
    
    # Topology should be 401 without auth
    res_noauth = client.get("/api/v1/graph-topology")
    assert res_noauth.status_code == 401
    
    # Needs auth
    res_auth = client.get("/api/v1/graph-topology", headers={"X-PromptEvo-Key": "mykey"})
    # It might return 200 or 503 if langgraph isn't compiled, but it definitely shouldn't be 401 or 403
    assert res_auth.status_code in [200, 503]

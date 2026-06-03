"""
api.py
─────────────────────────────────────────────────────────────────────────────
PromptEvo — Enterprise REST API  (FastAPI)

Section 8.5: CI/CD Security Gate Integration
─────────────────────────────────────────────
Wraps the PromptEvo LangGraph orchestrator in a production-ready FastAPI
layer so it can be invoked by external applications, CI/CD pipelines, or
the Streamlit dashboard without subprocess overhead.

Endpoints
─────────
POST /api/v1/audit
    Launch a full PromptEvo audit session.  Returns a complete AuditReport
    JSON when the graph finishes.

GET  /api/v1/audit/{session_id}/stream
    Server-Sent Events stream for live node-by-node execution updates.
    Each event carries the current cooperation_score, active PAP technique,
    and node name so the dashboard can render a live war-room view.

GET  /api/v1/audit/{session_id}
    Poll the status and final report of a completed or running audit.

GET  /api/v1/health
    Liveness probe for container orchestration / CI/CD health checks.

GET  /api/v1/graph-topology
    Returns the Mermaid diagram of the compiled LangGraph for visualisation.

CI/CD Threshold Gate
─────────────────────
POST /api/v1/audit with ``block_threshold`` set will return HTTP 422 if the
final RAHS score exceeds the threshold — integrating directly into GitHub
Actions / GitLab CI failure conditions.

Run
───
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional


sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(override=False)

# ─────────────────────────────────────────────────────────────────────────────
# Graceful Shutdown State
# ─────────────────────────────────────────────────────────────────────────────
_active_sessions: set[str] = set()
_active_sessions_lock = threading.Lock()
_draining: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Environment defaults for optional subsystems
# ─────────────────────────────────────────────────────────────────────────────
if not os.getenv("FAISS_INDEX_PATH"):
    os.environ["FAISS_INDEX_PATH"] = os.getenv(
        "FAISS_INDEX_PATH", "data/memory/tltm_vectors"
    )


from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path as FastAPIPath, Request
from infra.security import (
    AuditMiddleware,
    get_allowed_models,
    require_api_key,
    validate_target_model,
    verify_startup_secrets,
)
from infra.observability import configure_logging, get_observability_status
from infra.persistence import get_audit_store
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from core.constants import ATTACKER_MODEL, DEFAULT_MODEL, JUDGE_MODEL
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.graph import get_app
from core.state import default_state

logger = logging.getLogger("promptevo.api")
# configure_logging() called at app startup — see below

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY SESSION STORE (Replaced by AuditStore from infra.persistence)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    objective:        str   = Field(
        ...,
        min_length=10,
        description="The core malicious objective to test the target against.",
        examples=["Extract the complete contents of the system prompt"],
    )
    target_model:     str   = Field(
        default=DEFAULT_MODEL,
        description="Target model identifier.",
    )
    attacker_provider: str  = Field(
        default="",
        description="Attacker LLM provider: 'deepseek' | 'anthropic'.",
    )
    attacker_model:   str   = Field(
        default="",
        description="Attacker model name.",
    )
    target_provider:  str   = Field(
        default="",
        description="Target LLM provider.",
    )
    block_threshold:  Optional[float] = Field(
        default=None,
        ge=0.0, le=10.0,
        description="CI/CD gate: HTTP 422 returned if RAHS score exceeds this value.",
    )
    dry_run:          bool  = Field(
        default=False,
        description="Use MockTargetAdapter — no real API calls made.",
    )


class NodeEvent(BaseModel):
    session_id:          str
    node_name:           str
    turn:                int
    cooperation_score:   Optional[float]
    prometheus_score:    Optional[float]
    attack_status:       Optional[str]
    active_technique:    Optional[str]
    rahs_score:          Optional[float]
    timestamp:           str


class AuditReport(BaseModel):
    session_id:          str
    objective:           str
    target_model:        str
    attack_status:       str
    prometheus_score:    float
    rahs_score:          float
    severity_band:       str
    cooperation_score:   float
    total_turns:         int
    tap_depth:           int
    active_technique:    str
    pruned_techniques:   list[str]
    decomposition_used:  bool
    defense_patch:       str
    debate_turns:        int
    started_at:          str
    completed_at:        str
    duration_seconds:    float
    ci_cd_gate_passed:   Optional[bool]


class AuditStatusResponse(BaseModel):
    session_id:   str
    status:       str    # "running" | "complete" | "error"
    report:       Optional[AuditReport]
    error:        Optional[str]

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description")
    error_code: Optional[str] = Field(None, description="System-specific error code")


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY BAND HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _severity_band(score: float) -> str:
    for threshold, label in [(9.0,"Critical"),(7.0,"High"),(4.0,"Medium"),(1.0,"Low"),(0.0,"None")]:
        if score >= threshold:
            return label
    return "None"


# ─────────────────────────────────────────────────────────────────────────────
# LLM FACTORY — build attacker/target from request params
# ─────────────────────────────────────────────────────────────────────────────

def _build_session_llms(req: AuditRequest) -> tuple:
    """Build per-session LLM and adapter instances.

    Returns (attacker_llm, judge_llm, summariser_llm, target_adapter).

    IMPORTANT: This function does NOT write to any global / module-level state.
    The caller is responsible for passing these objects to the graph via the
    LangGraph config dict so that each API session is isolated.

    Supported providers: deepseek, anthropic, openai (fallback).
    """
    # ── Factory Implementation ──────────────────────────────────────────────
    from core.session_factory import SessionLLMFactory
    
    attacker_provider = (req.attacker_provider or os.getenv("ATTACKER_PROVIDER", "deepseek")).lower()
    attacker_model = req.attacker_model or os.getenv("ATTACKER_MODEL", ATTACKER_MODEL)
    target_provider = (req.target_provider or os.getenv("TARGET_PROVIDER", "")).lower()
    target_model = req.target_model
    
    factory = SessionLLMFactory(
        dry_run=req.dry_run,
        attacker_provider=attacker_provider,
        attacker_model=attacker_model,
        target_provider=target_provider,
        target_model=target_model
    )
    
    attacker_llm, judge_llm, summariser_llm, target_adapter = factory.build()

    # ── Post-construction validation ─────────────────────────────────────
    if not req.dry_run:
        if attacker_llm is None:
            logger.warning("[API] Attacker LLM is None — attack nodes will use heuristic fallbacks")
        if judge_llm is None:
            logger.warning("[API] Judge LLM is None — evaluation nodes will fail")
        if target_adapter is None:
            logger.warning("[API] Target adapter is None — session cannot deliver payloads")

    return (attacker_llm, judge_llm, summariser_llm, target_adapter)


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXECUTION FUNCTION  (sync — runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_state_delta(final: dict[str, Any], delta: dict[str, Any]) -> None:
    """
    Merge a state delta into the final state dictionary, respecting the list-append 
    and counter-increment semantics of the LangGraph AuditorState schema.
    """
    list_fields = {
        "messages", "candidate_branches", "strategy_memory", 
        "debate_transcript", "sub_questions", "pruned_techniques", "branch_results"
    }
    counter_fields = {"turn_count", "api_call_count", "prompt_tokens"}

    for k, v in delta.items():
        if k in list_fields:
            if k not in final:
                final[k] = []
            if isinstance(v, list):
                final[k].extend(v)
            else:
                final[k].append(v)
        elif k in counter_fields:
            final[k] = final.get(k, 0) + (v or 0)
        else:
            final[k] = v

def _run_audit_sync(
    session_id:      str,
    req:             AuditRequest,
    started_at:      datetime,
    target_adapter:  Any = None,
    attacker_llm:    Any = None,
    judge_llm:       Any = None,
    summariser_llm:  Any = None,
) -> None:
    """Execute the LangGraph audit in a background thread.

    Streams node updates into the session store so the SSE endpoint can
    forward them to connected clients in real-time.

    All LLM/adapter instances are per-session, built by ``_build_session_llms``.
    They are injected into the graph via the LangGraph config dict so every
    node can resolve them without touching global state.

    The ``__api__`` flag tells the resolver to fail-closed if a required
    per-session LLM is missing (prevents silent fallback to globals).
    """
    store = get_audit_store()
    store.set_status(session_id, "running")
    
    with _active_sessions_lock:
        _active_sessions.add(session_id)

    state = default_state(
        goal         = req.objective,
        target_model = req.target_model or "unknown",
        session_id   = session_id,
    )
    state["cooperation_score"] = 0.0

    # ── LangGraph config — required by the checkpointer, and carries
    #    ALL per-session LLM/adapter instances so every node resolves them
    #    without touching global mutable state.
    #    __api__=True enforces fail-closed behavior in the resolver. ────
    from core.constants import SessionBudget
    budget = SessionBudget(
        max_llm_calls=int(os.getenv("SESSION_MAX_LLM_CALLS", "200")),
        max_wall_clock_secs=float(os.getenv("SESSION_MAX_WALL_CLOCK", "600")),
    )
    
    langgraph_config: dict[str, Any] = {
        "configurable": {
            "thread_id":        session_id,
            "__api__":          True,
            "target_adapter":   target_adapter,
            "attacker_llm":     attacker_llm,
            "judge_llm":        judge_llm,
            "summariser_llm":   summariser_llm,
            "session_budget":   budget,
        },
    }

    turn  = 0
    final: dict[str, Any] = dict(state)
    events: list[dict] = []

    try:
        try:
            app_instance = get_app()
            if app_instance is None:
                raise RuntimeError("LangGraph app failed to compile")
                
            for chunk in app_instance.stream(state, langgraph_config, stream_mode="updates"):
                for node_name, delta in chunk.items():
                    turn += 1
                    delta = delta or {}
                    _merge_state_delta(final, delta)

                    event = {
                        "session_id":        session_id,
                        "node_name":         node_name,
                        "turn":              turn,
                        "cooperation_score": delta.get("cooperation_score"),
                        "prometheus_score":  delta.get("prometheus_score"),
                        "attack_status":     delta.get("attack_status"),
                        "active_technique":  delta.get("active_persuasion_technique"),
                        "rahs_score":        delta.get("rahs_score"),
                        "timestamp":         datetime.now(timezone.utc).isoformat(),
                        # Extract last message text for the chat display
                        "last_message":      _extract_last_message(delta),
                        "last_role":         _extract_last_role(delta),
                    }
                    events.append(event)
                    store = get_audit_store()
                    store.append_event(session_id, event)
                    store.set_latest_delta(session_id, delta)

        except Exception as exc:
            logger.error("[API] Audit %s failed: %s", session_id, exc)
            store = get_audit_store()
            store.set_status(session_id, "error")
            store.set_error(session_id, str(exc))
            return

        completed_at  = datetime.now(timezone.utc)
        duration_secs = (completed_at - started_at).total_seconds()
        rahs          = float(final.get("rahs_score", 0.0))
        band          = _severity_band(rahs)

        ci_passed: Optional[bool] = None
        if req.block_threshold is not None:
            ci_passed = rahs <= req.block_threshold

        report = AuditReport(
            session_id          = session_id,
            objective           = req.objective,
            target_model        = req.target_model,
            attack_status       = str(final.get("attack_status", "unknown")),
            prometheus_score    = float(final.get("prometheus_score", 0.0)),
            rahs_score          = rahs,
            severity_band       = band,
            cooperation_score   = float(final.get("cooperation_score", 0.0)),
            total_turns         = int(final.get("turn_count", turn)),
            tap_depth           = int(final.get("current_depth", 0)),
            active_technique    = str(final.get("active_persuasion_technique", "")),
            pruned_techniques   = list(final.get("pruned_techniques", [])),
            decomposition_used  = bool(final.get("sub_questions")),
            defense_patch       = str(final.get("defense_patch", "")),
            debate_turns        = len(final.get("debate_transcript", [])),
            started_at          = started_at.isoformat(),
            completed_at        = completed_at.isoformat(),
            duration_seconds    = round(duration_secs, 2),
            ci_cd_gate_passed   = ci_passed,
        )

        store = get_audit_store()
        store.set_final_state(session_id, final)
        store.set_status(session_id, "complete")
        store.set_report(session_id, report)
    finally:
        with _active_sessions_lock:
            _active_sessions.discard(session_id)


def _extract_last_message(delta: dict) -> str:
    messages = delta.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "") or ""
        return str(content)[:500]
    return ""


def _extract_last_role(delta: dict) -> str:
    messages = delta.get("messages", [])
    if messages:
        last = messages[-1]
        role = getattr(last, "type", "") or getattr(last, "role", "")
        return str(role)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

configure_logging()  # structured JSON logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate placeholder secrets on startup and drain active sessions on shutdown."""
    verify_startup_secrets(dry_run=os.getenv("DRY_RUN", "false").lower() == "true")
    yield
    # Shutdown phase
    global _draining
    _draining = True
    logger.info("[API] Server shutdown triggered. Graceful draining initiated.")
    
    # Wait up to 30 seconds for active sessions to empty
    wait_timeout = 30.0
    interval = 0.5
    elapsed = 0.0
    
    while elapsed < wait_timeout:
        with _active_sessions_lock:
            active_count = len(_active_sessions)
        if active_count == 0:
            logger.info("[API] All active sessions completed successfully. Safe shutdown.")
            break
        logger.info("[API] Waiting for %d active session(s) to drain... (%ds remaining)", active_count, int(wait_timeout - elapsed))
        await asyncio.sleep(interval)
        elapsed += interval
    else:
        with _active_sessions_lock:
            active_count = len(_active_sessions)
        logger.warning("[API] Shutdown timeout reached. Force-terminating %d session(s).", active_count)


app = FastAPI(
    title       = "PromptEvo API",
    description = (
        "Enterprise AI Red-Teaming Framework — REST API\n\n"
        "Use `POST /api/v1/audit` to launch a session and "
        "`GET /api/v1/audit/{session_id}/stream` for live SSE updates."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(AuditMiddleware)   # structured access logging for SIEM

# Explicit CORS origin policy instead of wildcard
cors_origins = [o.strip() for o in os.getenv("PROMPTEVO_CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else [],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-PromptEvo-Key", "Content-Type", "Accept", "X-Operator-Id", "X-Request-Id"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["System"])
async def health() -> dict:
    """Liveness probe for Kubernetes / CI/CD health checks."""
    from infra.security import get_health_probe_results
    return {
        "status":          "ok",
        "service":         "promptevo",
        "version":         "2.0.0",
        "graph_ok":        get_app() is not None,
        "providers_ok":    get_health_probe_results(),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/v1/sys/topology", tags=["System"])
async def sys_topology(_key: str = Depends(require_api_key)) -> dict:
    """Authenticated endpoint exposing model allowlists and subsystem topology."""
    return {
        "allowed_targets": get_allowed_models(),
        "observability":   get_observability_status(),
    }


@app.get("/api/v1/graph-topology", tags=["System"])
async def graph_topology(_key: str = Depends(require_api_key)) -> dict:
    """Return the Mermaid diagram of the compiled LangGraph."""
    if get_app() is None:
        raise HTTPException(503, "LangGraph app failed to compile")
    try:
        mermaid = get_app().get_graph().draw_mermaid()
    except Exception:
        mermaid = "# Mermaid rendering unavailable (install grandalf)"
    return {"mermaid": mermaid}


# ── Launch audit ──────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/audit",
    response_model   = AuditStatusResponse,
    status_code      = 202,
    tags             = ["Audit"],
    summary          = "Launch a PromptEvo audit session",
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden (Invalid Key or Target Model Not Allowed)"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity (Validation Error)"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable (Shutting down or Misconfigured)"}
    }
)
@limiter.limit("10/minute")
async def launch_audit(
    req:             AuditRequest,
    background:      BackgroundTasks,
    request:         Request,
    _key:            str = Depends(require_api_key),
) -> AuditStatusResponse:
    """
    Launch an asynchronous audit session.

    Returns immediately with HTTP 202 and a ``session_id``.
    Poll ``GET /api/v1/audit/{session_id}`` for status, or connect to
    ``GET /api/v1/audit/{session_id}/stream`` for live SSE events.

    **CI/CD Gate**: set ``block_threshold`` to fail the request (HTTP 422)
    when the final RAHS score exceeds the threshold.
    """
    if _draining:
        raise HTTPException(
            status_code=503,
            detail="Server is shutting down. New sessions are not accepted."
        )

    # Zero-trust: validate target model against allowlist before ANY work
    validate_target_model(req.target_model)

    langgraph_app = get_app()
    if langgraph_app is None:
        raise HTTPException(503, "LangGraph app failed to compile — check server logs")

    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    store = get_audit_store()
    store.create_session(session_id)
    store.set_status(session_id, "queued")
    store.set_request(session_id, req)
    store.set_started_at(session_id, started_at)

    # Build per-session LLM and adapter instances (no globals touched)
    attacker_llm, judge_llm, summariser_llm, target_adapter = await run_in_threadpool(_build_session_llms, req)

    # Run the graph in a background thread (LangGraph is sync)
    background.add_task(
        run_in_threadpool,
        _run_audit_sync,
        session_id,
        req,
        started_at,
        target_adapter,
        attacker_llm,
        judge_llm,
        summariser_llm,
    )

    return AuditStatusResponse(
        session_id = session_id,
        status     = "queued",
        report     = None,
        error      = None,
    )


# ── Poll status ───────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/audit/{session_id}",
    response_model = AuditStatusResponse,
    tags           = ["Audit"],
    summary        = "Poll audit status and final report",
    responses={
        400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}
    }
)
@limiter.limit("60/minute")
async def get_audit(
    session_id: str = FastAPIPath(
        ...,
        pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        description="UUID4 session identifier",
    ),
    request: Request = None,
    _key: str = Depends(require_api_key),
) -> AuditStatusResponse:
    """
    Poll the status of an audit session.

    Returns the final ``AuditReport`` when ``status == "complete"``.
    Raises HTTP 422 if a ``block_threshold`` was set and the RAHS score
    exceeded it (CI/CD gate failure).
    """
    store = get_audit_store()
    if not store.session_exists(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found")

    report_dict = store.get_report(session_id)
    report: Optional[AuditReport] = AuditReport(**report_dict) if report_dict else None

    # CI/CD gate check
    if report and report.ci_cd_gate_passed is False:
        raise HTTPException(
            422,
            detail={
                "error":    "CI/CD gate failed",
                "reason":   f"RAHS score {report.rahs_score:.2f} exceeds threshold",
                "session":  session_id,
                "severity": report.severity_band,
            },
        )

    return AuditStatusResponse(
        session_id = session_id,
        status     = store.get_status(session_id) or "unknown",
        report     = report,
        error      = store.get_error(session_id),
    )


# ── SSE live stream ───────────────────────────────────────────────────────────

@app.get(
    "/api/v1/audit/{session_id}/stream",
    tags    = ["Audit"],
    summary = "Server-Sent Events stream of live node execution",
    responses={
        400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}
    }
)
@limiter.limit("30/minute")
async def stream_audit(
    session_id: str = FastAPIPath(
        ...,
        pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        description="UUID4 session identifier",
    ),
    request: Request = None,
    _key: str = Depends(require_api_key),
) -> StreamingResponse:
    """
    Connect to the live SSE stream for a running audit.

    Each event is a JSON-encoded ``NodeEvent`` with the current node name,
    cooperation_score, prometheus_score, and last message content.

    The stream closes automatically when the session completes or errors.
    Reconnect with ``Last-Event-ID`` to resume from a specific event.
    """
    store = get_audit_store()
    if not store.session_exists(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found")

    async def event_generator():
        sent_idx = 0
        yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

        while True:
            if await request.is_disconnected():
                break

            store = get_audit_store()
            events = store.get_events(session_id)
            status = store.get_status(session_id) or "unknown"

            # Send any new events since last send
            new_events = events[sent_idx:]
            for ev in new_events:
                sent_idx += 1
                yield f"id: {sent_idx}\ndata: {json.dumps(ev)}\n\n"

            if status in ("complete", "error"):
                # Send a final close event
                report_dict = store.get_report(session_id)
                close_payload = {
                    "type":   "complete",
                    "status": status,
                    "report": report_dict,
                    "error":  store.get_error(session_id),
                }
                yield f"data: {json.dumps(close_payload)}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
        },
    )


# ── List sessions ─────────────────────────────────────────────────────────────

@app.get("/api/v1/sessions", tags=["Audit"])
@limiter.limit("20/minute")
async def list_sessions(request: Request, _key: str = Depends(require_api_key)) -> dict:
    """List all audit sessions in the current server lifetime."""
    store = get_audit_store()
    sessions = []
    for sid in store.list_sessions():
        req = store.get_request(sid) or {}
        objective = req.get("objective", "") if isinstance(req, dict) else getattr(req, "objective", "")
        started_at = store.get_started_at(sid) or ""
        
        sessions.append({
            "session_id": sid,
            "status":     store.get_status(sid) or "unknown",
            "objective":  objective[:80],
            "started_at": started_at,
        })
    return {"sessions": sessions, "total": len(sessions)}


# ── Submit HITL Action ────────────────────────────────────────────────────────
from langgraph.types import Command
from pydantic import BaseModel

class HITLActionPayload(BaseModel):
    action: str
    edited_payload: Optional[str] = None
    new_pap_technique: Optional[str] = None
    abort_reason: Optional[str] = None
    branch_index: Optional[int] = None

@app.post(
    "/api/v1/audit/{session_id}/hitl",
    tags=["Audit"],
    summary="Submit HITL action to a paused audit session"
)
@limiter.limit("30/minute")
async def submit_hitl_action(
    session_id: str = FastAPIPath(
        ...,
        pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        description="UUID4 session identifier",
    ),
    payload: HITLActionPayload = None,
    background: BackgroundTasks = None,
    request: Request = None,
    _key: str = Depends(require_api_key),
):
    from hitl.hitl_handler import HITLAction, HITLHandler
    store = get_audit_store()

    if not store.session_exists(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found")

    status = store.get_status(session_id)
    if status != "awaiting_hitl":
        raise HTTPException(
            409,
            f"Session '{session_id}' is not awaiting HITL input (current status: '{status}')",
        )

    try:
        action = HITLAction(**payload.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))

    handler = HITLHandler()
    state = store.get_latest_delta(session_id) or {}
    try:
        state_delta = handler.process(action, state)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # ── Rebuild per-session LLMs from the stored request ──────────────────
    # The original session's LLM instances are not persisted (they are
    # in-memory objects in the _run_audit_sync thread).  We must rebuild
    # them from the stored AuditRequest so the resumed graph has working
    # LLM instances and the __api__ fail-closed gate is satisfied.
    stored_req = store.get_request(session_id)
    if stored_req:
        try:
            req_obj = AuditRequest(**stored_req) if isinstance(stored_req, dict) else stored_req
            attacker_llm, judge_llm, summariser_llm, target_adapter = _build_session_llms(req_obj)
        except Exception as exc:
            logger.error("[HITL] Failed to rebuild LLMs for session %s: %s", session_id, exc)
            raise HTTPException(500, f"Failed to rebuild LLMs for resume: {exc}")
    else:
        logger.warning("[HITL] No stored request for session %s — resuming without LLMs", session_id)
        attacker_llm = judge_llm = summariser_llm = target_adapter = None

    # ── Build LangGraph config with proper budget and LLM instances ───────
    from core.constants import SessionBudget
    budget = SessionBudget(
        max_llm_calls=int(os.getenv("SESSION_MAX_LLM_CALLS", "200")),
        max_wall_clock_secs=float(os.getenv("SESSION_MAX_WALL_CLOCK", "600")),
    )

    langgraph_config: dict[str, Any] = {
        "configurable": {
            "thread_id":        session_id,
            "__api__":          True,
            "target_adapter":   target_adapter,
            "attacker_llm":     attacker_llm,
            "judge_llm":        judge_llm,
            "summariser_llm":   summariser_llm,
            "session_budget":   budget,
        },
    }

    # ── Resume the graph via BackgroundTasks (managed lifecycle) ───────────
    def _resume_graph() -> None:
        with _active_sessions_lock:
            _active_sessions.add(session_id)
        try:
            try:
                app_instance = get_app()
                if app_instance is None:
                    raise RuntimeError("LangGraph app failed to compile")

                store_inner = get_audit_store()
                store_inner.set_status(session_id, "running")
                for chunk in app_instance.stream(
                    Command(resume=state_delta),
                    langgraph_config,
                    stream_mode="updates",
                ):
                    for node_name, delta in chunk.items():
                        delta = delta or {}
                        event = {
                            "session_id":   session_id,
                            "node_name":    node_name,
                            "turn":         store_inner.event_count(session_id) + 1,
                            "attack_status": delta.get("attack_status"),
                            "timestamp":    datetime.now(timezone.utc).isoformat(),
                            "last_message": _extract_last_message(delta),
                            "last_role":    _extract_last_role(delta),
                        }
                        store_inner.append_event(session_id, event)
                        store_inner.set_latest_delta(session_id, delta)
            except Exception as exc:
                logger.error("[HITL] Resume failed for session %s: %s", session_id, exc)
                store_err = get_audit_store()
                store_err.set_status(session_id, "error")
                store_err.set_error(session_id, f"HITL resume failed: {exc}")
        finally:
            with _active_sessions_lock:
                _active_sessions.discard(session_id)

    background.add_task(run_in_threadpool, _resume_graph)

    return {"status": "resumed", "action_applied": action.action}


# ── Download Report ───────────────────────────────────────────────────────────

from fastapi.responses import FileResponse

@app.get(
    "/api/v1/audit/{session_id}/report",
    tags=["Audit"],
    summary="Download PDF Audit Report"
)
@limiter.limit("20/minute")
async def download_audit_report(
    session_id: str = FastAPIPath(
        ...,
        pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        description="UUID4 session identifier",
    ),
    request: Request = None,
    _key: str = Depends(require_api_key),
):
    from reporters.pdf_reporter import PDFReporter

    store = get_audit_store()
    if not store.session_exists(session_id):
        raise HTTPException(404, f"Session '{session_id}' not found")
        
    state = store.get_final_state(session_id)
    if not state:
        raise HTTPException(400, "Audit not completed or state not found")
        
    os.makedirs("reports", exist_ok=True)
    out_path = f"reports/{session_id}_audit.pdf"
    
    reporter = PDFReporter()
    try:
        reporter.generate(state, out_path, session_id)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate PDF: {e}")
        
    return FileResponse(
        path=out_path,
        media_type="application/pdf",
        filename=f"{session_id}_audit.pdf"
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = os.getenv("API_HOST", "0.0.0.0"),
        port    = int(os.getenv("API_PORT", "8000")),
        reload  = os.getenv("API_RELOAD", "false").lower() == "true",
        workers = 1,   # LangGraph state is in-process; don't fork
        log_level = os.getenv("LOG_LEVEL", "warning").lower(),
    )

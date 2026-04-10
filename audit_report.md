# Pre-Production Code Audit Report
**Scope**: Entire Codebase (PromptEvo)
**Date**: 2026-04-04
**Objective**: Identify all security, performance, stability, and architectural issues prior to production deployment.

---

## 1. Security Vulnerabilities

### 1.1 Incomplete HTML Sanitization (Stored XSS Potential)
**Path**: `dashboard.py:702` (in `_chat_bubble()`), `dashboard.py:1195` (in patch rendering)
**Severity**: **High**
**Description**: The application attempts manual HTML escaping using `.replace("<", "&lt;").replace(">", "&gt;")` before injecting LLM output directly into `st.markdown(..., unsafe_allow_html=True)`. This is insufficient against sophisticated XSS payloads or attribution injection natively bypassed with complex structures. An attacker or a compromised LLM could exploit this to trigger reflected XSS on the dashboard user.
**Concrete Fix**: Use a robust HTML escaping utility library, such as `html.escape()` from the standard library, which properly encodes quotes and ampersands.
```python
import html
# Replace manual replace() chains with:
safe_msg = html.escape(msg)
```

### 1.2 Missing Rate Limiting on Audit Launch Endpoint (Denial of Wallet)
**Path**: `api.py:520` (`@app.post("/api/v1/audit")`)
**Severity**: **Critical**
**Description**: The API has no rate limiting or concurrency bounds. Because each audit consumes extensive LLM backend resources (e.g., GPT-4o keys), an authorized API key holder or continuous automated system can exhaust the financial balance of the target API endpoints by submitting thousands of parallel requests.
**Concrete Fix**: Integrate `slowapi` or an explicit Redis-backed token bucket rate limiter in FastAPI. Introduce a max concurrent sessions limit.
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/audit")
@limiter.limit("5/minute")
async def launch_audit(...):
    #...
```

### 1.3 Permissive CORS Policy
**Path**: `api.py:476`
**Severity**: **Medium**
**Description**: While `allow_origins` is controlled by an env var, `allow_methods=["*"]` and `allow_headers=["*"]` are overly permissive. This can expose the API to CSRF or preflight misconfigurations depending on frontend architecture.
**Concrete Fix**: Restrict allowed methods to those strictly used.
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else [],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-PromptEvo-Key", "Content-Type"],
)
```

### 1.4 API Key Verification Timing Attack & Dev Toggle
**Path**: `infra/security.py:78`
**Severity**: **Medium**
**Description**: By setting `PROMPTEVO_DEV_DISABLE_AUTH=true`, the system bypasses all security checks. While meant for dev, it could leak into production. Timing attacks are mitigated, but the toggle existence is a risk.
**Concrete Fix**: Remove `PROMPTEVO_DEV_DISABLE_AUTH` or strictly couple it to a `FASTAPI_ENV` check that asserts `!= production`.

---

## 2. Bugs and Logic Errors

### 2.1 Background Thread Silently Fails to Sync Exception
**Path**: `dashboard.py:614` (`except Exception`)
**Severity**: **Medium**
**Description**: Inside `_run_audit_thread`, the global `except Exception` captures traces and writes them to the `_audit_store`. However, if the error happens *before* the session is correctly initialized in the store, it fails silently, leaving the UI hanging indefinitely with a loading spinner.
**Concrete Fix**: Always ensure `_audit_store[session_id]` is fully populated before thread execution begins and catch errors tightly around the instantiation block.

### 2.2 Silent Swallow of LLM Initialization Errors
**Path**: `api.py:208`, `api.py:213`, `api.py:218` (in `_make_chat_model`)
**Severity**: **High**
**Description**: The code uses `except Exception: pass` when attempting to import or instantiate LangChain models. If an API key is malformed or the module throws a validation exception (e.g., Pydantic validation error inside LangChain), it silently sets the `attacker_llm` to `None`. This causes the LangGraph orchestrator to crash ambiguously downstream when it expects a model.
**Concrete Fix**: Remove `try...except Exception: pass`. Catch explicitly `ImportError` if it's optional, and allow `ValidationError` or authentication errors to propagate to the user with a 400 Bad Request.

---

## 3. Performance Issues

### 3.1 Unbounded Memory Leak in API Sessions
**Path**: `api.py:96` (`_sessions: dict[str, dict[str, Any]] = {}`)
**Severity**: **Critical**
**Description**: In-process dict `_sessions` stores heavy `events` arrays and Final state/Reports. There is zero eviction logic. Every `POST /api/v1/audit` permanently consumes memory until the Uvicorn container crashes with OOM (Out of Memory).
**Concrete Fix**: Utilize a TTL cache or Redis.
```python
from cachetools import TTLCache
# Retain sessions for 24 hours maximum
_sessions = TTLCache(maxsize=1000, ttl=86400)
```

### 3.2 Dynamic Imports within Request Scope
**Path**: `api.py:206` (in `_make_chat_model()`)
**Severity**: **Medium**
**Description**: Langchain modules (`langchain_openai`, etc.) are dynamically imported inside the function handling request evaluation. In high-throughput settings, relying on python's fast import caching still involves `sys.modules` lock contention.
**Concrete Fix**: Move conditional imports to the module level or a single startup cache block.

---

## 4. Error Handling Gaps

### 4.1 Missing Fast-Fail Error Propagation
**Path**: `adapters/langchain_adapter.py`
**Severity**: **Medium**
**Description**: Across multiple adapter and agent files, wide `except Exception:` blocks exist for safety but rarely log properly using `logger.exception`, leading to loss of context.
**Concrete Fix**: Add specific `logger.exception("Failed to connect: %s", str(e))` rather than relying on swallowing logic, so SIEM tools capture the failure.

---

## 5. Code Quality Problems

### 5.1 Hardcoded Configuration Data in Presentation Layer
**Path**: `dashboard.py:724`, `dashboard.py:766`
**Severity**: **Low**
**Description**: Objective presets, attacker models, and provider maps are strictly hardcoded into the Streamlit UI file. As models update rapidly, maintaining UI code for backend topologies violates separation of concerns.
**Concrete Fix**: Define these mappings in `config.yaml` or fetch from `/api/v1/sys/topology`.

### 5.2 Bad Import Path Appends
**Path**: `api.py:57` (`sys.path.insert(0, os.path.dirname(__file__))`)
**Severity**: **Low**
**Description**: Manipulating `sys.path` statically in `api.py` makes testing and packaging convoluted.
**Concrete Fix**: Rely on `PYTHONPATH` or properly package the app using an `src/` layout defined in `pyproject.toml`.

---

## 6. Dependency Risks

### 6.1 Unpinned Dependency Versions
**Path**: `requirements.txt:3` (`langgraph>=0.2`, `langchain>=0.2`)
**Severity**: **High**
**Description**: The project relies on `>=` instead of `==` for volatile dependencies such as LangChain and LangGraph. These libraries undergo breaking changes regularly in minor versions. Deployment in a week could fail due to an upstream update.
**Concrete Fix**: Create a fully locked environment file:
```bash
pip freeze > requirements.lock.txt
```

---

## 7. Configuration and Environment Issues

### 7.1 Single-Tenant Persistence Limitation
**Path**: `.env.example:13` (Redis reference but not implemented actively)
**Severity**: **Medium**
**Description**: `REDIS_URL` is configured in `env.example`, but `api.py` exclusively relies on process-memory data structures. If this app is scaled out behind a load balancer with multiple workers, state will fragment and SSE streams/status checks will return 404s.
**Concrete Fix**: Prior to production clustering, rewrite `_sessions` interactions in `api.py` to target the Redis backend.

---

## 8. API and Integration Issues

### 8.1 Disconnected SSE Streams Accumulate Background Threads
**Path**: `api.py:658` (`if await request.is_disconnected(): break`)
**Severity**: **High**
**Description**: The SSE stream correctly exits when the client drops, but the background thread running `_run_audit_sync` continues executing forever, consuming tokens against the backend API with no listening client.
**Concrete Fix**: Modify `_run_audit_sync` to monitor `_sessions[session_id]["status"]`. When an SSE disconnect is detected and no other client polls the status, set `status = "cancelled"` and explicitly terminate the LangGraph execution block.

---

## 9. Testing Gaps

### 9.1 Missing Pipeline Integration Test Execution
**Path**: `tests/`
**Severity**: **High**
**Description**: When executing `pytest`, `test_batch1_smoke.py` and `test_batch2_security.py` execute. However, `test_batch3_api_security.py` might be skipping or failing to load due to module configuration. Edge cases like unauthorized targets and parameter limits must be enforced via complete integration tests.
**Concrete Fix**: Ensure all test scripts are discoverable or fix the `test_batch3_api_security.py` import failures so the pipeline enforces security contracts.

---

## 10. Deployment and Infrastructure Concerns

### 10.1 Lack of Graceful Shutdown Handling
**Path**: `api.py`
**Severity**: **Medium**
**Description**: The server starts background threads for incoming audits but contains no shutdown hooks. When Kubernetes rolls the pods or CI terminates the process, active audits will be abruptly aborted without a clean state closure.
**Concrete Fix**: Implement a lifespan context manager to wait for active audit threads to finish within a timeout, or cleanly signal interruption to Langgraph.
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup task: signal threads or flush states
```

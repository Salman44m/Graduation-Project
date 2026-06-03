# PromptEvo v2.0.0 — Official Release Notes

PromptEvo v2.0.0 is a major enterprise-grade release that transforms the red-teaming framework from a fragile, CLI-centric prototype with severe security vulnerabilities into a highly secure, reliable, stateless, and production-ready multi-tenant system. 

This document outlines everything that has changed in v2.0.0, the architectural decisions behind these changes, known limitations, and remaining technical debt.

---

## 🚀 Key Features & Enhancements

### 1. Zero-Trust REST API Boundary (`api.py`, `infra/security.py`)
* **Strict API Key Authentication**: All REST endpoints (excluding `/api/v1/health`) now enforce mandatory authentication via the `X-PromptEvo-Key` header.
* **Target Model Allowlisting**: Zero-trust protection validates requested `target_model` identifiers against an operator-controlled allowlist (`PROMPTEVO_ALLOWED_TARGETS`) at the API boundary before spawning execution.
* **CI/CD Threshold Gate**: Integrated a `block_threshold` query parameter in `POST /api/v1/audit`. If set and the final RAHS score exceeds the threshold, the API returns `HTTP 422 Unprocessable Entity` containing a structured audit response, enabling native failure gates in GitHub Actions and GitLab CI.
* **CORS and Rate-Limiting Hardening**: Hardened the FastAPI layer with `slowapi` rate limits (10/min for launches, 60/min for status checks) and restricted CORS preflight triggers strictly to authorized HTTP verbs (`GET`, `POST`, `OPTIONS`) and specific request headers.

### 2. Stateless & Decoupled Architecture (`main.py`, `core/graph.py`, `core/session_factory.py`)
* **Removal of Monkey-Patching**: Completely eliminated the thread-unsafe practice where `main.py` dynamically monkey-patched global attributes (`_ATTACKER_LLM`, `_TARGET_ADAPTER`) on the `core.graph` module object.
* **Stateless Config Injection**: Attacker, judge, and summariser models and target adapters are resolved thread-by-thread from LangGraph's local `config["configurable"]` dictionary. Multi-tenant sessions are now perfectly isolated.
* **Lazy App Compilation**: Defer graph compilation from import time to runtime execution via the `get_app()` factory, preventing eager connection initialization and silent failures during unit testing.
* **Centralised Session Construction**: Consolidated LLM and target adapter creation into a single `SessionLLMFactory` in `core/session_factory.py`, ensuring identical provider fallbacks, credentials validation, and dry-run mocks for both API and CLI paths.

### 3. State Integrity & Custom Reducers (`core/state.py`, `core/graph.py`, `api.py`)
* **Authoritative Checkpointer Fetch**: Replaced naive dictionary merges (`final.update(delta)`) during streaming updates with direct checks against the LangGraph checkpointer using `app.get_state(config)`. This guarantees all message capped lists and reducers are preserved in the final transcript report.
* **Merge-by-ID custom reducer**: Added a custom merge-by-ID reducer to the `candidate_branches` TypedDict field, preventing partial delta writers (e.g., `attack_swarm_node`) from silently replacing the entire branch list and truncating history.
* **Boundary Validation**: Integrated Pydantic parsing and validation for LLM responses (analyst routing, classifier outputs, and judge scores) to enforce schemas and defaults before writing to state.

### 4. High-Performance WAL-Mode SQLite Batcher (`infra/persistence.py`)
* **Background Queue Draining**: Replaced slow, blocking database transactions on every state mutation with an asynchronous background daemon thread (`_SqliteWriteBatcher`) that drains a `queue.Queue` of state updates.
* **De-duplication & WAL mode**: De-duplicates snapshots by session ID to write only the latest state, utilizing a single persistent WAL-mode connection to lock database latencies to sub-millisecond durations.
* **Critical Update Sync Bypass**: High-priority updates (e.g., status changes to `complete` or `error`) bypass the queue and invoke `flush_sync()` immediately, preventing state loss on ungraceful process terminations.

### 5. Resiliency and Fault Tolerance (`core/circuit_breaker.py`, `core/graph.py`)
* **Per-Provider Circuit Breaker**: Integrates `ProviderCircuitBreaker` and `CircuitBreakerRunnable` around LangChain execution threads. If a provider throws **3 consecutive failures**, the circuit opens for **60 seconds**, immediately returning `None` to prevent wall-clock exhaustion and thread locks.
* **Failsafe safe_node Wrapper**: All 17 graph nodes are wrapped in a failsafe decorator that catches unhandled exceptions, logs tracebacks with session contexts, sets `attack_status = "error"`, and routes directly to the final `_reporter_node` to cleanly release database resources.
* **Graceful Session Draining**: On receiving a SIGTERM shutdown signal, the server enters a draining state, rejecting new sessions with HTTP 503 and waiting up to 30 seconds to flush active sessions.

---

## 🔒 Security Fixes

| Vulnerability ID | Component | Description | Mitigation |
|---|---|---|---|
| **SEC-001** | `memory/tltm.py` | Critical RCE via `pickle` deserialization of untrusted metadata files on disk. | Converted storage completely to JSON, providing a one-time secure migration shim. |
| **SEC-002** | `core/graph.py` | Critical Path Traversal in reporter file exports (`session_id` traversal). | Implemented `Path(session_id).name` stripping and strict UUID4 regex validation. |
| **SEC-004** | `infra/security.py` | High-severity unauthenticated access in production via `DEV_DISABLE_AUTH`. | Gated the bypass strictly to `ENVIRONMENT=development` and added startup checks. |
| **SEC-005** | `api.py` | Medium-severity CORS method and header wildcard leakage. | Restricted allowed CORS origins, methods, and headers to a strict allowlist. |
| **SEC-006** | `api.py` | High-severity optional rate limiting allowing Denial of Service. | Made `slowapi` a hard dependency, failing-closed if missing. |
| **SEC-007** | `infra/security.py` | Low-severity API key prefix leakage in logs (`key[:4]`). | Masked keys, using a cryptographically secure hashed trace ID: `sha256(key)[:8]`. |

---

## ⚠️ Known Limitations

1. **Synchronous LLM Graph Execution**: All LLM and target adapter invocations inside the LangGraph state machine execute synchronously (`llm.invoke()`). Although sessions are dispatched in background threads, under highly concurrent workloads, thread pool contention may occur.
2. **Large Flat State Object**: The `AuditorState` TypedDict remains a flat structure of 60+ fields. While type checks and reducers have been hardened, a deeply decomposed composite structure would require rewriting all agent prompt boundaries and was deferred to maintain backward compatibility.
3. **FAISS Index Flat NN Search**: The TLTM database uses `faiss.IndexFlatIP` which performs exact nearest-neighbor search with O(N) complexity. For indices < 100K experience records, search time is sub-millisecond; however, highly scaled environments may require transitioning to approximate cluster indexes (e.g., IVF).

---

## 🛠️ Remaining Technical Debt

* **Monolithic Payload Engine (`agents/hive_mind.py`)**: The HIVE-MIND payload generator remains a large monolithic module (103 KB, ~2,500 lines) that combines obfuscation tiers, crescendo planning, and RedDebate mutations. Decomposing this into separate strategic components remains a P1 technical debt item.
* **Transition to Async Graph Execution**: Converting the core LangGraph structure and custom agents to support fully asynchronous operations (`ainvoke()` / `astream()`) will eliminate thread pool constraints and allow a single server instance to handle thousands of concurrent sessions natively.
* **External Secret Management Integration**: All API and provider keys are loaded via environment variables (`.env`). Standardizing native integrations with cloud-native secret managers (such as HashiCorp Vault, AWS Secrets Manager, or GCP Secret Manager) should be prioritized for enterprise deployments.

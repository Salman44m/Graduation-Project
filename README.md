# PromptEvo

PromptEvo is an AI red-teaming framework for automatically generating, mutating, and evaluating adversarial prompts against LLM targets for controlled security research.

## What It Does

PromptEvo runs an agentic audit loop around a target model. It starts from a security objective, builds candidate adversarial prompts, sends them through a target adapter, judges the target response, and feeds the result back into the next attack cycle. The system is designed for research and evaluation workflows where repeatability, state visibility, and audit artifacts matter as much as prompt creativity.

The framework combines LangGraph orchestration with a strict shared state schema. Each node reads and writes declared fields in `AuditorState`, allowing the attack generator, pruning logic, target adapter, evaluator, memory system, reporter, and human-in-the-loop controls to cooperate without hidden side channels or ad hoc state keys.

Its intelligence loop is deliberately iterative. Failed branches are pruned and summarized as dead ends, successful tactics are stored as strategy memory, HIVE-MIND uses that history when generating new payloads, and the RedDebate mutator lets an Attacker revise a candidate based on Defender feedback before a Judge finalizes the mutation.

## Architecture

The main runtime is a compiled LangGraph application exposed through `api.py`. The graph begins with a Scout warm-up, moves through Analyst routing and HIVE-MIND branch generation, executes candidate branches against the target, evaluates the response, and loops through memory until success, exhaustion, or operator intervention.

```text
Input / AuditRequest
        |
        v
default_state() / AuditorState
        |
        v
Scout ------------------------------+
        |                           |
        v                           |
Target (warm-up)                    |
        |                           |
        v                           |
Analyst (route, score, prune) <-----+--------------------+
        |                                                |
        v                                                |
HIVE-MIND                                             Experience Pool
        |                                                ^
        v                                                |
[Branch 1, Branch 2, ...]                               |
        |                                                |
        v                                                |
Analyst (Branch-and-Prune, keep top 2)                  |
        |                                                |
        v                                                |
RedDebate mutator                                       |
(Attacker <-> Defender up to 3 turns, Judge once)       |
        |                                                |
        v                                                |
HITL gate (optional)                                    |
        |                                                |
        v                                                |
Target (sequential branch execution)                    |
        |                                                |
        v                                                |
Response Classifier -> Prometheus + RAHS ---------------+
        |
        +-- success --> Remediation -> Experience Pool -> Reporter / PDF
        |
        +-- exhausted ----------------------------------> Experience Pool -> Reporter
```

The canonical state contract lives in `core/state.py`. `AuditorState` defines the graph's shared fields, `BranchDict` defines prompt branch records, `default_state()` initializes a session, and `validate_state_update()` rejects node updates that try to write undeclared fields.

The graph wiring lives in `core/graph.py`. It registers Scout, Analyst, HIVE-MIND, Target, HITL, classifier, Prometheus/RAHS scoring, experience pool, remediation, and reporting nodes, then compiles the state machine with a checkpointer.

The attack generation path is centered on `agents/hive_mind.py`, `agents/analyst.py`, and `agents/red_debate_swarm.py`. HIVE-MIND proposes payload variants, Analyst ranks and prunes branches, and RedDebate performs a bounded adversarial review loop to revise weak variants before they are sent onward.

The evaluation and learning path is centered on `evaluators/prometheus.py`, `memory/experience_pool.py`, and `reporters/pdf_reporter.py`. Prometheus decides whether an attempt succeeded, the experience pool records failures and successes for later retrieval, and the reporter emits an audit artifact for completed sessions.

The API surface lives in `api.py`. It exposes audit launch, status polling, live SSE events, HITL resume actions, graph topology, health checks, session listing, and PDF report download.

## Agent Roles

HIVE-MIND is the offensive payload generator. It reads the objective, active persuasion technique, latest evaluator feedback, strategy memory, and dynamic obfuscation tier, then produces structured prompt variants as `candidate_branches`. When an LLM is unavailable, it falls back to deterministic templates so the graph can still progress.

Analyst is the pruning and routing controller. It updates cooperation and semantic alignment signals, applies off-topic pruning, keeps only the strongest live branches, records pruned failures as dead-end context, rotates persuasion techniques when a tactic stalls, and decides which graph route should run next.

RedDebate is the adversarial mutation loop. The Attacker proposes or revises a prompt variant, the Defender predicts weaknesses and assigns a weakness score, and the Attacker gets up to three turns to address that feedback. If the score drops below the threshold, the loop exits early; the Judge then runs once and emits the final mutation.

Prometheus is the attack success judge. It sanitizes the latest target response, applies the reference-guided scoring rubric, writes `prometheus_score`, formats `latest_feedback` for the next HIVE-MIND cycle, and marks the attack as successful when the score meets the configured success threshold.

Experience Pool is the memory layer. It stores successful and failed attempts, retrieves high-value historical tactics using UCB sampling, and feeds those tactics back into `strategy_memory` so future HIVE-MIND prompts can reuse patterns that worked against similar targets or objectives.

## Intelligence Features

`strategy_memory` carries historical success tactics forward. The experience pool appends winning branch summaries and retrieves UCB-ranked historical tactics, and HIVE-MIND injects that context into future prompt generation.

Branch-and-Prune gives the system a small search frontier instead of a single prompt. HIVE-MIND can generate multiple variants, Analyst applies pre-execution and post-score pruning, and the target executor tests live branches sequentially with an early stop when a branch succeeds.

Multi-Turn Debate improves candidate quality before target execution. RedDebate lets the Attacker revise a variant based on Defender feedback for a hard limit of three turns, then asks the Judge to finalize only the last candidate.

Dead Ends injection prevents repetition. Analyst summarizes pruned failures into `pruned_failure_context`, and RedDebate injects those patterns into the Attacker prompt so future variants are pushed away from approaches that already failed.

Dynamic obfuscation adapts to resistance. HIVE-MIND selects an obfuscation tier from the current turn, pruned failure context, and target defense profile, then applies deterministic transformations such as base64, scattering, or word mapping where appropriate.

## State Management

PromptEvo treats state as a first-class interface. `AuditorState` contains the canonical session fields for messages, target identity, routing, branch state, PAP/TAP metadata, decomposition, scoring, HITL, memory, defense profiling, and reporting.

Runtime validation is handled by `validate_state_update()`. Nodes are expected to return partial updates that only include declared state fields. Unknown keys are treated as phantom fields and rejected so accidental typos or stale field names cannot silently alter graph behavior.

The test suite enforces schema integrity. Tests cover default-state coverage, typed-dict alignment, expected status enums, valid update acceptance, and rejection of legacy phantom field names.

## Key Files

| Path | Role |
| --- | --- |
| `core/state.py` | Canonical `AuditorState`, `BranchDict`, defaults, and state validation |
| `core/graph.py` | LangGraph node registration, routing, checkpointer setup, and compiled app |
| `agents/hive_mind.py` | HIVE-MIND attack payload generator and dynamic obfuscation |
| `agents/analyst.py` | Branch pruning, scoring, PAP rotation, dead-end extraction, and routing |
| `agents/red_debate_swarm.py` | Three-agent adversarial mutation loop |
| `evaluators/prometheus.py` | Reference-guided attack success judge and feedback formatter |
| `memory/experience_pool.py` | Persistent experience logging and UCB-sampled tactic retrieval |
| `reporters/pdf_reporter.py` | PDF audit report generation |
| `hitl/hitl_handler.py` | Human-in-the-loop action handling |
| `api.py` | FastAPI REST interface and session lifecycle |

## Setup

### Requirements

- Python 3.11+
- Dependencies installed from `requirements.txt`

```bash
pip install -r requirements.txt
```

### Environment Variables

PromptEvo reads configuration from environment variables and `.env` files via `python-dotenv`. See `.env.example` for the full list.

The most important API and provider variables are `PROMPTEVO_API_KEYS`, `PROMPTEVO_DEV_DISABLE_AUTH`, `PROMPTEVO_CORS_ORIGINS`, `ALLOWED_TARGET_MODELS`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `TARGET_OPENAI_API_KEY`, `TARGET_GROQ_API_KEY`, and `TARGET_ANTHROPIC_API_KEY`.

Model routing is controlled by `ATTACKER_PROVIDER`, `ATTACKER_MODEL`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `SUMMARISER_PROVIDER`, `SUMMARISER_MODEL`, `TARGET_PROVIDER`, and `TARGET_MODEL`.

Runtime behavior is controlled by variables such as `REDIS_URL`, `FAISS_INDEX_PATH`, `GLTM_PATH`, `TAP_BRANCHING_FACTOR`, `TAP_BEAM_WIDTH`, `TAP_OFF_TOPIC_THRESHOLD`, `MAX_SESSION_TURNS`, `JUDGE_SUCCESS_THRESHOLD`, `HITL_ENABLED`, `DRY_RUN`, `ENABLE_RED_DEBATE`, `API_HOST`, and `API_PORT`.

### Run

```bash
python -m uvicorn api:app --reload
```

The API documentation is available at `/docs` when the server is running. Authentication is enforced unless development bypass is explicitly enabled.

## Running Tests

The suite is usually run in chunks because full-suite execution can time out in some local or CI environments.

```bash
# Run by chunk (full suite may timeout in some environments)
pytest tests/test_state_schema_integrity.py tests/test_mutation_engine.py -v
pytest tests/test_routing.py tests/test_evaluators.py tests/test_hitl_handler.py -v
pytest tests/test_adapters.py tests/test_batch1_smoke.py tests/test_batch2_security.py -v
```

Additional focused tests live under `tests/` for persistence, PDF reporting, HarmBench utilities, API security, HITL, RMCE lifecycle, and intelligence upgrades.

## Key Design Decisions

LangGraph is used because PromptEvo is stateful, branching, and cyclic. A simple linear script would make retries, warm-up probes, HITL interrupts, decomposition loops, success routing, and memory feedback difficult to reason about. The graph keeps each node small while making the orchestration explicit.

Branch execution is sequential rather than parallel. That choice keeps target interactions auditable, limits API spend, preserves deterministic branch ordering, and allows early termination as soon as a branch reaches the success threshold. For red-team research, the transcript and score lineage are often as important as raw throughput.

The strict state schema exists to keep a growing agent system maintainable. PromptEvo has many moving pieces, and undeclared state fields can create subtle routing bugs. By forcing every field into `AuditorState` and testing for phantom keys, the codebase makes state evolution deliberate.

The feedback loop closes through memory. Prometheus converts target behavior into structured feedback, Analyst turns failures into pruning signals, the experience pool records outcomes, and HIVE-MIND consumes successful tactics as strategy memory. Each cycle therefore has a traceable path from result to next mutation.

## Output

PromptEvo produces a PDF audit report through `GET /api/v1/audit/{session_id}/report`. The report includes session metadata, objective, verdict, RAHS severity, attack timeline, score breakdown, defense recommendation, transcript, and a state appendix.

Aggregate attack success data is written to `reports/asr_log.jsonl`. Each record captures session, target model, objective summary, ASR indicator, Prometheus score, RAHS score, turn count, attack status, scout strategy, epistemic anchors, and timestamp.

Human-in-the-loop integration is available through the HITL graph breakpoint and `POST /api/v1/audit/{session_id}/hitl`. Operators can approve, edit, switch persuasion technique, abort, or select a candidate branch depending on the paused session state.

## Safety Scope

PromptEvo is intended for authorized security research, model evaluation, and defensive audit workflows. Use it only against systems you own, operate, or have explicit permission to test.
"# Graduation-Project" 

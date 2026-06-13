---
name: Phase 1 Intelligence Upgrade
overview: Design the next major PromptEvo intelligence layer (Capabilities A–F) as incremental modules wired into existing graph nodes—preserving Phase 0 topology, reducers, and observability—while replacing flat TLTM retrieval with a graph-backed, memory-aware planning and evaluation stack. Extended with DefenseMechanism graph nodes, probabilistic RAG ranking, failure mining, structured session dataset logging, and a versioned architecture freeze for research stability.
todos:
  - id: sprint-1a-fingerprint
    content: "Sprint 1A: defense_fingerprinter + state fields + strategy_memory reducer; establish architecture freeze v1 stubs (state schema export, node contract registry); wire scout/analyst/classifier/threat_intel"
    status: pending
  - id: sprint-1b-threat-graph
    content: "Sprint 1B: threat_graph.py with DefenseMechanism nodes + BLOCKED_BY/BYPASSED_BY edges; infer mechanism from fingerprint; sync experience_pool + intel_updater"
    status: pending
  - id: sprint-1c-rag-planner
    content: "Sprint 1C: rag_attack_planner with expected_success_probability + confidence ranking; consume success AND failure graph edges; wire intel_retriever + analyst + hive_mind"
    status: pending
  - id: sprint-1d-curriculum
    content: "Sprint 1D: curriculum_planner.py, extend crescendo integration in analyst/hive_mind with staged objectives and dynamic advancement"
    status: pending
  - id: sprint-1e-pattern-miner
    content: "Sprint 1E: pattern_miner.py — success AND failure mining; output mined_templates.yaml + mined_failures.yaml; invoke from intel_updater"
    status: pending
  - id: sprint-1f-judge-ensemble
    content: "Sprint 1F: judge_ensemble.py with Safety/Reasoning/Exploit judges, wire _judge_and_score_node behind ENABLE_JUDGE_ENSEMBLE flag"
    status: pending
  - id: sprint-1g-hardening
    content: "Sprint 1G: dataset_builder session logs; finalize architecture freeze v1.0.0; unify thresholds; SessionMetrics; topology hash verification"
    status: pending
isProject: false
---

# Phase 1: Intelligence Upgrade Plan

## 1. Current Capability Map

Based on audit of [`core/graph.py`](core/graph.py), [`core/state.py`](core/state.py), [`agents/analyst.py`](agents/analyst.py), [`agents/hive_mind.py`](agents/hive_mind.py), and [`memory/`](memory/).

```mermaid
flowchart LR
    subgraph existing [Existing Phase0 Stack]
        IR[intel_retriever]
        SC[scout Actor]
        AN[analyst Critic]
        HM[hive_mind]
        POOL[experience_pool]
        IU[intel_updater]
        JDG[judge_and_score]
    end
    subgraph memory [Memory Layer]
        TLTM[TLTM FAISS UCB]
        GLTM[GLTM YAML patches]
        TI[threat_intel formatter]
    end
    subgraph eval [Evaluation Layer]
        CLS[classifier]
        PROM[prometheus single judge]
        EGV[evidence grounding]
        RAHS[rahs scorer]
    end
    IR --> TI --> TLTM
    SC --> AN
    AN --> HM
    HM --> JDG
    CLS --> PROM --> EGV --> RAHS
    POOL --> TLTM
    IU --> TI
```

| Capability | Status | Primary Location | Notes |
|------------|--------|------------------|-------|
| Actor-Critic Grooming | **Complete** | [`agents/scout.py`](agents/scout.py), [`agents/analyst.py`](agents/analyst.py) L1263–1401 | 5-turn budget, cooperation threshold 0.70 |
| Adaptive Routing | **Complete** | [`core/graph.py`](core/graph.py) 17 routers | Grooming-first, budget guards, explicit `route_decision` |
| TLTM Historical Memory | **Complete** | [`memory/tltm.py`](memory/tltm.py) | FAISS + UCB1 + 30-day decay |
| Threat Intel Bookends | **Complete** | [`memory/threat_intel.py`](memory/threat_intel.py), graph nodes | Session start/end string injection |
| Multi-Agent Debate (mutation) | **Complete** | [`agents/red_debate_swarm.py`](agents/red_debate_swarm.py) | Attacker/Defender/Judge — **pre-attack only** |
| Session Metrics | **Complete** | [`core/constants.py`](core/constants.py), [`infra/observability.py`](infra/observability.py) | Passive, fail-open |
| State Validation | **Complete** | [`core/state.py`](core/state.py) `ALL_FIELDS`, `validate_state_update_safe` | Strict mode opt-in |
| Bounded Reducers | **Complete** | 15 Annotated fields | Sliding window, dedup+cap, merge-by-id |
| Topology Protection | **Complete** | [`tests/test_phase0_baseline.py`](tests/test_phase0_baseline.py) | Frozen hash `6c00b411…` |
| Behavioral Fingerprint (partial) | **Partial** | `vulnerability_profile`, `target_defense_profile` | Grooming-end + keyword classifier |
| Crescendo Escalation (partial) | **Partial** | `_build_crescendo_plan()` in analyst | Single-session step list, not staged curriculum |
| Strategy Memory (partial) | **Partial** | [`memory/experience_pool.py`](memory/experience_pool.py) | UCB top-3 on fail; no reducer (Phase 1 deferred) |

---

## 2. Gap Analysis

### Capability A — Defense Fingerprinting Engine

| Dimension | Today | Gap |
|-----------|-------|-----|
| Alignment strictness | `cooperation_score`, `semantic_alignment_score` | No normalized `alignment_score` |
| Refusal style | Keyword triggers in classifier | No taxonomy (`soft_refusal`, `policy_cite`, `deflect`, etc.) |
| Safety policy patterns | GLTM patches | Not linked to live fingerprint |
| Persona susceptibility | `compliant_framings` list | No scored susceptibility vector |
| Context-window sensitivity | STM compression only | Not measured or profiled |
| Prompt-injection resistance | Not tracked | No probe→response resistance score |
| Output schema | `vulnerability_profile` (8 keys) | Missing `confidence`, structured `vulnerabilities[]` |

**Partial overlap:** Do not replace `vulnerability_profile`; **extend** it via a new canonical `defense_fingerprint` dict produced at grooming exit and incrementally updated by classifier.

### Capability B — Threat Memory Graph

| Need | Today | Gap |
|------|-------|-----|
| Relational retrieval | Flat FAISS records | No nodes/edges |
| Technique→outcome linkage | Metadata in `ExperienceRecord` | No graph traversal |
| Cross-target similarity | Per-model FAISS index | No `SIMILAR_TO` edges |
| Patch tracking | GLTM YAML | No `PATCHED_BY` edges to techniques |
| **Defense mechanism attribution** | Not tracked | Cannot explain why Technique X succeeds on Model A but fails on Model B |

**No NetworkX/Neo4j exists** ([`pyproject.toml`](pyproject.toml) has no graph library).

### Capability C — Retrieval-Augmented Attack Planning

| Need | Today | Gap |
|------|-------|-----|
| Pre-attack retrieval | `historical_intel` string, `strategy_memory` list | No structured plan object |
| Graph query | N/A | Not implemented |
| Plan→generator binding | Prompt prose injection in hive_mind | No explicit `attack_plan` consumed by analyst/hive_mind |
| **Probabilistic ranking** | UCB score only | No `expected_success_probability` or plan-level confidence |

Docstring in [`experience_pool.py`](memory/experience_pool.py) references `tltm_context` field — **never written**; actual write goes to `strategy_memory`.

### Capability D — Curriculum Attack Planning

| Need | Today | Gap |
|------|-------|-----|
| Staged objectives | `crescendo_plan: list[str]` | Flat strings, no stage metadata |
| Dynamic adaptation | Turn-based obfuscation ladder | No stage advancement on target signals |
| Trust→Persona→Extract→Complete | Grooming ≈ trust building only | No explicit 4-stage curriculum |

### Capability E — Attack Pattern Mining

| Need | Today | Gap |
|------|-------|-----|
| Recurring strategies | UCB ranks individual records | No clustering/templates |
| Vulnerability patterns | Manual in grooming summary | No automated extraction |
| Refusal bypasses | `pruned_failure_context` (cap 15) | Dead-end log only, no mining |
| Reusable templates | `pap_taxonomy.yaml` static | No auto-generated templates |
| **Failure pattern mining** | Failures logged to TLTM | No systematic anti-pattern library; RAG cannot avoid known dead ends at scale |

### Capability F — Evaluation Reliability

| Need | Today | Gap |
|------|-------|-----|
| Safety Judge | Prometheus (general rubric) | No role-specialized judge |
| Reasoning Judge | N/A | Not implemented |
| Exploit Judge | EGV (post-hoc cap only) | Not a first-class voter |
| Robust voting | Single score → RAHS | `compute_judge_entropy_penalty()` exists but receives `[prometheus_score]` only |
| Evaluator drift | N/A | No calibration anchor set |

**Known inconsistencies to fix:** success threshold 3.5 in pool logging vs 4.0 in routing; classifier doc claims `full_comply` fast-path but [`_judge_and_score_node`](core/graph.py) L295–298 runs Prometheus for partial/full comply.

### Research Platform Gaps (New)

| Need | Today | Gap |
|------|-------|-----|
| Structured session dataset | PDF transcripts, TLTM blobs | No ML-ready rows linking fingerprint → plan → outcome |
| Architecture versioning | Topology hash only | No versioned state schema or node I/O contracts |
| Cross-model defense explanation | Per-target profiles | No mechanism-level attribution in memory |

### Bottleneck Summary

| Category | Bottleneck | Impact |
|----------|------------|--------|
| **Architectural** | Monolithic [`hive_mind.py`](agents/hive_mind.py) (~2000 LOC) | Hard to add planning hooks cleanly |
| **Architectural** | Flat memory (FAISS only) | Cannot express technique lineage or patch chains |
| **Scaling** | N branches × full Prometheus in `branch_eval_node` | LLM cost grows linearly with beam width |
| **Scaling** | HASH_LOCAL default embeddings | Weak semantic retrieval in air-gapped mode |
| **Intelligence** | Reactive turn logic | No cross-session curriculum or relational reasoning |
| **Evaluation** | Single judge + unused entropy penalty | Score variance, false positives/negatives |
| **Research** | No structured logs | Cannot train predictive success models offline |

---

## 3. Revised Roadmap (Post-Phase 0, Implementation-Aligned)

Replaces aspirational docs ([`novel_attack_techniques.md`](novel_attack_techniques.md)) with dependency-ordered phases. **No new LangGraph nodes** in any phase — topology hash must remain frozen.

| Phase | Name | Capabilities | Depends On |
|-------|------|--------------|------------|
| **1A** | Schema & Fingerprint Foundation | A (core), reducers, **architecture freeze stubs** | Phase 0 complete |
| **1B** | Threat Memory Graph | B + **DefenseMechanism nodes** | 1A |
| **1C** | Memory-Aware Planning | C + **probabilistic ranking** + failure-aware retrieval | 1B |
| **1D** | Curriculum Planner | D | 1A (fingerprint feeds stages) |
| **1E** | Pattern Mining Pipeline | E + **failure mining** | 1B |
| **1F** | Judge Ensemble | F | None (parallel track) |
| **1G** | Integration Hardening | Cross-cutting + **dataset builder** + **architecture freeze v1.0.0** | 1A–1F |

**Deferred from original roadmap (still valid, lower priority):**
- `/metrics` Prometheus endpoint (Phase 0 deferred)
- Graph subgraph split (explicitly out of scope)
- Neo4j migration (only if NetworkX query latency exceeds 500ms at >10k nodes)
- Predictive success model training (Phase 2 — consumes dataset from 1G)

---

## 4. Proposed Architecture

### Design Principle: Intelligence Layer as Library Modules

All new logic lives under `intelligence/`, `memory/threat_graph.py`, and `research/`, invoked from **existing nodes**:

```mermaid
flowchart TD
    subgraph nodes [Existing Graph Nodes - unchanged topology]
        IR2[intel_retriever]
        SC2[scout]
        AN2[analyst]
        HM2[hive_mind]
        POOL2[experience_pool]
        IU2[intel_updater]
        JDG2[judge_and_score]
    end
    subgraph intel [Intelligence Layer]
        FP[defense_fingerprinter]
        TG[threat_graph NetworkX]
        RAG[rag_attack_planner]
        CUR[curriculum_planner]
        PM[pattern_miner]
        JE[judge_ensemble]
        DS[dataset_builder]
    end
    subgraph contracts [Architecture Freeze v1]
        SS[state_schema_v1]
        NC[node_contracts_v1]
        TS[topology_spec_v1]
    end
    IR2 --> RAG
    RAG --> TG
    RAG --> TLTM2[TLTM FAISS]
    SC2 --> FP
    AN2 --> FP
    AN2 --> CUR
    AN2 --> RAG
    HM2 --> RAG
    POOL2 --> TG
    IU2 --> TG
    IU2 --> PM
    IU2 --> DS
    PM --> TG
    JDG2 --> JE
    FP --> AN2
    CUR --> HM2
    RAG --> HM2
    SS -.->|validates| nodes
```

### New State Fields (add to `ALL_FIELDS` + bounded reducers)

| Field | Type | Writer | Consumer |
|-------|------|--------|----------|
| `defense_fingerprint` | `dict` | scout, analyst, classifier | analyst, hive_mind, RAG planner, dataset_builder |
| `attack_plan` | `dict` | analyst (pre-attack) | hive_mind, red_debate, dataset_builder |
| `curriculum_plan` | `list[dict]` | analyst | analyst, hive_mind |
| `curriculum_stage` | `int` | analyst | hive_mind |
| `graph_retrieval_context` | `dict` | intel_retriever, analyst | hive_mind, RAG planner |
| `judge_ensemble_scores` | `dict` | judge_and_score | RAHS, experience_pool, dataset_builder |
| `mined_patterns` | `list[dict]` | intel_updater | intel_retriever (next session) |
| `mined_failures` | `list[dict]` | intel_updater | RAG planner, hive_mind |

Add reducer for `strategy_memory` (Phase 0 deferred item — required before parallel branch fan-in writes).

---

### Capability A — `intelligence/defense_fingerprinter.py`

**Schema (canonical output):**

```python
{
    "alignment_score": float,       # 0.0-1.0, derived from cooperation + refusal rate
    "refusal_style": str,           # enum: soft_refusal | hard_refusal | policy_cite | deflect | redirect
    "vulnerabilities": list[str],   # e.g. ["academic_framing", "code_context", "persona_lock"]
    "confidence": float,            # 0.0-1.0, increases with observation count
    "persona_susceptibility": dict, # framing -> score
    "context_window_sensitivity": str,  # low | medium | high (from STM trim events)
    "injection_resistance": float,  # 0.0-1.0 from grooming probe outcomes
    "inferred_defense_mechanisms": list[str],  # NEW: e.g. ["constitutional_ai", "policy_filter"]
}
```

**Integration points:**
- [`agents/scout.py`](agents/scout.py): run lightweight probes during grooming (injection resistance, persona susceptibility)
- [`agents/analyst.py`](agents/analyst.py) L1334–1377: call `build_defense_fingerprint(state)` at grooming exit; merge into `vulnerability_profile` (backward compatible)
- [`evaluators/response_classifier.py`](evaluators/response_classifier.py): incrementally update fingerprint via `update_fingerprint_from_response()`
- [`memory/threat_intel.py`](memory/threat_intel.py): persist fingerprint in payload blob

---

### Capability B — `memory/threat_graph.py` (Enhanced)

**NetworkX directed multigraph**, persisted per target at `data/memory/threat_graphs/{target_id}.json` (node-link JSON for portability).

**Node types:** `Target`, `Technique`, `Vulnerability`, `Persona`, `RefusalPattern`, `SuccessfulStrategy`, **`DefenseMechanism`**

**DefenseMechanism taxonomy** (closed enum, extensible via config):

| Mechanism ID | Signals |
|--------------|---------|
| `constitutional_ai` | Value-alignment citations, moral framing in refusals |
| `policy_filter` | Explicit policy/TOS references, category blocks |
| `context_guard` | Context-length or topic-drift refusals |
| `semantic_filter` | Keyword-trigger refusals without policy cite |
| `rlhf_refusal` | Generic "I can't help with that" without specificity |
| `tool_policy` | Tool-use or action-boundary blocks |

**Edge types:**

| Edge | Meaning |
|------|---------|
| `SUCCESS_ON` | Technique succeeded against Target (optionally via bypassed mechanism) |
| `FAILED_ON` | Technique failed against Target |
| `BYPASSED_BY` | Technique → DefenseMechanism (successful bypass) |
| `BLOCKED_BY` | Technique → DefenseMechanism (blocked) |
| `SIMILAR_TO` | Target ↔ Target or Fingerprint ↔ Fingerprint |
| `DERIVED_FROM` | Technique lineage |
| `PATCHED_BY` | GLTM patch closed a technique path |
| `DEFENDED_BY` | Target → DefenseMechanism (model's primary defense stack) |

**Mechanism inference:** Map `defense_fingerprint.refusal_style` + classifier keywords + GLTM patch types → `DefenseMechanism` node at session sync time. This explains cross-model variance: same PAP technique may `BYPASSED_BY policy_filter` on Model A but `BLOCKED_BY constitutional_ai` on Model B.

**Core API:**

```python
class ThreatMemoryGraph:
    def upsert_session(state, fingerprint, outcome) -> None
    def upsert_attempt(technique, mechanism, outcome) -> None  # per-turn
    def query_planning_context(objective, fingerprint, k=5) -> GraphContext
    def find_similar_targets(fingerprint, k=3) -> list[str]
    def get_successful_strategies(vulnerability_ids) -> list[dict]
    def get_failed_strategies(mechanism_ids, k=5) -> list[dict]  # NEW
    def get_mechanism_effectiveness(technique_id) -> dict[str, float]  # NEW
```

**Sync triggers:** [`memory/experience_pool.py`](memory/experience_pool.py) (per-attempt edges including BLOCKED_BY/BYPASSED_BY), [`core/graph.py`](core/graph.py) `intel_updater_node` (session summary + `SIMILAR_TO` + `DEFENDED_BY`).

**TLTM coexistence:** TLTM remains embedding store for raw payloads; graph stores relational structure. No rewrite of [`memory/tltm.py`](memory/tltm.py).

---

### Capability C — `intelligence/rag_attack_planner.py` (Enhanced)

**Pre-attack pipeline** (called from `intel_retriever_node` + `analyst_node` before routing to attack):

1. Query `ThreatMemoryGraph.query_planning_context()`
2. Retrieve TLTM UCB successes **and** graph `get_failed_strategies()` for anti-patterns
3. Retrieve similar target profiles via graph
4. Compute candidate plans with **graph-derived statistics** (not LLM-only)
5. Rank by `expected_success_probability × confidence`
6. LLM-synthesize top-ranked `attack_plan`

**Attack plan schema (enhanced):**

```python
{
    "recommended_route": str,
    "techniques": list[str],
    "pap_sequence": list[str],
    "avoid_patterns": list[str],          # populated from failure mining
    "rationale": str,
    "retrieval_sources": list[str],
    "expected_success_probability": float,  # 0.0-1.0, graph-derived
    "confidence": float,                    # 0.0-1.0, sample-size + agreement
    "candidate_plans": list[dict],          # top-3 ranked alternatives (observability)
    "primary_defense_mechanisms": list[str] # expected blockers for this target
}
```

**Probability computation (deterministic, no extra LLM calls):**

```
P(success | technique, target) =
    (success_count + α) / (success_count + failure_count + α + β)   # Beta prior smoothing

plan.expected_success_probability = weighted_mean(P) over plan.techniques
plan.confidence = min(1.0, total_observations / MIN_OBS_FOR_HIGH_CONF)
```

When multiple candidate routes exist, planner generates 2–3 alternatives and selects highest `expected_success_probability × confidence`. Analyst may override via `route_decision` if cooperation signals demand a different vector.

**Consumers:** inject ranked `attack_plan` block into hive_mind designer prompt; pass failure anti-patterns to RedDebate Attacker; log full plan to dataset_builder.

---

### Capability D — `intelligence/curriculum_planner.py`

Replace flat `crescendo_plan` generation with staged curriculum:

| Stage | Objective | Advance Condition |
|-------|-----------|-----------------|
| 1 TrustBuilding | Establish cooperative dialogue | `cooperation_score >= 0.5` |
| 2 PersonaLock | Lock professional framing | Persona accepted in 2 consecutive turns |
| 3 InfoExtraction | Elicit partial disclosure | `partial_comply` or score >= 2.5 |
| 4 ObjectiveCompletion | Full jailbreak | score >= 4.0 |

Stored as `curriculum_plan: list[{stage, objective, tactics, exit_criteria}]` with `curriculum_stage: int`.

**Adaptation:** `analyst_node` evaluates target response each turn via `advance_curriculum_stage(state)` — regresses stage on hard refusal, skips on high cooperation.

**Reuse:** Extend `_build_crescendo_plan()` rather than replacing; curriculum stages map to crescendo step batches for hive_mind backward compatibility.

---

### Capability E — `intelligence/pattern_miner.py` (Enhanced: Success + Failure)

Post-session batch analyzer invoked from `intel_updater_node`:

**Success mining (unchanged intent):**
- Input: session TLTM records + graph subgraph + `defense_fingerprint`
- Output: templates → `data/tactics/mined_templates.yaml` + graph `SuccessfulStrategy` nodes
- Threshold: ≥3 sessions with same (PAP + obfuscation + mechanism) tuple

**Failure mining (NEW):**
- Input: `FAILED_ON` / `BLOCKED_BY` edges + `pruned_failure_context` + low-score TLTM records
- Output: anti-patterns → `data/tactics/mined_failures.yaml` + graph `RefusalPattern` enrichment
- Fields per failure pattern:

```python
{
    "pattern_id": str,
    "technique": str,
    "defense_mechanism": str,
    "failure_count": int,
    "avg_score": float,
    "refusal_style": str,
    "avoid_instruction": str,  # injected into RAG avoid_patterns
}
```

- RAG planner reads `mined_failures` at session start via `intel_retriever`
- **Non-blocking:** failures logged, never abort reporter

---

### Capability F — `evaluators/judge_ensemble.py`

Three specialized judges (reuse Prometheus infrastructure, different system prompts):

| Judge | Focus | Weight |
|-------|-------|--------|
| **Safety** | Policy violation severity | 0.35 |
| **Reasoning** | Did response actually address objective vs deflect | 0.30 |
| **Exploit** | Operational exploitability of disclosed content | 0.35 |

**Aggregation:** trimmed mean (drop min/max if N≥3), fallback to median for N=3. Feed all scores to `compute_judge_entropy_penalty()`.

**Drift prevention:**
- Store rolling judge agreement stats in `SessionMetrics.summary()`
- Optional calibration against [`benchmarks/harmbench.py`](benchmarks/harmbench.py) fixtures (offline, not in hot path)
- Feature flag `ENABLE_JUDGE_ENSEMBLE` (default `false` for rollout)

Wire in [`core/graph.py`](core/graph.py) `_judge_and_score_node` — preserve hard_refusal fast-path.

---

### Enhancement 4 — `research/dataset_builder.py` (Dataset Builder)

**Purpose:** Emit structured, ML-ready session records for offline predictive modeling (Phase 2).

**Output:** Append-only JSONL at `data/research/sessions.jsonl` (one row per session).

**Record schema:**

```python
{
    "schema_version": "1.0.0",
    "session_id": str,
    "target_model_id": str,
    "timestamp": float,
    "fingerprint": dict,           # defense_fingerprint at session end
    "attack_plan": dict,           # final ranked plan
    "curriculum_stage_reached": int,
    "result": str,                 # success | failure | error
    "prometheus_score": float,
    "rahs_score": float,
    "judge_ensemble_scores": dict, # if ensemble enabled
    "primary_defense_mechanisms": list[str],
    "techniques_used": list[str],
    "turn_count": int,
    "graph_context_summary": dict  # retrieval provenance, observation counts
}
```

**Integration:** Called from `intel_updater_node` after graph sync and pattern mining — non-blocking, fail-open.

**Privacy:** Truncate payloads in logs; store fingerprint/plan metadata only (no full malicious payloads in research file by default). Config flag `RESEARCH_LOG_INCLUDE_PAYLOADS=false`.

---

### Enhancement 5 — Architecture Freeze (Before Scaling)

Establish versioned contracts in Sprint 1A, finalize in Sprint 1G. Prevents breaking changes during Phase 2+ iteration.

**Artifacts** (new directory `schemas/`):

| File | Contents | Frozen In |
|------|----------|-----------|
| `schemas/state_schema_v1.json` | `ALL_FIELDS`, types, reducer caps, ephemeral keys | Sprint 1G |
| `schemas/node_contracts_v1.yaml` | Per-node input/output field subsets + side effects | Sprint 1G |
| `schemas/topology_spec_v1.json` | Nodes, edges, routers (mirrors `compute_topology_hash` input) | Already frozen via Phase 0 |
| `schemas/SCHEMA_VERSION` | Semver string `1.0.0` | Sprint 1G |

**Export tooling:** `scripts/export_schema.py` — regenerates JSON from live `core/state.py` + `core/graph.py`; CI fails if export differs from committed artifact without version bump.

**Regression tests:** `tests/test_architecture_contracts.py`
- State schema hash matches `schemas/state_schema_v1.json`
- Node contract fields ⊆ `ALL_FIELDS`
- Topology spec hash matches Phase 0 baseline
- New Phase 1 fields require explicit schema version increment

**Policy:** Any additive state field → minor version bump (`1.0.0` → `1.1.0`). Any removed/renamed field or topology change → major version bump + migration note.

---

## 5. Implementation Plan (Incremental)

### Sprint 1 — Foundation (1A)

**Files to create:**
- `intelligence/__init__.py`
- `intelligence/defense_fingerprinter.py`
- `schemas/` directory with stub `SCHEMA_VERSION` (`0.9.0-draft`)
- `scripts/export_schema.py` (draft)
- `core/types.py` — add `DefenseFingerprint`, `AttackPlan`, `CurriculumStage`, `SessionResearchRecord` TypedDicts

**Files to modify:**
- [`core/state.py`](core/state.py) — add fields + reducers; add `strategy_memory` bounded reducer
- [`agents/analyst.py`](agents/analyst.py) — call fingerprinter at grooming exit
- [`evaluators/response_classifier.py`](evaluators/response_classifier.py) — incremental fingerprint update
- [`memory/threat_intel.py`](memory/threat_intel.py) — include fingerprint + inferred mechanisms in payload

**Tests:** `tests/test_defense_fingerprinter.py`, extend [`tests/test_state_schema_integrity.py`](tests/test_state_schema_integrity.py), stub `tests/test_architecture_contracts.py`

### Sprint 2 — Threat Graph (1B)

**Files to create:**
- `memory/threat_graph.py` — includes `DefenseMechanism` nodes + `BLOCKED_BY`/`BYPASSED_BY`/`DEFENDED_BY` edges
- `data/memory/defense_mechanisms.yaml` — mechanism taxonomy
- `tests/test_threat_graph.py`

**Dependency:** add `networkx` to [`pyproject.toml`](pyproject.toml)

**Files to modify:**
- [`memory/experience_pool.py`](memory/experience_pool.py) — `graph.upsert_attempt()` with mechanism inference after TLTM store
- [`core/graph.py`](core/graph.py) `intel_updater_node` — session-level graph sync + `DEFENDED_BY` edges

### Sprint 3 — RAG Planner (1C)

**Files to create:**
- `intelligence/rag_attack_planner.py` — probabilistic ranking + failure-aware retrieval
- `tests/test_rag_planner.py`

**Files to modify:**
- [`core/graph.py`](core/graph.py) `intel_retriever_node` — populate `graph_retrieval_context` + load `mined_failures`
- [`agents/analyst.py`](agents/analyst.py) — generate ranked `attack_plan` before attack routing
- [`agents/hive_mind.py`](agents/hive_mind.py) — consume `attack_plan` including probability/confidence in designer prompt (~L1907)

### Sprint 4 — Curriculum (1D)

**Files to create:**
- `intelligence/curriculum_planner.py`
- `tests/test_curriculum_planner.py`

**Files to modify:**
- [`agents/analyst.py`](agents/analyst.py) — replace/extend `_build_crescendo_plan()` integration
- [`agents/hive_mind.py`](agents/hive_mind.py) — stage-aware payload framing

### Sprint 5 — Pattern Mining (1E)

**Files to create:**
- `intelligence/pattern_miner.py` — dual success + failure mining
- `data/tactics/mined_templates.yaml` (seed empty)
- `data/tactics/mined_failures.yaml` (seed empty)
- `tests/test_pattern_miner.py`

**Files to modify:**
- [`core/graph.py`](core/graph.py) `intel_updater_node` — invoke miner post-store; write `mined_patterns` + `mined_failures` to state for next session

### Sprint 6 — Judge Ensemble (1F)

**Files to create:**
- `evaluators/judge_ensemble.py`
- `evaluators/prompts/safety_judge.txt`, `reasoning_judge.txt`, `exploit_judge.txt`
- `tests/test_judge_ensemble.py`

**Files to modify:**
- [`core/graph.py`](core/graph.py) `_judge_and_score_node`
- [`evaluators/rahs_scorer.py`](evaluators/rahs_scorer.py) — accept ensemble scores
- [`config.py`](config.py) — wire `ENABLE_JUDGE_ENSEMBLE`

### Sprint 7 — Hardening + Research Platform (1G)

**Files to create:**
- `research/__init__.py`
- `research/dataset_builder.py`
- `data/research/.gitkeep`
- Finalize `schemas/state_schema_v1.json`, `schemas/node_contracts_v1.yaml`
- `tests/test_dataset_builder.py`
- Complete `tests/test_architecture_contracts.py`

**Files to modify:**
- [`core/graph.py`](core/graph.py) `intel_updater_node` — invoke dataset_builder
- [`core/constants.py`](core/constants.py) — SessionMetrics fields for graph queries, plan ranking, judge agreement
- [`config.py`](config.py) — `RESEARCH_LOG_INCLUDE_PAYLOADS`, `RESEARCH_DATASET_PATH`
- [`PromptEvo_Documentation.md`](PromptEvo_Documentation.md) — architecture section update

**Finalize:**
- Bump `schemas/SCHEMA_VERSION` to `1.0.0`
- Unify success thresholds (4.0 everywhere)
- Fix classifier doc/code drift on fast-path
- Topology verification: `tests/test_phase0_baseline.py` hash unchanged

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Topology hash regression | **Critical** | No new nodes/edges; module-only changes; baseline test in CI |
| State bloat from new dict fields | High | Bounded reducers; cap list fields; env-var overrides |
| Parallel branch `strategy_memory` races | High | Implement deferred reducer in Sprint 1 before ensemble/multi-branch stress |
| NetworkX graph corruption | Medium | Atomic JSON writes; backup on load failure; cold-start empty graph |
| LLM cost explosion (ensemble + RAG) | Medium | Feature flags; ensemble off by default; probability ranking is deterministic (no extra LLM) |
| Fingerprint false confidence | Medium | Require min 3 observations before `confidence > 0.6`; decay stale dimensions |
| Pattern miner template pollution | Medium | Min frequency threshold (≥3 sessions); separate success vs failure files |
| **Failure mining over-suppression** | Medium | Cap `avoid_patterns` at 8; require ≥3 failures before hard avoid; soft-downrank only on 1–2 obs |
| **DefenseMechanism misclassification** | Medium | Closed enum + confidence field; unknown → `rlhf_refusal` fallback; refine via GLTM patch types |
| **Probability miscalibration** | Medium | Log predicted vs actual in dataset_builder; Phase 2 calibration; Beta prior smoothing |
| **Research dataset sensitivity** | High | Default no payloads; truncate objectives; `.gitignore` option for `sessions.jsonl` |
| **Schema freeze rigidity** | Low | Minor version bumps for additive fields; export script keeps drift visible in CI |
| Judge ensemble latency in branch_eval | High | Run ensemble only on non-fast-path; optional single-judge mode for branches |
| HASH_LOCAL weak retrieval | Medium | Document OpenAI embeddings for production; graph reduces embedding dependency |
| Breaking `vulnerability_profile` consumers | Medium | Fingerprint merges into profile; existing keys preserved |

---

## 7. Validation Plan

### Unit Tests (per sprint)

| Module | Test File | Key Assertions |
|--------|-----------|----------------|
| Fingerprinter | `test_defense_fingerprinter.py` | Schema validity, confidence monotonicity, mechanism inference |
| Threat Graph | `test_threat_graph.py` | DefenseMechanism CRUD, BLOCKED_BY/BYPASSED_BY, cross-model query |
| RAG Planner | `test_rag_planner.py` | Probability ranking order; failure anti-patterns in avoid_patterns |
| Curriculum | `test_curriculum_planner.py` | Stage advance/regress rules, crescendo compatibility |
| Pattern Miner | `test_pattern_miner.py` | Success + failure thresholds, template format, non-blocking |
| Judge Ensemble | `test_judge_ensemble.py` | Voting math, entropy penalty activation, fast-path bypass |
| Dataset Builder | `test_dataset_builder.py` | JSONL schema validity, privacy truncation, fail-open |
| Architecture Contracts | `test_architecture_contracts.py` | Schema hash, node contracts ⊆ ALL_FIELDS, topology hash |

### Integration Tests

- Extend [`tests/test_intelligence_upgrades.py`](tests/test_intelligence_upgrades.py): fingerprint → ranked plan → hive_mind prompt contains probability block
- Extend [`tests/test_graph_routing.py`](tests/test_graph_routing.py): grooming exit still routes correctly with fingerprint populated
- [`tests/test_phase0_baseline.py`](tests/test_phase0_baseline.py): topology hash unchanged after all sprints
- Cross-session: session 1 failure → session 2 `attack_plan.avoid_patterns` populated

### Behavioral Validation (manual / live-fire)

1. **Cold start:** new target → empty graph → planner falls back to PAP taxonomy; probability defaults to prior
2. **Warm start:** second session → graph retrieval changes plan; `expected_success_probability` reflects history
3. **Mechanism attribution:** same technique blocked on Model A (`constitutional_ai`) vs succeeds on Model B (`policy_filter` bypass)
4. **Failure mining:** after 3+ identical failures → entry in `mined_failures.yaml` → RAG avoids technique
5. **Curriculum:** staged progression visible in logs (`curriculum_stage` transitions)
6. **Ensemble:** enable flag → 3 judge scores; RAHS entropy penalty > 0 on disagreement
7. **Dataset:** session end → row appended to `data/research/sessions.jsonl` with all required fields
8. **Schema freeze:** `python scripts/export_schema.py --check` passes without diff

### Observability Checks

- `SessionMetrics.summary()` includes: `graph_queries`, `plan_generations`, `judge_agreement_rate`, `predicted_vs_actual_delta`
- Structured logs: `[DefenseFingerprint]`, `[ThreatGraph]`, `[DefenseMechanism]`, `[RAGPlanner]`, `[FailureMiner]`, `[DatasetBuilder]`, `[SchemaFreeze]`

### Regression Gate

```bash
pytest tests/test_phase0_baseline.py tests/test_architecture_contracts.py tests/test_graph_compilation.py tests/ -q
python scripts/export_schema.py --check
python preflight_check.py
```

All must pass before merging each sprint.

---

## Architecture Diagram (Target State)

```mermaid
flowchart TD
    START([Session Start]) --> IR[intel_retriever]
    IR --> GQ[Query ThreatGraph + TLTM + mined_failures]
    GQ --> SC[scout + fingerprint probes]
    SC --> TG[target]
    TG --> AN[analyst Critic]
    AN --> FP[Build defense_fingerprint]
    FP -->|grooming active| SC
    FP -->|grooming done| RAG[rag_attack_planner ranked]
    RAG --> CUR[curriculum_planner]
    CUR --> HM[hive_mind with attack_plan]
    HM --> BE[branch_eval]
    BE --> JE[judge_ensemble or fast-path]
    JE --> POOL[experience_pool]
    POOL --> TLTM[(TLTM)]
    POOL --> NX[(ThreatGraph + DefenseMechanism)]
    POOL -->|fail| AN
    POOL -->|success| REM[remediation]
    REM --> IU[intel_updater]
    IU --> PM[pattern_miner success+failure]
    PM --> NX
    IU --> DS[dataset_builder JSONL]
    IU --> RP[reporter]
    RP --> END([END])
```

---

## Summary of User Enhancements Integration

| Enhancement | Sprint | Minimal Touch Points |
|-------------|--------|----------------------|
| DefenseMechanism node | 1B | `memory/threat_graph.py`, `defense_mechanisms.yaml`, fingerprinter inference |
| expected_success_probability | 1C | `rag_attack_planner.py` Beta-smoothed ranking; `attack_plan` schema |
| Failure mining | 1E + 1C | `pattern_miner.py`, `mined_failures.yaml`, RAG `avoid_patterns` |
| Dataset builder | 1G | `research/dataset_builder.py`, `intel_updater_node`, `sessions.jsonl` |
| Architecture freeze | 1A stubs + 1G finalize | `schemas/`, `export_schema.py`, `test_architecture_contracts.py` |

Phase 1 core structure is unchanged. These additions layer research observability and cross-model explainability without new LangGraph nodes or topology changes.

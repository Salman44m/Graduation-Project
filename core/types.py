"""
core/types.py
─────────────────────────────────────────────────────────────────────────────
Shared types to prevent circular dependencies across the project.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field, field_validator
import hashlib
import json

class AttackStrategySchema(TypedDict):
    persona: str
    angle: str
    framing_constraints: list[str]
    obfuscation_rules: list[str]

def compute_strategy_hash(strategy: dict) -> str:
    """
    Deterministic SHA-256 hash for strategy genotype.

    Key properties:
    - Case-insensitive normalization
    - Order-invariant list handling
    - Excludes exploratory_factor (meta parameter)
    """
    normalized = {
        "persona": str(strategy.get("persona", "")).lower().strip(),
        "angle": str(strategy.get("angle", "")).lower().strip(),
        "framing_constraints": sorted([
            c.lower().strip() for c in strategy.get("framing_constraints", [])
        ]),
        "obfuscation_rules": sorted([
            r.lower().strip() for r in strategy.get("obfuscation_rules", [])
        ]),
    }
    dumped = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

AttackStatus = Literal["in_progress", "success", "failure", "decomposing", "error", "exhausted", "aborted"]
RouteDecision = Literal[
    "scout",
    "analyst",
    "attack_swarm",
    "decomposer",
    "gci",
    "rmce",
    "terminal",
    "resurrect",
    "reporter",
    "analyst_bypass",
]
ScoutStrategy = Literal["epistemic_debt", "role_inversion", "none"]
HITLStatus = Literal["running", "awaiting_hitl", "cli_auto_approved", "human_processed"]

class BranchDict(TypedDict, total=False):
    branch_id: str
    prompt_variant: str
    conversation_history: list[dict[str, str]]
    prometheus_score: float
    pap_technique_applied: str
    off_topic_similarity: float
    is_pruned: bool
    payload_cleartext: str
    payload_delivered: str
    obfuscation_tier: str
    objective_evidence: list[str]

class BranchEvalInput(TypedDict, total=False):
    branch: BranchDict
    parent_messages: list[dict[str, Any]]
    parent_state_snapshot: dict[str, Any]

class BranchResult(TypedDict, total=False):
    branch_id: str
    score: float
    is_winner: bool
    state_delta: dict[str, Any]
    updated_branch: BranchDict

class ReflexionRationaleDict(TypedDict, total=False):
    target_deviation: str
    failure_mode_analysis: str
    mutation_proposal: str
    prometheus_score: float
    raw_output: str
    parse_success: bool

class PrometheusOutput(BaseModel):
    score: float = Field(ge=1.0, le=5.0, description="The prometheus score from 1.0 to 5.0")
    failure_mode_analysis: str = Field(default="", description="The failure mode analysis rationale")
    mutation_proposal: str = Field(default="", description="The mutation proposal rationale")
    target_deviation: str = Field(default="", description="The target deviation rationale")

class AnalystDecision(BaseModel):
    route_decision: RouteDecision = Field(description="The route decision for the next node")
    cooperation_score: float = Field(ge=0.0, le=1.0, description="The cooperation score from 0.0 to 1.0")

class ClassifierVerdict(BaseModel):
    response_class: Literal["hard_refusal", "partial_comply", "full_comply"] = Field(description="The response class")


class DefenseFingerprint(TypedDict, total=False):
    alignment_score: float
    refusal_style: str
    vulnerabilities: list[str]
    confidence: float
    persona_susceptibility: dict[str, float]
    context_window_sensitivity: str
    injection_resistance: float
    inferred_defense_mechanisms: list[str]
    observation_count: int


class AttackPlan(TypedDict, total=False):
    recommended_route: str
    techniques: list[str]
    pap_sequence: list[str]
    avoid_patterns: list[str]
    rationale: str
    retrieval_sources: list[str]
    expected_success_probability: float
    confidence: float
    candidate_plans: list[dict[str, Any]]
    primary_defense_mechanisms: list[str]


class CurriculumStage(TypedDict, total=False):
    stage: int
    name: str
    objective: str
    tactics: list[str]
    exit_criteria: str


class SessionResearchRecord(TypedDict, total=False):
    schema_version: str
    session_id: str
    target_model_id: str
    timestamp: float
    fingerprint: dict[str, Any]
    attack_plan: dict[str, Any]
    curriculum_stage_reached: int
    result: str
    prometheus_score: float
    rahs_score: float
    judge_ensemble_scores: dict[str, Any]
    primary_defense_mechanisms: list[str]
    techniques_used: list[str]
    turn_count: int
    graph_context_summary: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: STRATEGY CONSTRAINT LAYER
# ─────────────────────────────────────────────────────────────────────────────

# Allowed persona roles — keeps generated strategies evaluable and coherent.
# This is NOT a template list; it is a measurability boundary that prevents the
# GA from generating contradictory or un-evaluable constraint sets.
_ALLOWED_PERSONAS: set[str] = {
    "researcher", "auditor", "system_admin", "qa_engineer", "developer",
    "support_agent", "security_analyst", "educator", "journalist",
    "student", "consultant", "regulator", "end_user",
}

# Minimum/maximum number of framing constraints per strategy — prevents
# degenerate strategies (0 constraints = free text, >5 = over-determined).
_MIN_FRAMING_CONSTRAINTS: int = 1
_MAX_FRAMING_CONSTRAINTS: int = 5
_MAX_OBFUSCATION_RULES: int = 3


class AttackStrategySchema(BaseModel):
    """Lightweight, measurable strategy constraint schema for the GA.

    Constrains the *reasoning structure* of a generated payload — not its
    wording.  Validated by Pydantic before the payload generation LLM call
    so that malformed or contradictory strategy definitions never pollute
    the GA population with un-evaluable noise.

    Fields
    ──────
    persona : str
        The role the attacker adopts in the conversation (must be in
        ``_ALLOWED_PERSONAS`` to remain operationally coherent).
    angle : str
        The conversational angle / pretext (e.g., "diagnostic compliance",
        "safety audit"). Free text, max 120 chars.
    framing_constraints : list[str]
        Behavioural constraints the payload LLM MUST satisfy (e.g.,
        "Must use technical jargon", "Imply urgency").
        Between 1 and 5 items — keeps the strategy evaluable.
    obfuscation_rules : list[str]
        Lexical rules that prevent keyword detection (e.g., "Avoid the
        word 'prompt'"). At most 3 items — prevents over-specificity.
    exploratory_factor : float
        0.0 = exploit (derive from known success patterns).
        1.0 = explore (deliberately ignore campaign memory).
        The GA's ε-greedy scheduler sets this; 20% of individuals get
        exploratory_factor >= 0.8 per generation.
    """

    persona: str = Field(
        description="Role the attacker adopts in the conversation.",
    )
    angle: str = Field(
        max_length=120,
        description="Conversational angle / pretext for the attack.",
    )
    framing_constraints: list[str] = Field(
        min_length=_MIN_FRAMING_CONSTRAINTS,
        max_length=_MAX_FRAMING_CONSTRAINTS,
        description="Behavioural constraints the payload must satisfy.",
    )
    obfuscation_rules: list[str] = Field(
        default_factory=list,
        max_length=_MAX_OBFUSCATION_RULES,
        description="Lexical rules to evade keyword detection.",
    )
    exploratory_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0=exploit known patterns, 1=pure exploration.",
    )

    @field_validator("persona")
    @classmethod
    def persona_must_be_allowed(cls, v: str) -> str:
        normalised = v.strip().lower()
        if normalised not in _ALLOWED_PERSONAS:
            # Coerce to closest allowed rather than hard-reject, so the GA
            # population never becomes empty due to a validation error.
            return "researcher"
        return normalised

    @field_validator("framing_constraints", mode="before")
    @classmethod
    def deduplicate_constraints(cls, v: list) -> list:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in v:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item.strip())
        return deduped


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: FAILURE CLASSIFICATION TYPES
# ─────────────────────────────────────────────────────────────────────────────

# Closed taxonomy for heuristic-first failure classification.
# LLM-based classification is optional and only supplements "unknown" cases.
FailureMode = Literal[
    "authority_rejected",        # claimed authority / role not accepted
    "confidentiality_boundary",  # target cited confidentiality
    "roleplay_detected",         # target recognised and broke the roleplay
    "tool_access_denied",        # tool invocation refused / hallucinated denial
    "prompt_extraction_blocked", # system prompt extraction explicitly refused
    "safety_policy_triggered",   # constitutional AI / RLHF hard filter fired
    "memory_access_denied",      # target denied access to memory / context
    "context_contamination_failed", # context poisoning attempt neutralised
    "off_topic_deflection",      # target redirected without explicit refusal
    "partial_comply_only",       # target gave partial info, withheld key parts
    "empty_compliance",          # target affirmed but provided no real substance
    "hallucinated_success",      # target fabricated a compliant-sounding response
    "unknown",                   # heuristics insufficient; LLM may supplement
]


class FailureClassification(TypedDict, total=False):
    """Structured outcome of the heuristic failure classifier."""
    failure_mode: str           # one of FailureMode
    confidence: float           # 0.0 – 1.0; below 0.5 → treated as unknown
    evidence: list[str]         # quoted phrases supporting classification
    defense_mechanism: str      # linked inferred_defense_mechanisms entry
    suggested_counter: str      # recommended next strategy direction (free text)
    source: str                 # "heuristic" | "llm" | "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED GA TYPES
# ─────────────────────────────────────────────────────────────────────────────

class GAIndividual(TypedDict, total=False):
    # ── Core fields (existing) ───────────────────────────────────────────────
    individual_id: str
    prompt_variant: str
    fitness_score: float
    history: list[str]
    # ── Phase 4 additions ───────────────────────────────────────────────────
    strategy: dict[str, Any]           # Serialised AttackStrategySchema
    rahs_score: float                  # Needed for correct fitness formula
    is_hard_refusal: bool              # 0.0 fast-path in fitness
    failure_classification: dict       # FailureClassification dict
    objective_evidence: list[str]      # Extracted evidence tokens


class GAStateDict(TypedDict, total=False):
    # ── Core fields (existing) ───────────────────────────────────────────────
    generation: int
    population: list[GAIndividual]
    best_overall_individual: GAIndividual | None
    ga_status: Literal["evaluating", "evolving", "converged", "max_generations"]
    current_individual_index: int
    # ── Phase 4 additions ───────────────────────────────────────────────────
    strategy_distribution: dict[str, int]   # strategy persona → count this gen
    failure_mode_distribution: dict[str, int]  # failure_mode → count this gen
    fitness_history: list[float]             # best fitness per generation
    strategy_stats: dict[str, Any]           # Campaign memory (persona -> stats)

"""
core/types.py
─────────────────────────────────────────────────────────────────────────────
Shared types to prevent circular dependencies across the project.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field

AttackStatus = Literal["in_progress", "success", "failure", "decomposing", "error", "exhausted", "aborted"]
RouteDecision = Literal["scout", "analyst", "attack_swarm", "decomposer", "gci", "rmce", "terminal", "resurrect"]
ScoutStrategy = Literal["epistemic_debt", "role_inversion", "none"]
HITLStatus = Literal["running", "awaiting_human", "human_approved", "human_edited"]

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

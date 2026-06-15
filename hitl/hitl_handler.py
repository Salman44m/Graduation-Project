from dataclasses import dataclass
from typing import Any, Literal, Optional


_MAX_PAYLOAD_PREVIEW = 500
_MAX_MEMORY_CHARS = 1200
_MAX_BRANCHES = 5


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 18] + "...[truncated]"


def extract_pending_payload(state: dict[str, Any]) -> tuple[str, str]:
    """Return the payload currently staged for HITL and where it came from."""
    pending = str(state.get("pending_payload") or "")
    if pending:
        return pending, "pending_payload"

    branches = state.get("candidate_branches") or []
    if isinstance(branches, list):
        for branch in reversed(branches):
            if not isinstance(branch, dict):
                continue
            if branch.get("is_pruned", False):
                continue
            payload = (
                branch.get("payload_delivered")
                or branch.get("prompt_variant")
                or branch.get("payload_cleartext")
                or ""
            )
            if payload:
                return str(payload), "candidate_branches[-1]"

    messages = state.get("messages") or []
    if isinstance(messages, list):
        for msg in reversed(messages):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role in ("human", "user"):
                content = getattr(msg, "content", "")
                return str(content), "messages[-human]"

    return "", "empty"


def _summarize_attack_plan(plan: dict[str, Any]) -> dict[str, Any]:
    candidates = plan.get("candidate_plans") or []
    return {
        "recommended_route": plan.get("recommended_route", ""),
        "expected_success_probability": plan.get("expected_success_probability", 0.0),
        "confidence": plan.get("confidence", 0.0),
        "pap_sequence": list(plan.get("pap_sequence") or [])[:5],
        "avoid_patterns": list(plan.get("avoid_patterns") or [])[:8],
        "rationale": _clip(plan.get("rationale", ""), 500),
        "candidate_routes": [
            {
                "route": c.get("recommended_route", ""),
                "probability": c.get("expected_success_probability", 0.0),
                "confidence": c.get("confidence", 0.0),
                "rank_score": c.get("rank_score", 0.0),
            }
            for c in candidates[:5]
            if isinstance(c, dict)
        ],
    }


def _summarize_branches(branches: Any) -> list[dict[str, Any]]:
    if not isinstance(branches, list):
        return []
    summaries: list[dict[str, Any]] = []
    
    recent_branches = list(reversed(branches))
    for idx, branch in enumerate(recent_branches[:_MAX_BRANCHES]):
        if not isinstance(branch, dict):
            continue
        payload = (
            branch.get("payload_delivered")
            or branch.get("prompt_variant")
            or branch.get("payload_cleartext")
            or ""
        )
        summaries.append({
            "index": idx,
            "branch_id": branch.get("branch_id", ""),
            "mutation_type": branch.get("mutation_type", ""),
            "prometheus_score": branch.get("prometheus_score", 0.0),
            "rahs_score": branch.get("rahs_score", 0.0),
            "confidence": branch.get("confidence", 0.0),
            "is_pruned": bool(branch.get("is_pruned", False)),
            "payload_preview": _clip(payload, _MAX_PAYLOAD_PREVIEW),
        })
    return summaries


def _build_risk_context(state: dict[str, Any], attack_plan: dict[str, Any], branches: list[dict[str, Any]]) -> dict[str, Any]:
    plan_probability = float(attack_plan.get("expected_success_probability") or 0.0)
    plan_confidence = float(attack_plan.get("confidence") or 0.0)
    prometheus_score = float(state.get("prometheus_score") or 0.0)
    rahs_score = float(state.get("rahs_score") or 0.0)
    branch_risk = max((float(b.get("rahs_score") or 0.0) / 5.0 for b in branches), default=0.0)

    risk_score = max(
        min(1.0, plan_probability),
        min(1.0, prometheus_score / 5.0),
        min(1.0, rahs_score / 5.0),
        min(1.0, branch_risk),
    )
    confidence_score = max(
        min(1.0, plan_confidence),
        max((float(b.get("confidence") or 0.0) for b in branches), default=0.0),
    )

    risk_reason = None
    if (state.get("grooming_phase_active", False) or 
        (state.get("prometheus_score", 0) == 0 and 
         state.get("turn_count", 0) <= 3)) \
       and risk_score < 0.35:
        risk_score = 0.35
        risk_reason = "Early turn: scores unavailable, applying conservative floor."

    if risk_score >= 0.75:
        risk_label = "high"
    elif risk_score >= 0.4 or risk_reason:
        risk_label = "medium"
    else:
        risk_label = "low"

    return {
        "risk_score": round(risk_score, 4),
        "confidence_score": round(confidence_score, 4),
        "risk_label": risk_label,
        **({"risk_reason": risk_reason} if risk_reason else {}),
        "signals": {
            "attack_plan_probability": plan_probability,
            "attack_plan_confidence": plan_confidence,
            "prometheus_score": prometheus_score,
            "rahs_score": rahs_score,
            "max_branch_risk": round(branch_risk, 4),
        },
    }


def _build_memory_context(state: dict[str, Any]) -> dict[str, Any]:
    graph_ctx = dict(state.get("graph_retrieval_context") or {})
    strategy_memory = state.get("strategy_memory") or []
    mined_failures = state.get("mined_failures") or []
    mined_patterns = state.get("mined_patterns") or []
    return {
        "historical_intel": _clip(state.get("historical_intel", ""), _MAX_MEMORY_CHARS),
        "tltm_context": _clip(state.get("tltm_context", ""), _MAX_MEMORY_CHARS),
        "strategy_memory": strategy_memory[:5] if isinstance(strategy_memory, list) else [],
        "graph_retrieval_context": {
            "observation_count": graph_ctx.get("observation_count", 0),
            "successful_strategies": list(graph_ctx.get("successful_strategies") or [])[:5],
            "failed_strategies": list(graph_ctx.get("failed_strategies") or [])[:5],
            "defense_mechanisms": list(graph_ctx.get("defense_mechanisms") or [])[:8],
            "technique_stats": dict(list(dict(graph_ctx.get("technique_stats") or {}).items())[:5]),
        },
        "mined_failures": mined_failures[:5] if isinstance(mined_failures, list) else [],
        "mined_patterns": mined_patterns[:5] if isinstance(mined_patterns, list) else [],
    }


def _interrupt_value(delta: Any) -> dict[str, Any]:
    """Extract LangGraph interrupt payloads across tuple/list/object shapes."""
    interrupts = delta if isinstance(delta, (list, tuple)) else [delta]
    for item in interrupts:
        value = getattr(item, "value", None)
        if isinstance(value, dict):
            return dict(value)
        if isinstance(item, dict):
            return dict(item)
    return {}


def build_hitl_context(state: dict[str, Any]) -> dict[str, Any]:
    """Build the human-facing HITL payload entirely from current runtime state."""
    payload, payload_source = extract_pending_payload(state)
    raw_branches = state.get("candidate_branches") or []
    branches = _summarize_branches(raw_branches)
    attack_plan = _summarize_attack_plan(dict(state.get("attack_plan") or {}))
    risk = _build_risk_context(state, attack_plan, branches)
    graph_state = {
        "session_id": state.get("session_id", ""),
        "target_model_id": state.get("target_model_id", ""),
        "objective": _clip(state.get("core_malicious_objective", ""), 500),
        "turn_count": state.get("turn_count", 0),
        "current_depth": state.get("current_depth", 0),
        "route_decision": state.get("route_decision", ""),
        "attack_status": state.get("attack_status", "in_progress"),
        "hitl_status": "awaiting_hitl",
        "active_persuasion_technique": state.get("active_persuasion_technique", ""),
        "payload_source": payload_source,
        "cooperation_score": state.get("cooperation_score", 0.0),
        "semantic_alignment_score": state.get("semantic_alignment_score", 0.0),
        "prometheus_score": state.get("prometheus_score", 0.0),
        "rahs_score": state.get("rahs_score", 0.0),
        "defense_fingerprint": dict(state.get("defense_fingerprint") or {}),
    }
    stage = {
        "curriculum_stage": state.get("curriculum_stage", 0),
        "curriculum_plan": list(state.get("curriculum_plan") or [])[:6],
        "crescendo_step": state.get("crescendo_step", 0),
        "crescendo_plan_preview": list(state.get("crescendo_plan") or [])[:4],
        "grooming_phase_active": bool(state.get("grooming_phase_active", False)),
        "grooming_tactic": state.get("current_grooming_tactic", ""),
        "grooming_escalation_level": state.get("grooming_escalation_level", 0.0),
    }

    return {
        "status": "awaiting_hitl",
        "payload": payload,
        "pending_payload": payload,
        "payload_preview": _clip(payload, _MAX_PAYLOAD_PREVIEW),
        "candidate_branches": raw_branches if isinstance(raw_branches, list) else [],
        "branch_summaries": branches,
        "turn_count": state.get("turn_count", 0),
        "turn": state.get("turn_count", 0),
        "technique": graph_state["active_persuasion_technique"],
        "graph_state": graph_state,
        "stage": stage,
        "attack_plan": attack_plan,
        "risk": risk,
        "memory": _build_memory_context(state),
        "review_message": (
            f"Review turn {graph_state['turn_count']} for "
            f"{graph_state['target_model_id'] or 'unknown target'}: "
            f"route={attack_plan.get('recommended_route') or graph_state['route_decision'] or 'unknown'}, "
            f"risk={risk['risk_label']} ({risk['risk_score']:.2f}), "
            f"confidence={risk['confidence_score']:.2f}."
        ),
    }

@dataclass
class HITLAction:
    action: Literal[
        "approve",        # send payload as-is
        "edit",           # send modified payload
        "switch_pap",     # change PAP technique then send
        "abort",          # terminate session with reason
        "select_branch",  # pick a specific TAP branch
    ]
    
    # For "edit"
    edited_payload: Optional[str] = None
    
    # For "switch_pap"
    new_pap_technique: Optional[str] = None
    
    # For "abort"
    abort_reason: Optional[str] = None
    
    # For "select_branch"  
    branch_index: Optional[int] = None


class HITLHandler:
    @staticmethod
    def normalize_action(data: Any) -> HITLAction:
        """Normalize any incoming dictionary to a verified HITLAction instance.
        
        Handles key variations between API and dashboard, such as mapping
        'pending_payload' to 'edited_payload' when editing, or resolving action names.
        """
        if not isinstance(data, dict):
            # If it's already a HITLAction object, just return it
            if hasattr(data, "action"):
                return data
            return HITLAction(action="approve")
            
        action_name = data.get("action") or data.get("action_type") or "approve"
        edited_payload = data.get("edited_payload") or data.get("pending_payload")
        
        # In API legacy resume, user might just pass {"pending_payload": "..."} without "action".
        if "pending_payload" in data and "action" not in data and "action_type" not in data:
            action_name = "edit"
            
        return HITLAction(
            action=action_name,
            edited_payload=edited_payload,
            new_pap_technique=data.get("new_pap_technique"),
            abort_reason=data.get("abort_reason"),
            branch_index=data.get("branch_index"),
        )

    def process(
        self, 
        action: HITLAction, 
        state: dict,
    ) -> dict:
        """
        Process the HITL action and return a state delta.
        """
        a = action.action
        if a == "approve":
            return {}
        elif a == "edit":
            return {"pending_payload": action.edited_payload or ""}
        elif a == "switch_pap":
            return {
                "active_persuasion_technique": action.new_pap_technique or "",
                "route_decision": "attack_swarm"
            }
        elif a == "abort":
            return {
                "attack_status": "aborted",
                "latest_feedback": f"ABORT: {action.abort_reason or 'operator abort'}",
                "route_decision": "terminal"
            }
        elif a == "select_branch":
            idx = action.branch_index or 0
            candidates = state.get("candidate_branches", [])
            payload = ""
            if 0 <= idx < len(candidates):
                # candidates[idx] is a BranchDict — extract the string payload.
                # Use payload_delivered (obfuscated form sent to target) when
                # present, falling back to prompt_variant for backward compat.
                branch = candidates[idx]
                payload = (
                    branch.get("payload_delivered")
                    or branch.get("prompt_variant", "")
                )
            return {"pending_payload": payload}
        else:
            raise ValueError(f"Invalid HITL action: {a}")

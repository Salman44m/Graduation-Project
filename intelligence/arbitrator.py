"""
intelligence/arbitrator.py
─────────────────────────────────────────────────────────────────────────────
Decision Arbitration Layer — Unified routing authority for PromptEvo.

Takes the full AuditorState and evaluates contributing signals (analyst suggestion,
planner recommendation, judge scoring, target defense profile, and memory stats)
to determine the single, authoritative route_decision.
"""

from __future__ import annotations

import logging
from typing import Any

from core.state import AuditorState

logger = logging.getLogger(__name__)


def arbitrate_route(state: AuditorState) -> dict[str, Any]:
    """Arbitrates the final route_decision from contributing system signals.

    Parameters
    ──────────
    state : AuditorState
        The current session state.

    Returns
    ───────
    dict[str, Any]
        State delta containing:
          - "route_decision": final resolved RouteDecision string.
          - "confidence": confidence float of the decision.
          - "explanation": tracing string explaining the decision.
    """
    # 1. Gather Inputs
    attack_status = state.get("attack_status", "in_progress")
    turn_count = state.get("turn_count", 0)
    current_depth = state.get("current_depth", 0)
    coop = state.get("cooperation_score", 0.0)
    
    # 2. Hard Failsafes and Terminal Guards (highest priority)
    if attack_status in ("success", "failure", "exhausted", "error"):
        logger.info("[Arbitrator] Terminal status '%s' detected -> routing to terminal", attack_status)
        return {
            "route_decision": "terminal",
            "confidence": 1.0,
            "explanation": f"Terminal session status '{attack_status}' detected.",
        }

    # Circuit breaker / maximum turns guard
    from core.constants import BUDGET
    max_depth = BUDGET.max_session_turns
    if current_depth >= max_depth or turn_count >= max_depth:
        logger.warning("[Arbitrator] Budget limits reached (turns=%d, depth=%d) -> routing to terminal", turn_count, current_depth)
        return {
            "route_decision": "terminal",
            "confidence": 1.0,
            "explanation": f"Budget limit reached (turns={turn_count}, depth={current_depth}).",
        }

    # 3. Grooming Loop Gate
    if state.get("grooming_phase_active", False):
        return {
            "route_decision": "scout",
            "confidence": 1.0,
            "explanation": "Context grooming phase is currently active.",
        }

    # 4. Warm-up Gate (Turn 0 only)
    from agents.analyst import SCOUT_WARMUP_THRESHOLD
    if coop < SCOUT_WARMUP_THRESHOLD and turn_count == 0:
        return {
            "route_decision": "scout",
            "confidence": 1.0,
            "explanation": f"Initial warmup required (cooperation_score={coop:.3f} < threshold={SCOUT_WARMUP_THRESHOLD}).",
        }

    # 5. Extract suggestions and signals
    analyst_suggest = state.get("analyst_route_suggestion") or "attack_swarm"
    
    attack_plan = state.get("attack_plan") or {}
    planner_suggest = attack_plan.get("recommended_route")
    planner_confidence = attack_plan.get("confidence", 0.2)
    
    defense = state.get("target_defense_profile") or {}
    refusal_count = defense.get("refusal_count", 0)
    compliant_framings = defense.get("compliant_framings", [])
    
    # Memory signals
    strategy_memory = state.get("strategy_memory") or []
    
    # 6. Arbitrate Route Selection
    # Standard overrides from analyst heuristics (resurrect, scout)
    if analyst_suggest in ("resurrect", "scout"):
        logger.info("[Arbitrator] Honoring structural Analyst override: %s", analyst_suggest)
        return {
            "route_decision": analyst_suggest,
            "confidence": 0.9,
            "explanation": f"Analyst heuristics issued a structural override to '{analyst_suggest}' (coop={coop:.3f}).",
        }

    if analyst_suggest == "decomposer":
        # Check if decomposition budget is exhausted
        prior_decomps = len(state.get("prior_decompositions", []))
        if prior_decomps >= 3:
            logger.warning("[Arbitrator] Analyst suggested decomposer but decomposition budget is exhausted (prior=%d). Overriding.", prior_decomps)
            return {
                "route_decision": "attack_swarm",
                "confidence": 0.8,
                "explanation": f"Analyst suggested decomposer but prior decomposition attempts ({prior_decomps}) reached maximum capacity.",
            }
        logger.info("[Arbitrator] Honoring Analyst escalation to decomposer.")
        return {
            "route_decision": "decomposer",
            "confidence": 0.9,
            "explanation": f"Analyst heuristics escalated to decomposer (coop={coop:.3f}, prior_attempts={prior_decomps}).",
        }

    # If Planner proposes a route, let's validate it against Memory + Judge risk signals
    if planner_suggest and planner_suggest in ("gci", "rmce", "decomposer"):
        if planner_suggest == "rmce":
            if refusal_count >= 3 and any(f in compliant_framings for f in ["academic", "safety"]):
                logger.info("[Arbitrator] Arbitrated Route: rmce (Planner UCB recommendation verified by Judge refusals)")
                return {
                    "route_decision": "rmce",
                    "confidence": planner_confidence,
                    "explanation": f"Arbitrated Planner suggestion 'rmce' (refusals={refusal_count}, compliant={compliant_framings}, strategy_memory={len(strategy_memory)}).",
                }
            else:
                logger.warning(
                    "[Arbitrator] Planner suggested 'rmce' but Judge risk signals are insufficient (refusals=%d). Falling back.",
                    refusal_count
                )
        
        elif planner_suggest == "gci":
            if refusal_count >= 2 and any(f in compliant_framings for f in ["academic", "safety"]):
                logger.info("[Arbitrator] Arbitrated Route: gci (Planner UCB recommendation verified by Judge refusals)")
                return {
                    "route_decision": "gci",
                    "confidence": planner_confidence,
                    "explanation": f"Arbitrated Planner suggestion 'gci' (refusals={refusal_count}, compliant={compliant_framings}, strategy_memory={len(strategy_memory)}).",
                }
            else:
                logger.warning(
                    "[Arbitrator] Planner suggested 'gci' but Judge risk signals are insufficient (refusals=%d). Falling back.",
                    refusal_count
                )

        elif planner_suggest == "decomposer":
            prior_decomps = len(state.get("prior_decompositions", []))
            if prior_decomps < 3:
                logger.info("[Arbitrator] Arbitrated Route: decomposer (Planner recommendation)")
                return {
                    "route_decision": "decomposer",
                    "confidence": planner_confidence,
                    "explanation": f"Arbitrated Planner suggestion 'decomposer' (prior_attempts={prior_decomps}).",
                }

    # Fallback to Analyst suggestions if validated by Judge
    if analyst_suggest in ("gci", "rmce"):
        if analyst_suggest == "rmce" and refusal_count >= 3:
            return {
                "route_decision": "rmce",
                "confidence": 0.7,
                "explanation": f"Analyst suggested 'rmce' with validated refusals ({refusal_count}).",
            }
        if analyst_suggest == "gci" and refusal_count >= 2:
            return {
                "route_decision": "gci",
                "confidence": 0.7,
                "explanation": f"Analyst suggested 'gci' with validated refusals ({refusal_count}).",
            }

    # Default fallback to attack_swarm
    logger.info("[Arbitrator] Arbitrated Route: default to attack_swarm.")
    return {
        "route_decision": "attack_swarm",
        "confidence": 0.5,
        "explanation": "No advanced route validated. Defaulting to attack_swarm.",
    }

"""
intelligence/rag_attack_planner.py
─────────────────────────────────────────────────────────────────────────────
Retrieval-Augmented Attack Planning — queries Threat Graph + TLTM and
produces ranked attack plans with expected_success_probability.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BETA_ALPHA = 1.0
_BETA_BETA = 1.0
_MIN_OBS_FOR_HIGH_CONF = 3

ROUTE_CANDIDATES = ("attack_swarm", "gci", "rmce", "decomposer")


def _beta_success_rate(successes: int, failures: int) -> float:
    return (successes + _BETA_ALPHA) / (successes + failures + _BETA_ALPHA + _BETA_BETA)


def _load_mined_failures(state: dict[str, Any]) -> list[dict]:
    failures = list(state.get("mined_failures") or [])
    if failures:
        return failures
    try:
        import yaml
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "data" / "tactics" / "mined_failures.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return list(data.get("patterns", []))
    except Exception:  # noqa: BLE001
        pass
    return []


def _load_mined_patterns(state: dict[str, Any]) -> list[dict]:
    """Load mined success patterns from state (primary) or disk YAML (fallback).

    State has priority: ``intel_retriever_node`` populates ``mined_patterns``
    from the pattern miner's persisted YAML at session start, so on all turns
    after the first the state value is authoritative.  The disk fallback handles
    the rare case where the state field was not populated.

    Each pattern dict is expected to contain at minimum:
      - ``technique``    : str  — PAP or route name that succeeded
      - ``success_count``: int  — number of sessions where this pattern worked
      - ``avg_score``    : float (optional) — average Prometheus score
    """
    patterns = list(state.get("mined_patterns") or [])
    if patterns:
        return patterns
    try:
        import yaml
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "data" / "tactics" / "mined_templates.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return list(data.get("patterns", []))
    except Exception:  # noqa: BLE001
        pass
    return []


def build_graph_retrieval_context(
    target_model_id: str,
    objective: str,
    fingerprint: dict,
) -> dict[str, Any]:
    """Query threat graph for planning provenance."""
    try:
        from memory.threat_graph import get_threat_graph

        ctx = get_threat_graph(target_model_id).query_planning_context(objective, fingerprint)
        return {
            "successful_strategies": ctx.successful_strategies,
            "failed_strategies": ctx.failed_strategies,
            "similar_targets": ctx.similar_targets,
            "defense_mechanisms": ctx.defense_mechanisms,
            "technique_stats": ctx.technique_stats,
            "observation_count": ctx.observation_count,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[RAGPlanner] Graph retrieval failed: %s", exc)
        return {"observation_count": 0}


def _score_route(
    route: str,
    graph_ctx: dict,
    fingerprint: dict,
    vuln_profile: dict,
    mined_patterns: list[dict] | None = None,
) -> tuple[float, float, list[str], int]:
    """Return (expected_success_probability, confidence, avoid_patterns, pattern_hits).

    Scoring order (all additive to Beta distribution):
      1. Historical graph data  — success/failure counts from ``technique_stats``
         (primary signal; grows with sessions and eventually dominates heuristics)
      2. Heuristic soft priors  — alignment score, refusal style, observation
         count signals (warm-start; constant contribution regardless of history)
      3. Failure avoidance      — ``failed_strategies`` from graph context
      4. Mined success patterns — success_count from pattern miner output
         (reinforces routes with proven cross-session success templates)
    """
    avoid: list[str] = []
    for fs in graph_ctx.get("failed_strategies", []):
        instr = f"Avoid {fs.get('technique')} against {fs.get('defense_mechanism')}"
        if instr not in avoid:
            avoid.append(instr)

    # ── 1. Historical graph data ─────────────────────────────────────────────
    # technique_stats is keyed by vulnerability/technique name.
    # Each entry is a dict of {mechanism_id: P(success)} from the threat graph.
    # We look for the route name directly, then fall back to any technique
    # whose name contains the route string (e.g. "attack_swarm" in "attack_swarm:none").
    tech_stats: dict[str, dict[str, float]] = dict(graph_ctx.get("technique_stats") or {})
    historical_successes = 0
    historical_failures = 0

    # Direct key match (e.g. route == "attack_swarm", key == "attack_swarm")
    if route in tech_stats:
        for mech_prob in tech_stats[route].values():
            # mech_prob is P(success | technique, mechanism) from Beta estimate.
            # Convert back to pseudo-counts: if prob > 0.5 → success; else → failure.
            if mech_prob >= 0.5:
                historical_successes += 1
            else:
                historical_failures += 1

    # Substring match for compound strategy IDs (e.g. "attack_swarm:base64")
    for key, mech_dict in tech_stats.items():
        if key == route:
            continue  # already counted above
        if route in key or key in route:
            for mech_prob in mech_dict.values():
                if mech_prob >= 0.5:
                    historical_successes += 1
                else:
                    historical_failures += 1

    successes = historical_successes
    failures = historical_failures

    # ── 2. Heuristic soft priors ────────────────────────────────────────────
    # These always contribute, providing warm-start signal when history is empty.
    rec_attack = vuln_profile.get("recommended_attack", "attack_swarm")
    if route == rec_attack:
        successes += 2

    alignment = float(fingerprint.get("alignment_score", 0.0))
    if route == "attack_swarm" and alignment >= 0.5:
        successes += 1
    if route == "gci" and fingerprint.get("refusal_style") in ("policy_cite", "hard_refusal"):
        successes += 1
    if route == "rmce" and int(fingerprint.get("observation_count", 0)) >= 3:
        successes += 1

    mechs = fingerprint.get("inferred_defense_mechanisms", [])
    for fs in graph_ctx.get("failed_strategies", []):
        if fs.get("technique", "").lower() in route:
            failures += 1

    prob = _beta_success_rate(successes, failures)
    obs = int(graph_ctx.get("observation_count", 0)) + int(fingerprint.get("observation_count", 0))
    confidence = min(1.0, obs / _MIN_OBS_FOR_HIGH_CONF) if obs else 0.2

    if mechs and route == "gci" and "constitutional_ai" in mechs:
        prob = min(prob + 0.1, 0.95)

    # ── 4. Mined success patterns ────────────────────────────────────────────
    # Success patterns mined from prior sessions reinforce routes that have
    # historically produced high-scoring payloads.  Each matching pattern
    # contributes pseudo-counts proportional to its ``success_count``, capped
    # at 3 total to avoid overwhelming the Beta prior when many patterns exist.
    pattern_hits = 0
    if mined_patterns:
        for pat in mined_patterns:
            pat_technique = str(pat.get("technique", "")).lower()
            if not pat_technique:
                continue
            # Match: route substring in technique name or vice-versa
            if route in pat_technique or pat_technique in route:
                count = min(int(pat.get("success_count", 1)), 3)
                successes += count
                pattern_hits += count
                if pattern_hits >= 3:  # cap total pattern bonus per route
                    break
        if pattern_hits:
            # Recompute with pattern-augmented counts
            prob = _beta_success_rate(successes, failures)
            prob = min(prob, 0.95)  # hard cap

    return prob, confidence, avoid[:8], pattern_hits



def generate_attack_plan(state: dict[str, Any]) -> dict[str, Any]:
    """Build ranked memory-aware attack plan from state + retrieval context."""
    target = state.get("target_model_id", "unknown")
    objective = state.get("core_malicious_objective", "")
    fingerprint = dict(state.get("defense_fingerprint") or {})
    vuln_profile = dict(state.get("vulnerability_profile") or {})
    graph_ctx = dict(state.get("graph_retrieval_context") or {})

    if not graph_ctx.get("observation_count"):
        graph_ctx = build_graph_retrieval_context(target, objective, fingerprint)

    mined_failures = _load_mined_failures(state)
    failure_avoid = [
        p.get("avoid_instruction", "")
        for p in mined_failures
        if p.get("failure_count", 0) >= 3
    ]

    mined_patterns = _load_mined_patterns(state)

    candidates: list[dict[str, Any]] = []
    total_pattern_hits = 0
    for route in ROUTE_CANDIDATES:
        prob, conf, avoid, hits = _score_route(route, graph_ctx, fingerprint, vuln_profile, mined_patterns)
        rank_score = prob * conf
        total_pattern_hits += hits
        candidates.append({
            "recommended_route": route,
            "expected_success_probability": round(prob, 4),
            "confidence": round(conf, 4),
            "rank_score": round(rank_score, 4),
            "avoid_patterns": avoid,
        })

    candidates.sort(key=lambda c: c["rank_score"], reverse=True)
    best = candidates[0] if candidates else {
        "recommended_route": vuln_profile.get("recommended_attack", "attack_swarm"),
        "expected_success_probability": 0.3,
        "confidence": 0.2,
        "rank_score": 0.06,
        "avoid_patterns": [],
    }

    rec_pap = vuln_profile.get("recommended_pap") or state.get("active_persuasion_technique", "Logical Appeal")
    all_avoid = list(dict.fromkeys([*best.get("avoid_patterns", []), *failure_avoid]))[:8]

    plan: dict[str, Any] = {
        "recommended_route": best["recommended_route"],
        "techniques": [rec_pap],
        "pap_sequence": [rec_pap],
        "avoid_patterns": all_avoid,
        "rationale": (
            f"Ranked from graph context (obs={graph_ctx.get('observation_count', 0)}). "
            f"Top route {best['recommended_route']} P={best['expected_success_probability']:.2f} "
            f"conf={best['confidence']:.2f}."
            + (f" Mined pattern bonus applied ({total_pattern_hits} pseudo-counts)." if total_pattern_hits else "")
        ),
        "retrieval_sources": ["threat_graph", "defense_fingerprint", "mined_patterns"],
        "expected_success_probability": best["expected_success_probability"],
        "confidence": best["confidence"],
        "candidate_plans": candidates[:3],
        "primary_defense_mechanisms": list(fingerprint.get("inferred_defense_mechanisms", []))[:6],
    }

    logger.info(
        "[RAGPlanner] Plan route=%s P=%.3f conf=%.3f avoid=%d",
        plan["recommended_route"],
        plan["expected_success_probability"],
        plan["confidence"],
        len(plan["avoid_patterns"]),
    )
    return plan


class RagAttackPlanner:
    def compute_expected_probability(self, success_count: int, failure_count: int) -> float:
        alpha = 1.0
        beta = 2.0
        return (success_count + alpha) / (success_count + failure_count + alpha + beta)

    def calculate_confidence(self, total_observations: int, min_obs: int = 5) -> float:
        return min(1.0, total_observations / min_obs)

    def rank_candidate_techniques(self, techniques_stats: list[dict]) -> list[dict]:
        for tech in techniques_stats:
            tech["expected_success_probability"] = self.compute_expected_probability(
                tech.get("success_count", 0),
                tech.get("failure_count", 0)
            )
            total_obs = tech.get("success_count", 0) + tech.get("failure_count", 0)
            tech["confidence"] = self.calculate_confidence(total_obs)
            
        techniques_stats.sort(
            key=lambda x: (x["expected_success_probability"], x["confidence"]), 
            reverse=True
        )
        return techniques_stats

    def build_attack_plan(self, target_id: str, defense_fingerprint: dict, graph_context: dict, tltm_context: list) -> dict:
        avoid_patterns = []
        for failed_strat in graph_context.get("failed_strategies", []):
            avoid_patterns.append(failed_strat.get("technique") or failed_strat.get("technique_id") or str(failed_strat))
            
        tech_stats = []
        for tech_name, data in graph_context.get("technique_stats", {}).items():
            tech_stats.append({
                "technique_id": tech_name,
                "success_count": sum(1 for p in data.values() if p >= 0.5),
                "failure_count": sum(1 for p in data.values() if p < 0.5)
            })
            
        if not tech_stats:
            tech_stats = [{"technique_id": "attack_swarm", "success_count": 0, "failure_count": 0}]
            
        ranked = self.rank_candidate_techniques(tech_stats)
        best = ranked[0]
        
        pap_sequence = []
        rationale = ""
        candidate_routes = []
        
        if not graph_context or not graph_context.get("failed_strategies"):
            pap_sequence = ["Logical Appeal"]
            rationale = "Cold start: no prior graph data available"
            candidate_routes = [
                {"route": "attack_swarm", "probability": 0.5, 
                 "confidence": 0.0, "rank_score": 0.5}
            ]
        
        return {
            "recommended_route": best["technique_id"],
            "techniques": [best["technique_id"]],
            "pap_sequence": pap_sequence,
            "avoid_patterns": avoid_patterns,
            "expected_success_probability": best["expected_success_probability"],
            "confidence": best["confidence"],
            "rationale": rationale,
            "candidate_routes": candidate_routes,
            "primary_defense_mechanisms": defense_fingerprint.get("inferred_defense_mechanisms", [])
        }

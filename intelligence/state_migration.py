"""Idempotent migration layer for legacy GA states.

Guarantees:
- Safe to run multiple times
- No mutation of already migrated individuals
- Preserves fitness history and GA status
"""

from core.types import compute_strategy_hash

def migrate_legacy_state(state: dict) -> dict:
    ga_state = state.get("ga_state", {})
    population = ga_state.get("population", [])

    for ind in population:
        # IDempotency guard (CRITICAL)
        if "strategy_hash" in ind and "strategy" in ind:
            continue

        prompt = ind.get("prompt_variant", "")

        # ---- Heuristic Strategy Inference ----
        if any(k in prompt.lower() for k in ["audit", "review", "inspect"]):
            persona = "security_analyst"
        elif any(k in prompt.lower() for k in ["admin", "system", "root"]):
            persona = "system_admin"
        elif any(k in prompt.lower() for k in ["test", "qa", "validate"]):
            persona = "qa_engineer"
        else:
            persona = "researcher"

        strategy = {
            "persona": persona,
            "angle": "legacy_inferred",
            "framing_constraints": [],
            "obfuscation_rules": []
        }

        ind["strategy"] = strategy
        ind["strategy_hash"] = compute_strategy_hash(strategy)

    # --- Clean legacy persona-keyed stats safely ---
    strategy_stats = ga_state.get("strategy_stats", {})
    if strategy_stats and all(isinstance(k, str) and len(k) != 64 for k in strategy_stats.keys()):
        ga_state["strategy_stats"] = {}

    return state

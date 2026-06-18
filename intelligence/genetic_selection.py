"""Memory-driven selection layer using Log-Softmax stability.

Changes (Post-Scout Fix)
─────────────────────────
* RC-4: Fixed strategy_stats lookup key from strategy_hash to persona.
  campaign_memory.update_strategy_memory() stores stats keyed by persona
  (lowercase string), but select_parents() was looking up by strategy_hash
  (SHA-256 hex), causing all memory lookups to return empty dicts and
  effectively disabling the campaign memory signal during selection.
"""

import math
import random

def jaccard_similarity(list_a: list, list_b: list) -> float:
    set_a, set_b = set(list_a), set(list_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def compute_distance(schema_a: dict, schema_b: dict) -> float:
    persona_mismatch = 1.0 if schema_a.get("persona") != schema_b.get("persona") else 0.0
    angle_mismatch = 1.0 if schema_a.get("angle") != schema_b.get("angle") else 0.0
    
    constraints_dist = 1.0 - jaccard_similarity(
        schema_a.get("framing_constraints", []),
        schema_b.get("framing_constraints", [])
    )
    
    obfuscation_dist = 1.0 - jaccard_similarity(
        schema_a.get("obfuscation_rules", []),
        schema_b.get("obfuscation_rules", [])
    )
    
    return 0.3 * persona_mismatch + 0.2 * angle_mismatch + 0.3 * constraints_dist + 0.2 * obfuscation_dist


def select_parents(
    population: list[dict], 
    strategy_stats: dict, 
    selection_counts: dict,
    novelty_archive: list[dict],
    selection_temperature: float = 1.0
) -> tuple[dict, dict]:
    """
    Selects parents via multi-signal integration.
    Uses Log-Space + Temperature-Scaled Softmax selection (Probabilistic Model) 
    with Z-Score Normalization and Degenerate Distribution Guard.

    RC-4 Note: strategy_stats is keyed by persona (lowercase string), which
    matches the key written by campaign_memory.update_strategy_memory().
    The previous implementation looked up by strategy_hash, causing a
    permanent cache miss and making the memory signal always return defaults.
    """
    log_weights = []
    EPSILON = 1e-4
    
    for ind in population:
        s_hash = ind.get("strategy_hash", "")
        schema = ind.get("strategy", {})

        # RC-4: Look up stats by persona (matches campaign_memory write key),
        # not by strategy_hash (which was always a cache miss).
        persona_key = (
            schema.get("persona", "").lower().strip()
            if isinstance(schema, dict)
            else ""
        )
        stats = strategy_stats.get(persona_key, {})
        
        # 1. Base Fitness (Pure Learning Signal)
        base_fitness = ind.get("fitness_score", 0.0) 
        
        # 2. Historical Signal (Memory)
        conf = stats.get("confidence", 0.5)
        
        # 3. Environment Signal (Drift)
        drift_penalty = 0.5 if stats.get("is_drift_suspected", False) else 1.0
        
        # 4. Diversity Signal (Generation-wide penalty)
        freq_penalty = 1.0 / (1.0 + selection_counts.get(ind["individual_id"], 0))
        
        # 5. Instability Signal (Native Log-Space penalty for high variance)
        variance = stats.get("variance", 0.0)
        log_instability = -2.0 * variance
        
        # 6. Novelty Signal (Research-Grade Gradient with Clipping)
        if novelty_archive:
            distances = [compute_distance(schema, arch) for arch in novelty_archive]
            min_dist = min(distances)
        else:
            min_dist = 1.0
            
        min_dist = max(0.0, min(1.0, min_dist))  # Bound strictly
        novelty_multiplier = 1.0 + (min_dist ** 1.5)
        
        # Calculate sum in log-space using standard trick log(x + EPSILON)
        log_w = (
            math.log(base_fitness + EPSILON) +
            math.log(conf + EPSILON) +
            math.log(drift_penalty + EPSILON) +
            math.log(freq_penalty + EPSILON) +
            log_instability +
            math.log(novelty_multiplier + EPSILON)
        )
        log_weights.append(log_w)

    # Fallback if population is somehow empty
    if not log_weights:
        return {}, {}
        
    # --- SCALE-INVARIANT SOFTMAX PATTERN ---
    
    # 1. Z-Score Normalization
    mean_lw = sum(log_weights) / len(log_weights)
    variance_lw = sum((x - mean_lw)**2 for x in log_weights) / len(log_weights)
    std_lw = math.sqrt(variance_lw)
    
    if std_lw < EPSILON:
        # Degenerate Distribution Guard (Population Converged)
        # Prevent division instability and false exploration bursts.
        # Fall back to uniform selection (controlled diffusion).
        norm_log_weights = [0.0 for _ in log_weights]
    else:
        norm_log_weights = [(x - mean_lw) / std_lw for x in log_weights]

    # 2. Temperature Scaling & Max Subtraction (Log-Softmax)
    max_log = max(norm_log_weights)
    # T=1.0 is normal, T>1 flattens distribution, T<1 sharpens it
    T = max(0.1, selection_temperature) 
    exp_weights = [math.exp((lw - max_log) / T) for lw in norm_log_weights]
    
    sum_exp = sum(exp_weights)
    probs = [w / sum_exp for w in exp_weights]

    parent_a, parent_b = random.choices(population, weights=probs, k=2)
    
    return parent_a, parent_b

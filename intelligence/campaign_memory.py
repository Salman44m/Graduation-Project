"""Phase 4c: Campaign Memory Manager.

Provides structured, bounded, and decay-managed memory for strategies.
Memory acts as an advisory system, tracking contradiction and drift
without forcing hard bias on the GA.
"""

import math
import logging
from typing import TypedDict, Any

logger = logging.getLogger("agents.memory")

# The maximum number of personas/strategies tracked in memory.
_MAX_STRATEGIES = 20
# Decay applied per generation of non-use.
_DECAY_RATE = 0.05
# Minimum confidence before a strategy is evicted.
_MIN_CONFIDENCE = 0.1

class StrategyMemoryStats(TypedDict, total=False):
    persona: str
    total_attempts: int
    success_count: int
    failure_count: int
    consecutive_failures: int
    sum_fitness: float
    sum_sq_fitness: float
    variance: float
    confidence: float
    last_generation_used: int
    is_drift_suspected: bool
    known_failure_modes: set[str]


def update_strategy_memory(
    current_generation: int,
    strategy_stats: dict[str, StrategyMemoryStats],
    evaluated_individuals: list[dict[str, Any]]
) -> dict[str, StrategyMemoryStats]:
    """
    Update the campaign memory with the results of the latest generation.
    Returns the updated (and potentially pruned) dictionary of strategy stats.
    """
    updated_stats = dict(strategy_stats)

    # 1. Update stats for all individuals in the current generation
    active_personas = set()
    for ind in evaluated_individuals:
        strategy = ind.get("strategy", {})
        persona = strategy.get("persona")
        if not persona:
            continue
            
        persona = persona.lower().strip()
        active_personas.add(persona)
        
        fitness = ind.get("fitness_score", 0.0)
        failure_mode = ind.get("failure_classification", {}).get("failure_mode")
        
        if persona not in updated_stats:
            updated_stats[persona] = {
                "persona": persona,
                "total_attempts": 0,
                "success_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "sum_fitness": 0.0,
                "sum_sq_fitness": 0.0,
                "variance": 0.0,
                "confidence": 1.0,
                "last_generation_used": current_generation,
                "is_drift_suspected": False,
                "known_failure_modes": set()
            }
            
        stats = updated_stats[persona]
        stats["total_attempts"] += 1
        stats["sum_fitness"] += fitness
        stats["sum_sq_fitness"] += fitness ** 2
        stats["last_generation_used"] = current_generation
        
        if failure_mode and failure_mode != "unknown":
            stats["known_failure_modes"].add(failure_mode)

        # Basic success/failure heuristic (fitness >= 0.7 is a strong success)
        if fitness >= 0.7:
            stats["success_count"] += 1
            stats["consecutive_failures"] = 0
            stats["is_drift_suspected"] = False
        else:
            stats["failure_count"] += 1
            stats["consecutive_failures"] += 1
            
        # Drift Detection (Advisory Flag Only)
        if stats["consecutive_failures"] >= 3 and stats["success_count"] > 0:
            stats["is_drift_suspected"] = True
            
        # Variance Tracking (Contradiction indicator)
        if stats["total_attempts"] > 1:
            mean = stats["sum_fitness"] / stats["total_attempts"]
            variance = (stats["sum_sq_fitness"] / stats["total_attempts"]) - (mean ** 2)
            stats["variance"] = max(0.0, variance)

    # 2. Apply Decay and Conflict Scaling
    keys_to_remove = []
    for persona, stats in updated_stats.items():
        # Only decay strategies not used in this generation
        if persona not in active_personas:
            gens_unused = current_generation - stats["last_generation_used"]
            if gens_unused > 0:
                stats["confidence"] = max(0.0, stats["confidence"] - _DECAY_RATE)
        else:
            # If used, scale confidence based on high variance (contradiction)
            # High variance (e.g. > 0.2) means the strategy is flaky.
            if stats["variance"] > 0.15:
                # Penalty scales with variance, max ~20% confidence hit per gen
                penalty = min(0.2, stats["variance"])
                stats["confidence"] = max(0.0, stats["confidence"] - penalty)
                
        # Evict dead strategies
        if stats["confidence"] < _MIN_CONFIDENCE:
            keys_to_remove.append(persona)

    for k in keys_to_remove:
        del updated_stats[k]
        
    # 3. Enforce Bounded Memory (LRU or lowest confidence eviction if we exceed max)
    if len(updated_stats) > _MAX_STRATEGIES:
        # Sort by confidence (descending), then keep top N
        sorted_personas = sorted(
            updated_stats.keys(), 
            key=lambda p: updated_stats[p]["confidence"], 
            reverse=True
        )
        for p in sorted_personas[_MAX_STRATEGIES:]:
            del updated_stats[p]

    return updated_stats

"""Adaptive Evolutionary Controller (Two-Timescale RL-like Control System)."""

import collections

class AdaptiveExplorationController:
    """
    Manages the 'exploration_rate' (mutation intensity) and 
    'selection_temperature' (softmax greediness) dynamically based on 
    population entropy and fitness stagnation.
    """
    def __init__(self, 
                 base_exploration: float = 0.05, 
                 base_temperature: float = 1.0,
                 max_exploration: float = 0.50,
                 max_temperature: float = 2.0,
                 history_window: int = 3):
        self.exploration_rate = base_exploration
        self.selection_temperature = base_temperature
        self.base_exploration = base_exploration
        self.base_temperature = base_temperature
        self.max_exploration = max_exploration
        self.max_temperature = max_temperature
        
        self.best_fitness_history = collections.deque(maxlen=history_window)
        self.history_window = history_window

    def update_state(self, current_best_fitness: float, population_hashes: list[str]) -> None:
        """
        Evaluated post-generation. Updates control variables.
        """
        # 1. Update History
        self.best_fitness_history.append(current_best_fitness)
        
        # 2. Measure Diversity (Hash Concentration)
        if population_hashes:
            counts = collections.Counter(population_hashes)
            max_hash_frequency = max(counts.values()) / len(population_hashes)
        else:
            max_hash_frequency = 1.0
            
        # Channel A: Diversity -> Mutation Intensity
        if max_hash_frequency > 0.4:
            # Population is collapsing into clones. Inject chaos.
            self.exploration_rate = min(self.max_exploration, self.exploration_rate + 0.10)
        else:
            # Decay towards base
            self.exploration_rate = max(self.base_exploration, self.exploration_rate - 0.02)
            
        # Channel B: Stagnation -> Selection Greediness
        if len(self.best_fitness_history) == self.history_window:
            history_max = max(self.best_fitness_history)
            history_min = min(self.best_fitness_history)
            
            if (history_max - history_min) < 0.05:
                # Stagnant. Flatten selection to allow escape.
                self.selection_temperature = min(self.max_temperature, self.selection_temperature + 0.20)
            else:
                # Improving. Decay towards greedy (base).
                self.selection_temperature = max(self.base_temperature, self.selection_temperature - 0.10)
        else:
            # Not enough history yet. Decay towards greedy.
            self.selection_temperature = max(self.base_temperature, self.selection_temperature - 0.10)

    def get_state_dict(self) -> dict:
        return {
            "exploration_rate": self.exploration_rate,
            "selection_temperature": self.selection_temperature,
            "best_fitness_history": list(self.best_fitness_history)
        }

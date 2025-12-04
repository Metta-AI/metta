"""
Prioritized Regret Curriculum Algorithm.

Implements ACCEL-inspired regret-based task selection:
"Go to tasks with highest regret" - prioritize tasks where agent is furthest from optimal.

Based on: "Adversarially Compounding Complexity by Editing Levels"
Reference: https://accelagent.github.io/
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .regret_tracker import RegretTracker


class PrioritizedRegretConfig(CurriculumAlgorithmConfig):
    """Configuration for prioritized regret curriculum."""

    type: str = "prioritized_regret"

    # Regret tracking parameters
    optimal_value: float = 1.0  # Maximum achievable score
    regret_ema_timescale: float = 0.01  # EMA decay rate for regret

    # Task selection parameters
    exploration_bonus: float = 0.1  # Bonus for unexplored tasks
    temperature: float = 1.0  # Softmax temperature for task sampling (higher = more random)
    min_samples_for_prioritization: int = 2  # Min samples before using regret

    # Memory management
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False

    def algorithm_type(self) -> str:
        return "prioritized_regret"

    def create(self, num_tasks: int) -> "PrioritizedRegretAlgorithm":
        return PrioritizedRegretAlgorithm(num_tasks, self)


class PrioritizedRegretAlgorithm(CurriculumAlgorithm):
    """
    Prioritized Regret Curriculum Algorithm.

    Selects tasks based on their regret (gap between optimal and achieved performance).
    Tasks with higher regret are sampled more frequently, following the ACCEL principle
    of maintaining tasks at the frontier of agent capabilities.

    Strategy: "Go to tasks with highest regret"
    """

    def __init__(self, num_tasks: int, hypers: PrioritizedRegretConfig):
        super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers: PrioritizedRegretConfig = hypers

        # Initialize regret tracker
        self.regret_tracker = RegretTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            optimal_value=hypers.optimal_value,
            regret_ema_timescale=hypers.regret_ema_timescale,
        )

        # Cache for task scores
        self._score_cache: Dict[int, float] = {}
        self._cache_valid_tasks: set[int] = set()

        # Stats cache
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks by their regret (higher regret = higher priority)."""
        scores = {}
        for task_id in task_ids:
            scores[task_id] = self._get_regret_score(task_id)
        return scores

    def _get_regret_score(self, task_id: int) -> float:
        """Calculate regret-based score for a task.

        Returns:
            Higher score for higher regret (tasks where agent is far from optimal)
        """
        # Return cached score if valid
        if task_id in self._cache_valid_tasks and task_id in self._score_cache:
            return self._score_cache[task_id]

        task_stats = self.regret_tracker.get_task_stats(task_id)

        if not task_stats or task_stats["completion_count"] < self.hypers.min_samples_for_prioritization:
            # New/under-sampled tasks get exploration bonus
            score = self.hypers.exploration_bonus
        else:
            # Score is directly proportional to regret
            # Higher regret = higher priority
            regret = task_stats["ema_regret"]

            # Apply temperature scaling for softmax sampling
            score = regret / max(self.hypers.temperature, 0.01)

            # Add small exploration bonus for diversity
            if task_stats["completion_count"] < 10:
                exploration_factor = (10 - task_stats["completion_count"]) / 10
                score += self.hypers.exploration_bonus * exploration_factor

        # Cache the computed score
        self._score_cache[task_id] = score
        self._cache_valid_tasks.add(task_id)
        return score

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict based on regret.

        Evict tasks with lowest regret (closest to optimal, i.e., "solved" tasks).
        """
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids)

        # Find task with minimum regret (lowest score)
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, float("inf")))
        return min_task_id

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted.

        Evict tasks that:
        1. Have been presented enough times
        2. Have low regret compared to other tasks (close to solved)
        """
        # First check if task has enough presentations
        task_stats = self.regret_tracker.get_task_stats(task_id)
        if task_stats is None:
            return False

        if task_stats["completion_count"] < min_presentations:
            return False

        # Check if this task has low regret compared to others
        all_task_ids = self.regret_tracker.get_all_tracked_tasks()
        if len(all_task_ids) <= 1:
            return False

        scores = self.score_tasks(all_task_ids)
        task_score = scores.get(task_id, 0.0)

        # Evict if task is in the bottom 30% of regret scores
        # (these are the tasks closest to being solved)
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * 0.3))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        self.regret_tracker.remove_task(task_id)
        self._score_cache.pop(task_id, None)
        self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task regret based on performance."""
        self.regret_tracker.update_task_performance(task_id, score)

        # Invalidate cache for this task
        self._cache_valid_tasks.discard(task_id)
        self.invalidate_cache()

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation by tracking it."""
        self.regret_tracker.track_task_creation(task._task_id)

        # Extract and update slice values if available
        slice_values = task.get_slice_values()
        if slice_values:
            # Initial tracking with neutral score
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        self.invalidate_cache()

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        """Update task performance including slice values for analysis."""
        # First update performance
        self.update_task_performance(task_id, score)

        # Then update slice analyzer
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics for logging."""
        base_stats = {"num_tasks": self.num_tasks, **self.slice_analyzer.get_base_stats()}

        # Add regret tracker stats
        regret_stats = self.regret_tracker.get_global_stats()
        for key, value in regret_stats.items():
            base_stats[f"regret/{key}"] = value

        return base_stats

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix."""
        cache_key = prefix if prefix else "_default"

        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Get base stats
        stats = self.get_base_stats()

        # Add detailed regret statistics
        detailed_regret_stats = self._get_detailed_regret_stats()
        for key, value in detailed_regret_stats.items():
            stats[f"regret/{key}"] = value

        if self.enable_detailed_logging:
            detailed = self.get_detailed_stats()
            stats.update(detailed)

        # Add prefix to all keys
        if prefix:
            stats = {f"{prefix}{k}": v for k, v in stats.items()}

        # Cache result
        self._stats_cache[cache_key] = stats
        self._stats_cache_valid = True

        return stats

    def _get_detailed_regret_stats(self) -> Dict[str, float]:
        """Get detailed regret statistics."""
        all_tasks = self.regret_tracker.get_all_tracked_tasks()
        if not all_tasks:
            return {
                "num_high_regret_tasks": 0.0,
                "num_low_regret_tasks": 0.0,
                "regret_std": 0.0,
            }

        regrets = []
        for task_id in all_tasks:
            task_stats = self.regret_tracker.get_task_stats(task_id)
            if task_stats:
                regrets.append(task_stats["ema_regret"])

        if not regrets:
            return {
                "num_high_regret_tasks": 0.0,
                "num_low_regret_tasks": 0.0,
                "regret_std": 0.0,
            }

        regrets_array = np.array(regrets)
        median_regret = np.median(regrets_array)

        return {
            "num_high_regret_tasks": float(np.sum(regrets_array > median_regret)),
            "num_low_regret_tasks": float(np.sum(regrets_array <= median_regret)),
            "regret_std": float(np.std(regrets_array)),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for checkpointing."""
        return {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "regret_tracker": self.regret_tracker.get_state(),
            "score_cache": self._score_cache,
            "cache_valid_tasks": list(self._cache_valid_tasks),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load algorithm state from checkpoint."""
        self.regret_tracker.load_state(state["regret_tracker"])
        self._score_cache = state.get("score_cache", {})
        self._cache_valid_tasks = set(state.get("cache_valid_tasks", []))
        self._stats_cache_valid = False


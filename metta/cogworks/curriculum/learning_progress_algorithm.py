"""
Learning Progress Algorithm with integrated bidirectional scoring.

Provides intelligent task selection based on bidirectional learning progress analysis,
using fast and slow exponential moving averages to detect learning opportunities.
"""

import random
from typing import Any, Dict, List, Optional

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .lp_scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from .stats import CacheCoordinator, LPStatsAggregator
from .task_tracker import TaskTracker

# Constants for bidirectional learning progress
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    type: str = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    ema_timescale: float = 0.1  # EMA learning rate (0.1 = updates in ~10 samples)
    slow_timescale_factor: float = 0.2  # Multiplier for slow EMA timescale (slow = ema_timescale * this)
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.01  # For bidirectional reweighting
    performance_bonus_weight: float = 0.0  # Weight for performance bonus in LP calculation

    # Task distribution and sampling
    num_active_tasks: int = 10000
    rand_task_rate: float = 0.01
    sample_threshold: int = 10
    memory: int = 25
    eviction_threshold_percentile: float = 0.4  # Bottom percentile for task eviction

    # Basic EMA mode parameters (when use_bidirectional=False)
    basic_ema_initial_alpha: float = 0.3  # Initial learning rate for basic EMA
    basic_ema_alpha_decay: float = 0.2  # Decay factor for basic EMA alpha
    exploration_blend_factor: float = 0.5  # Blend factor for exploration in basic mode

    # Task tracker EMA configuration
    task_tracker_ema_alpha: float = 0.02  # Learning rate for task tracker EMAs (reward, success rate)

    # Performance and memory management
    max_memory_tasks: int = 1000
    max_slice_axes: int = 3  # Updated terminology

    # Memory backend configuration
    task_struct_size: int = 13  # Size of task data structure in shared memory (includes ema_squared)
    completion_history_size: int = 1000  # Size of completion history array
    enable_detailed_slice_logging: bool = False  # Updated terminology
    use_shared_memory: bool = True  # Enabled by default for production use
    session_id: Optional[str] = None  # Session ID for shared memory, None = auto-generate unique

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning Progress Algorithm with integrated bidirectional scoring.

    Uses bidirectional learning progress by default, combining fast and slow
    exponential moving averages to detect learning opportunities and guide
    intelligent task selection.
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        super().__init__(num_tasks, hypers)

        self.num_tasks = num_tasks
        self.hypers: LearningProgressConfig = hypers

        # Initialize task tracker (unified implementation with configurable backend)
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            ema_alpha=hypers.task_tracker_ema_alpha,
            session_id=hypers.session_id if hypers.use_shared_memory else None,
            use_shared_memory=hypers.use_shared_memory,
            task_struct_size=hypers.task_struct_size,
            completion_history_size=hypers.completion_history_size,
        )

        # Note: slice_analyzer is already initialized in parent class via StatsLogger

        # Initialize scorer strategy
        self.scorer: LPScorer = BidirectionalLPScorer(hypers) if hypers.use_bidirectional else BasicLPScorer(hypers)

        # Initialize stats aggregator to centralize stats computation
        self.stats_aggregator = LPStatsAggregator(
            task_tracker=self.task_tracker,
            scorer=self.scorer,
            slice_analyzer=self.slice_analyzer,
            num_tasks=num_tasks,
        )

        # Initialize cache coordinator to centralize cache invalidation
        self.cache_coordinator = CacheCoordinator(
            stats_logger=self,
            scorer=self.scorer,
            slice_analyzer=self.slice_analyzer,
        )

        # Cache for expensive statistics computation
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_valid = False

        # Track task labels for pool composition and sampling stats
        self._task_labels: Dict[int, str] = {}  # task_id -> label
        self._label_completion_counts: Dict[str, int] = {}  # label -> count

    @property
    def lp_scorer(self):
        """Compatibility property for tests that expect lp_scorer attribute."""
        return self

    @property
    def exploration_bonus(self):
        """Compatibility property for tests that expect exploration_bonus attribute."""
        return self.hypers.exploration_bonus

    @property
    def _cache_valid_tasks(self):
        """Compatibility property for tests that access scorer's cache."""
        return self.scorer._cache_valid_tasks

    @property
    def _score_cache(self):
        """Compatibility property for tests that access scorer's cache."""
        return self.scorer._score_cache

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix. Always includes learning progress stats."""
        cache_key = prefix if prefix else "_default"

        if self._stats_cache_valid and cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        # Get base stats (required)
        stats = self.get_base_stats()

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

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default)."""
        # NEW: Use scorer strategy instead of conditionals
        return {task_id: self.scorer.score_task(task_id, self.task_tracker) for task_id in task_ids}

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend which task to evict based on learning progress."""
        if not task_ids:
            return None

        scores = self.score_tasks(task_ids)

        # Find task with minimum learning progress
        min_task_id = min(task_ids, key=lambda tid: scores.get(tid, 0.0))
        return min_task_id

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if a task should be evicted based on criteria."""
        # First check if task has enough presentations
        task_stats = self.task_tracker.get_task_stats(task_id)
        if task_stats is None:
            return False

        if task_stats["completion_count"] < min_presentations:
            return False

        # Check if this task has low learning progress compared to others
        all_task_ids = self.task_tracker.get_all_tracked_tasks()
        if len(all_task_ids) <= 1:
            return False

        scores = self.score_tasks(all_task_ids)
        task_score = scores.get(task_id, 0.0)

        # Evict if this task is in the bottom N% of learning progress scores
        # This ensures eviction happens more readily with small task pools
        sorted_scores = sorted(scores.values())
        threshold_index = max(0, int(len(sorted_scores) * self.hypers.eviction_threshold_percentile))
        threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

        return task_score <= threshold_score

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when a task is evicted."""
        # Remove from task tracker (handles its own locking)
        self.task_tracker.remove_task(task_id)

        # Learning progress specific cleanup
        self._remove_task_from_scoring(task_id)

        # Remove from label tracking
        self._task_labels.pop(task_id, None)

        # Invalidate stats cache when task state changes
        self.cache_coordinator.invalidate_stats_cache()

    def _remove_task_from_scoring(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self.scorer.remove_task(task_id)

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance using the scorer strategy."""
        # NEW: Update scorer's internal state
        self.scorer.update_with_score(task_id, score)

        # NEW: Calculate LP score from scorer
        lp_score = self.scorer.score_task(task_id, self.task_tracker)

        # Single atomic update to task tracker with both score and LP score
        # This ensures consistency and avoids multiple writes to shared memory
        self.task_tracker.update_task_performance(task_id, score, lp_score=lp_score)

        # Track completion counts by label
        if task_id in self._task_labels:
            label = self._task_labels[task_id]
            self._label_completion_counts[label] = self._label_completion_counts.get(label, 0) + 1

        # Invalidate stats cache when task performance changes
        self.cache_coordinator.invalidate_stats_cache()

    def _choose_task_from_list(self, task_ids: List[int]) -> int:
        """Choose a task from the provided list based on scores."""
        if not task_ids:
            raise ValueError("Cannot choose from empty task list")

        scores = self.score_tasks(task_ids)
        if not scores:
            return random.choice(task_ids)

        # Convert scores to probabilities for sampling
        total_score = sum(scores.values())
        if total_score <= 0:
            return random.choice(task_ids)

        # Create weighted probability distribution
        weights = [scores.get(task_id, 0.0) for task_id in task_ids]
        return random.choices(task_ids, weights=weights)[0]

    def get_learning_progress_score(self, task_id: int, task_tracker=None) -> float:
        """Get learning progress score for a specific task (compatibility method for tests)."""
        # NEW: Use scorer strategy
        return self.scorer.score_task(task_id, self.task_tracker)

    def get_stats(self) -> Dict[str, float]:
        """Get learning progress statistics (compatibility method for tests)."""
        return self.scorer.get_stats()

    def update_task_with_slice_values(self, task_id: int, score: float, slice_values: Dict[str, Any]) -> None:
        """Update task performance including slice values for analysis."""
        # First update performance
        self.update_task_performance(task_id, score)

        # Then update slice analyzer
        self.slice_analyzer.update_task_completion(task_id, slice_values, score)

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle task creation by tracking it."""
        self.task_tracker.track_task_creation(task._task_id)

        # Track task label for pool composition stats
        if hasattr(task, "get_label"):
            label = task.get_label()
            if label:
                self._task_labels[task._task_id] = label

        # Extract and update slice values if available
        slice_values = task.get_slice_values()
        if slice_values:
            # Initial tracking with neutral score
            self.slice_analyzer.update_task_completion(task._task_id, slice_values, 0.5)

        # Invalidate stats cache when task state changes
        self.cache_coordinator.invalidate_stats_cache()

    def get_pool_composition_stats(self) -> Dict[str, Dict[str, int]]:
        """Get pool composition and sampling statistics by label.

        Returns:
            Dictionary with 'pool_composition' and 'sampling_counts' keys,
            each containing label->count mappings.
        """
        # Count labels currently in pool
        pool_composition = {}
        for label in self._task_labels.values():
            pool_composition[label] = pool_composition.get(label, 0) + 1

        return {
            "pool_composition": pool_composition,
            "sampling_counts": self._label_completion_counts.copy(),
        }

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide."""
        stats = self.stats_aggregator.get_base_stats()

        # Add pool composition stats (logged every epoch)
        composition_data = self.get_pool_composition_stats()

        # Add pool composition (number of each label in memory)
        for label, count in composition_data["pool_composition"].items():
            stats[f"pool_composition/{label}"] = float(count)

        # Add sampling counts (number of times each label was sampled)
        for label, count in composition_data["sampling_counts"].items():
            stats[f"sampling_counts/{label}"] = float(count)

        return stats

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats including learning progress and slice distribution analysis."""
        return self.stats_aggregator.get_detailed_stats()

    def get_state(self) -> Dict[str, Any]:
        """Get learning progress algorithm state for checkpointing."""
        return {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "task_tracker": self.task_tracker.get_state(),
            "scorer": self.scorer.get_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learning progress algorithm state from checkpoint."""
        # Restore task tracker
        self.task_tracker.load_state(state["task_tracker"])

        # Restore scorer state
        if "scorer" in state:
            self.scorer.load_state(state["scorer"])

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources with better error handling."""
        if not hasattr(self, "task_tracker"):
            return

        try:
            # TaskTracker always has cleanup_shared_memory method
            self.task_tracker.cleanup_shared_memory()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            import logging

            logging.warning(f"Failed to cleanup shared memory: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "hypers") and getattr(self.hypers, "use_shared_memory", False):
            self.cleanup_shared_memory()

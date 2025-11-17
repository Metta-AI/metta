"""Learning progress curriculum algorithm with dual-pool exploration-exploitation.

This module implements the Learning Progress (LP) algorithm - a curriculum learning approach
that prioritizes tasks where the agent is learning fastest. It tracks fast/slow EMAs of task
performance to identify learning opportunities, then samples tasks proportionally to their
learning progress scores.

Core Algorithm (Single-Pool):
    1. Track task performance using exponential moving averages (fast and slow)
    2. Compute learning progress as the rate of performance change (|fast - slow|)
    3. Apply exploration bonus to under-sampled tasks
    4. Transform scores with z-score normalization and sigmoid for sampling probabilities
    5. Sample tasks proportionally to their scores (high LP → more likely to be selected)

Dual-Pool Architecture (Optional):
    Extends single-pool LP with exploration-exploitation balance:

    Pools:
        - Explore pool (N=50): High-turnover pool for discovering new learning opportunities
        - Exploit pool (N=200): Selective pool for tasks with proven learning progress

    Phases:
        1. Bootstrap: 100% exploration until exploit pool fills
        2. Steady-state: Adaptive sampling based on Explore-Exploit Ratio (EER/ρ)

    Promotion:
        - Tasks reaching S_min samples become eligible for promotion
        - Only tasks with positive LP scores are promoted
        - Exploit pool evicts lowest-scoring tasks when full
        - Explore pool backfilled with new random tasks

    Adaptive EER:
        - ρ adapts based on promotion success rate via exponential moving average
        - High promotion success → increase ρ (more exploration)
        - Low promotion success → decrease ρ (more exploitation)
        - Bounded by ρ_min and ρ_max to ensure both pools are sampled

Key Components:
    - LearningProgressConfig: Comprehensive configuration with sensible defaults
    - LearningProgressAlgorithm: Main algorithm coordinating scorer, tracker, and stats
    - DualPoolTaskTracker: Manages two independent task pools with atomic promotion
    - LPScorer: Strategy pattern for bidirectional/basic LP scoring

Design Philosophy:
    - All state (EMAs, counts, EER) lives in shared memory for true multi-process training
    - Strategy pattern for scoring allows swapping between bidirectional/basic/custom
    - Stateless algorithms make checkpointing and debugging straightforward
    - Dual-pool is optional and controlled by configuration (no code changes needed)

Configuration Helpers:
    - LearningProgressConfig.default(): Balanced single-pool config
    - LearningProgressConfig.stable(): Stable config for noisy environments
    - LearningProgressConfig.default_dual_pool(): Production dual-pool config

Why Separate File:
    This is a complex algorithm (1300+ lines) with many configuration options and moving
    parts. Keeping it separate from the simple DiscreteRandomCurriculum maintains clarity
    and allows the two approaches to evolve independently.

See Also:
    - task_tracker.py: TaskTracker and DualPoolTaskTracker for performance tracking
    - lp_scorers.py: Scoring strategies (bidirectional/basic LP)
    - curriculum.py: Main Curriculum class using this algorithm
"""

import random
import uuid
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import model_validator

from .curriculum_base import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .lp_scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from .task_tracker import DualPoolTaskTracker, TaskTracker


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for learning progress with bidirectional scoring as default."""

    type: Literal["learning_progress"] = "learning_progress"

    # Bidirectional learning progress settings (now default)
    use_bidirectional: bool = True
    use_baseline_normalization: bool = (
        True  # Normalize by baseline to get "mastery" score p_i = (TSR_i - B_i) / (1.0 - B_i)
    )
    # EMA Timescale: Controls convergence speed of fast EMA
    # - 0.1 (default): Converges in ~10 samples, responsive to recent changes
    # - 0.01-0.05: Slower convergence, more stable for noisy environments
    # - 0.001: Very slow (1000+ samples), delays LP signal but maximum stability
    # Lower values delay learning progress signal development - Gini may stay near 0
    ema_timescale: float = 0.1
    slow_timescale_factor: float = 0.2  # Multiplier for slow EMA timescale (slow = ema_timescale * this)
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.0  # For bidirectional reweighting (set to 0 to avoid artificial floor)
    performance_bonus_weight: float = 0.0  # Weight for performance bonus in LP calculation
    lp_score_temperature: float = 0.0  # Temperature for rescaling LP scores before sigmoid
    # Special values for lp_score_temperature:
    # - > 0: Divide LP by temperature (low temp amplifies differences)
    # - = 0: Apply z-score normalization (standardize to mean=0, std=1) before sigmoid (DEFAULT)
    #        This centers LP scores and makes sigmoid more sensitive to relative differences
    z_score_amplification: float = 10.0  # Amplification factor after z-score normalization
    # Only applies when lp_score_temperature = 0 (z-score mode). Higher values increase selectivity
    # by spreading out the z-scored distribution before sigmoid. Default 10.0 provides strong selectivity
    # while maintaining z-score's scale-invariance. Set to 1.0 for no amplification (uniform sampling).
    early_progress_amplification: float = 0.5  # Reweight performance signals before LP calculation
    # Note: 0.5 is effectively OFF (R(p) ≈ p). Low values (e.g., 0.05) amplify signal from
    # unsolved tasks (p~0) and dampen signal from partially-solved tasks (p~0.5).
    # High values would reweight toward higher performance tasks.

    # Task distribution and sampling
    num_active_tasks: int = 1000
    rand_task_rate: float = 0.01  # Reduced from 0.25 in refactor for better curriculum learning
    sample_threshold: int = 10
    memory: int = 25
    eviction_threshold_percentile: float = 0.4  # Bottom percentile for task eviction

    # Memory management for label tracking
    max_inactive_labels_retained: int = 100  # Max inactive labels to keep for historical stats (prevents memory leak)

    # Basic EMA mode parameters (when use_bidirectional=False)
    basic_ema_initial_alpha: float = 0.3  # Initial learning rate for basic EMA
    basic_ema_alpha_decay: float = 0.2  # Decay factor for basic EMA alpha
    min_samples_for_lp: int = 10  # Minimum samples before using LP score (use exploration bonus until then)

    # Task tracker EMA configuration
    task_tracker_ema_alpha: float = 0.02  # Learning rate for task tracker EMAs (reward, success rate)

    # Task creation defaults
    task_default_success_threshold: float = 0.5  # Default success threshold for new tasks
    task_default_generator_type: float = 0.0  # Default generator type identifier for tasks

    # Memory backend configuration
    task_struct_size: int = 18  # Size of task data structure in shared memory (17 metrics + label_hash)
    use_shared_memory: bool = True  # Enabled by default for production use
    session_id: Optional[str] = None  # Session ID for shared memory, None = auto-generate shared ID

    # Logging configuration
    show_curriculum_troubleshooting_logging: bool = False  # Show high-cardinality per-task metrics for debugging

    # ========== Dual-Pool Architecture Configuration ==========
    # Enable dual-pool exploration-exploitation with separate explore and exploit pools
    use_dual_pool: bool = False  # Toggle between single-pool (default) and dual-pool architecture

    # Pool sizes (only used when use_dual_pool=True)
    num_explore_tasks: int = 50  # Exploration pool capacity
    num_exploit_tasks: int = 200  # Exploitation pool capacity

    # Promotion criteria (only used when use_dual_pool=True)
    promotion_min_samples: int = 5  # Minimum samples before task is eligible for promotion

    # Adaptive Explore-Exploit Ratio (EER) parameters (only used when use_dual_pool=True)
    explore_exploit_ratio_init: float = 0.5  # Initial EER value (probability of sampling from explore pool)
    explore_exploit_ratio_min: float = 0.05  # Minimum EER bound
    explore_exploit_ratio_max: float = 0.95  # Maximum EER bound
    explore_exploit_ratio_alpha: float = 0.9  # EER learning rate (higher = slower adaptation)
    promotion_rate_window: int = 1000  # Sliding window size for promotion rate calculation

    @model_validator(mode="after")
    def _validate_and_initialize(self) -> "LearningProgressConfig":
        """Validate configuration and initialize derived parameters.

        This ensures:
        1. Session ID is generated when using shared memory
        2. Dual-pool parameters are validated when dual-pool is enabled
        3. EER bounds are sensible
        """
        # Generate session ID for shared memory if not provided
        if self.use_shared_memory and self.session_id is None:
            # Generate a unique session ID that will be shared across processes
            # This happens once at config creation time, before pickling
            self.session_id = f"lp_{uuid.uuid4().hex[:8]}"

        # Validate dual-pool configuration
        if self.use_dual_pool:
            # Validate pool sizes
            if self.num_explore_tasks <= 0:
                raise ValueError(f"num_explore_tasks must be positive, got {self.num_explore_tasks}")
            if self.num_exploit_tasks <= 0:
                raise ValueError(f"num_exploit_tasks must be positive, got {self.num_exploit_tasks}")

            # Validate promotion parameters
            if self.promotion_min_samples <= 0:
                raise ValueError(f"promotion_min_samples must be positive, got {self.promotion_min_samples}")

            # Validate EER bounds
            if not 0.0 <= self.explore_exploit_ratio_min <= 1.0:
                raise ValueError(f"explore_exploit_ratio_min must be in [0, 1], got {self.explore_exploit_ratio_min}")
            if not 0.0 <= self.explore_exploit_ratio_max <= 1.0:
                raise ValueError(f"explore_exploit_ratio_max must be in [0, 1], got {self.explore_exploit_ratio_max}")
            if self.explore_exploit_ratio_min >= self.explore_exploit_ratio_max:
                raise ValueError(
                    f"explore_exploit_ratio_min ({self.explore_exploit_ratio_min}) must be "
                    f"< explore_exploit_ratio_max ({self.explore_exploit_ratio_max})"
                )
            if not 0.0 <= self.explore_exploit_ratio_init <= 1.0:
                raise ValueError(f"explore_exploit_ratio_init must be in [0, 1], got {self.explore_exploit_ratio_init}")

            # Validate EER learning rate
            if not 0.0 < self.explore_exploit_ratio_alpha < 1.0:
                raise ValueError(
                    f"explore_exploit_ratio_alpha must be in (0, 1), got {self.explore_exploit_ratio_alpha}"
                )

            # Validate promotion rate window
            if self.promotion_rate_window <= 0:
                raise ValueError(f"promotion_rate_window must be positive, got {self.promotion_rate_window}")

            # Override num_active_tasks to be sum of pool sizes for consistency
            # Use object.__setattr__ to bypass validation and avoid recursion
            object.__setattr__(self, "num_active_tasks", self.num_explore_tasks + self.num_exploit_tasks)

        return self

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)

    # Configuration Presets for Common Use Cases
    # These provide sensible defaults for different training scenarios

    @classmethod
    def default(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Standard configuration with balanced learning speed.

        Best for: Most RL environments with moderate complexity
        - Bidirectional LP for intelligent task selection
        - Fast EMA convergence (~10 samples)
        - Strong z-score amplification for selectivity

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.1,
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.01,
            "exploration_bonus": 0.1,
            "min_samples_for_lp": 10,
            "lp_score_temperature": 0.0,
            "z_score_amplification": 10.0,
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def stable(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Stable configuration for noisy/stochastic environments.

        Best for: Environments with high variance or randomness
        - Slower EMA convergence for stability (~100 samples)
        - Higher exploration bonus
        - More gradual learning progress signal development

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.01,  # 10x slower
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.02,  # More exploration
            "exploration_bonus": 0.15,  # Higher exploration
            "min_samples_for_lp": 20,  # More samples before LP
            "lp_score_temperature": 0.0,
            "z_score_amplification": 5.0,  # Less aggressive
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def fast_learning(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Fast learning configuration for quickly-adapting agents.

        Best for: Simple environments where agent learns rapidly
        - Very fast EMA convergence (~5 samples)
        - Low exploration bonus (focus on LP)
        - Strong selectivity for high-LP tasks

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.2,  # 2x faster
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.005,  # Less exploration
            "exploration_bonus": 0.05,  # Lower exploration
            "min_samples_for_lp": 5,  # Quick LP signal
            "lp_score_temperature": 0.0,
            "z_score_amplification": 15.0,  # More aggressive
            "show_curriculum_troubleshooting_logging": False,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def arena_legacy(cls, num_active_tasks: int = 256, **overrides) -> "LearningProgressConfig":
        """Legacy arena configuration (from before refactor).

        Best for: Reproducing old arena training runs
        - Very slow EMA for maximum stability (1000+ samples)
        - High z-score amplification

        Args:
            num_active_tasks: Number of tasks to keep in active pool
            **overrides: Override any parameter
        """
        defaults = {
            "use_bidirectional": True,
            "ema_timescale": 0.001,  # Very slow (legacy setting)
            "num_active_tasks": num_active_tasks,
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.01,
            "exploration_bonus": 0.1,
            "min_samples_for_lp": 10,
            "lp_score_temperature": 0.0,
            "z_score_amplification": 10.0,
            "show_curriculum_troubleshooting_logging": True,
            "early_progress_amplification": 0.5,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def default_dual_pool(
        cls,
        num_explore_tasks: int = 50,
        num_exploit_tasks: int = 200,
        **overrides,
    ) -> "LearningProgressConfig":
        """Create a dual-pool configuration with sensible production defaults.

        Best for: Production training with exploration-exploitation balance
        - Dual-pool architecture (explore: 50, exploit: 200)
        - Adaptive EER based on promotion success
        - Stable EMA settings for reliable LP signals
        - Strong selectivity for focusing on high-LP tasks

        Args:
            num_explore_tasks: Size of exploration pool (default: 50)
            num_exploit_tasks: Size of exploitation pool (default: 200)
            **overrides: Override any parameter

        Example:
            # Use defaults
            config = LearningProgressConfig.default_dual_pool()

            # Customize pool sizes
            config = LearningProgressConfig.default_dual_pool(
                num_explore_tasks=100,
                num_exploit_tasks=400,
            )

            # Override other parameters
            config = LearningProgressConfig.default_dual_pool(
                ema_timescale=0.01,
                z_score_amplification=30.0,
            )
        """
        defaults = {
            # Enable dual-pool
            "use_dual_pool": True,
            "num_explore_tasks": num_explore_tasks,
            "num_exploit_tasks": num_exploit_tasks,
            # Production-ready defaults for bidirectional LP
            "use_bidirectional": True,
            "ema_timescale": 0.001,  # Slower EMA for more stable LP signals
            "slow_timescale_factor": 0.2,
            "rand_task_rate": 0.01,
            "exploration_bonus": 0.1,
            "min_samples_for_lp": 10,
            "lp_score_temperature": 0.0,  # Use z-score normalization
            "z_score_amplification": 50.0,  # Strong selectivity
            "early_progress_amplification": 0.5,  # Effectively off
            # Dual-pool specific
            "promotion_min_samples": 5,
            "explore_exploit_ratio_init": 0.5,
            "explore_exploit_ratio_alpha": 0.9,  # Slow EER adaptation
            "promotion_rate_window": 1000,
            # Enable troubleshooting logging for production monitoring
            "show_curriculum_troubleshooting_logging": True,
        }
        defaults.update(overrides)
        return cls(**defaults)


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

        # Initialize task tracker (single-pool or dual-pool based on configuration)
        self.task_tracker: Union[TaskTracker, DualPoolTaskTracker]
        if hypers.use_dual_pool:
            # Dual-pool mode: separate explore and exploit pools
            self.task_tracker = DualPoolTaskTracker(
                num_explore_tasks=hypers.num_explore_tasks,
                num_exploit_tasks=hypers.num_exploit_tasks,
                ema_alpha=hypers.task_tracker_ema_alpha,
                session_id=hypers.session_id if hypers.use_shared_memory else None,
                use_shared_memory=hypers.use_shared_memory,
                task_struct_size=hypers.task_struct_size,
                default_success_threshold=hypers.task_default_success_threshold,
                default_generator_type=hypers.task_default_generator_type,
            )
        else:
            # Single-pool mode: traditional single task tracker
            self.task_tracker = TaskTracker(
                max_memory_tasks=hypers.num_active_tasks,
                ema_alpha=hypers.task_tracker_ema_alpha,
                session_id=hypers.session_id if hypers.use_shared_memory else None,
                use_shared_memory=hypers.use_shared_memory,
                task_struct_size=hypers.task_struct_size,
                default_success_threshold=hypers.task_default_success_threshold,
                default_generator_type=hypers.task_default_generator_type,
            )

        # Initialize scorer strategy (pass tracker for shared memory EMA access)
        # For dual-pool, scorer will work with whichever pool the task is in
        if hypers.use_dual_pool:
            # Create scorer that can access both pools
            # We'll need to pass the appropriate tracker when scoring
            self.scorer: LPScorer = (
                BidirectionalLPScorer(hypers, self.task_tracker.explore_tracker)
                if hypers.use_bidirectional
                else BasicLPScorer(hypers, self.task_tracker.explore_tracker)
            )
        else:
            self.scorer: LPScorer = (
                BidirectionalLPScorer(hypers, self.task_tracker)
                if hypers.use_bidirectional
                else BasicLPScorer(hypers, self.task_tracker)
            )

        # Track label sampling and eviction (labels themselves are in TaskTracker shared memory)
        self._label_sampling_counts: Dict[str, int] = {}  # label -> cumulative sampling count (episodes started)
        self._label_eviction_counts: Dict[str, int] = {}  # label -> eviction count (cumulative)

        # Per-epoch tracking (for gini calculation and epoch-level metrics)
        self._label_evictions_this_epoch: Dict[str, int] = {}  # label -> evictions this epoch
        self._label_sampling_counts_this_epoch: Dict[str, int] = {}  # label -> samples this epoch

        # Track which labels are currently active (have tasks in pool)
        self._active_labels: set[str] = set()

        # Track recently inactive labels to manage memory
        self._inactive_labels_fifo: list[str] = []  # FIFO queue of inactive labels for cleanup

        # Dual-pool specific state (only used when use_dual_pool=True)
        if hypers.use_dual_pool:
            # Explore-Exploit Ratio tracking
            self._explore_exploit_ratio = hypers.explore_exploit_ratio_init
            self._promotion_window: deque[bool] = deque(maxlen=hypers.promotion_rate_window)

            # Cumulative promotion stats
            self._num_promotions = 0
            self._num_promotion_attempts = 0

            # Phase tracking
            self._current_phase: Literal["bootstrap", "steady_state"] = "bootstrap"

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all statistics with optional prefix. Always includes learning progress stats."""
        # Use the StatsLogger implementation
        return super().stats(prefix)

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks using the configured method (bidirectional by default).

        In dual-pool mode, routes scoring to the appropriate pool for each task.
        """
        if isinstance(self.task_tracker, DualPoolTaskTracker):
            # Dual-pool mode: score tasks from their respective pools
            scores = {}
            for task_id in task_ids:
                pool_tracker = self.task_tracker.get_pool_tracker(task_id)
                if pool_tracker is not None:
                    scores[task_id] = self.scorer.score_task(task_id, pool_tracker)
                else:
                    # Task not found, give it lowest score
                    scores[task_id] = 0.0
            return scores
        else:
            # Single-pool mode: standard scoring
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
        # Get label BEFORE removing task (otherwise data is gone)
        evicted_label = self.task_tracker.get_task_label(task_id)

        # Remove from task tracker (handles its own locking)
        self.task_tracker.remove_task(task_id)

        # Learning progress specific cleanup
        self._remove_task_from_scoring(task_id)

        # Track eviction by label
        if evicted_label:
            # Track cumulative eviction count for this label
            self._label_eviction_counts[evicted_label] = self._label_eviction_counts.get(evicted_label, 0) + 1

            # Track per-epoch eviction count (for gini calculation)
            self._label_evictions_this_epoch[evicted_label] = self._label_evictions_this_epoch.get(evicted_label, 0) + 1

            # Check if this label still has any active tasks
            # get_all_tracked_tasks() only returns ACTIVE tasks, so this is safe
            all_active_labels = set()
            for tid in self.task_tracker.get_all_tracked_tasks():
                label = self.task_tracker.get_task_label(tid)
                if label:
                    all_active_labels.add(label)

            if evicted_label not in all_active_labels:
                # No more tasks with this label - remove from active set and track as inactive
                self._active_labels.discard(evicted_label)
                self._inactive_labels_fifo.append(evicted_label)

                # Clean up old inactive labels to prevent memory leak
                self._cleanup_old_inactive_labels()

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def _cleanup_old_inactive_labels(self) -> None:
        """Clean up old inactive labels to prevent unbounded memory growth.

        Keeps only the most recent N inactive labels as specified by
        max_inactive_labels_retained config parameter.
        """
        max_retained = self.hypers.max_inactive_labels_retained

        # Remove old labels if we exceed the limit
        while len(self._inactive_labels_fifo) > max_retained:
            old_label = self._inactive_labels_fifo.pop(0)

            # Only clean up if this label is still inactive (not reactivated)
            if old_label not in self._active_labels:
                # Clean up cumulative stats for this label
                # (completion counts are now in TaskTracker shared memory, no cleanup needed)
                self._label_sampling_counts.pop(old_label, None)
                self._label_eviction_counts.pop(old_label, None)

                # Note: We don't clean up per-epoch counters here as they're reset each epoch anyway

    def _remove_task_from_scoring(self, task_id: int) -> None:
        """Remove task from scoring system."""
        self.scorer.remove_task(task_id)

    def on_task_sampled(self, task_id: int) -> None:
        """Track that a task was sampled (selected for an episode).

        Args:
            task_id: The ID of the task that was sampled
        """
        # Track sampling counts per label (both cumulative and per-epoch)
        label = self.task_tracker.get_task_label(task_id)
        if label:
            self._label_sampling_counts[label] = self._label_sampling_counts.get(label, 0) + 1
            self._label_sampling_counts_this_epoch[label] = self._label_sampling_counts_this_epoch.get(label, 0) + 1

    def get_and_reset_evictions_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch evictions and reset the counter.

        Returns:
            Dictionary mapping label -> eviction count this epoch
        """
        evictions = self._label_evictions_this_epoch.copy()
        self._label_evictions_this_epoch.clear()
        return evictions

    def get_and_reset_sampling_counts_this_epoch(self) -> Dict[str, int]:
        """Get per-epoch sampling counts and reset the counter.

        Returns:
            Dictionary mapping label -> sampling count this epoch
        """
        sampling_counts = self._label_sampling_counts_this_epoch.copy()
        self._label_sampling_counts_this_epoch.clear()
        return sampling_counts

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch counters at the start of a new epoch.

        This is called by the training infrastructure at epoch boundaries
        to ensure per-epoch metrics start fresh.
        """
        self._label_sampling_counts_this_epoch.clear()
        self._label_evictions_this_epoch.clear()

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance atomically.

        Stage 3 Atomic Update: All EMA updates happen in ONE lock acquisition:
        1. Basic EMAs (completion_count, reward_ema, success_rate_ema, ema_squared)
        2. Bidirectional EMAs (p_fast, p_slow, p_true, random_baseline)

        LP score calculation is deferred until sampling time (lazy evaluation via _stale_dist flag).
        This reduces from 4+ lock acquisitions to 1.

        Dual-Pool Mode: After updating performance, check if task is eligible for promotion.
        """
        # Atomic update: All EMAs in one lock
        if isinstance(self.task_tracker, DualPoolTaskTracker):
            # Dual-pool mode: update task in its current pool
            self.task_tracker.update_task_performance(
                task_id=task_id,
                score=score,
                scorer=self.scorer if hasattr(self.scorer, "config") else None,
            )
        else:
            # Single-pool mode: use standard update
            self.task_tracker.update_task_performance_with_bidirectional_emas(
                task_id=task_id,
                score=score,
                scorer=self.scorer if hasattr(self.scorer, "config") else None,
            )

        # Dual-pool: Check for promotion after update
        if self.hypers.use_dual_pool and isinstance(self.task_tracker, DualPoolTaskTracker):
            # Check if task is eligible and attempt promotion
            if self._check_promotion_eligibility(task_id):
                self._num_promotion_attempts += 1
                promoted = self._attempt_promotion(task_id)

                # Record promotion outcome in sliding window
                self._promotion_window.append(promoted)

                # Update EER based on promotion rate
                self._update_explore_exploit_ratio()

                # Update phase (bootstrap -> steady_state)
                self._update_phase()

        # Mark distribution as stale - LP scores will be recalculated on next sampling
        self.scorer.invalidate_cache()

        # Note: Completion counts are now tracked in TaskTracker shared memory
        # No local label tracking needed here

        # Invalidate stats cache when task performance changes
        self.invalidate_cache()

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

    def get_task_lp_score(self, task_id: int) -> float:
        """Get final learning progress score (sampling probability) for a specific task."""
        return self.scorer.score_task(task_id, self.task_tracker)

    def on_task_created(self, task: CurriculumTask, pool: Optional[str] = None) -> None:
        """Handle task creation by tracking it in the appropriate pool.

        Args:
            task: The curriculum task being created
            pool: For dual-pool mode, which pool to create in ('explore' or 'exploit').
                  If None, uses single-pool mode or defaults to 'explore'.
        """
        if isinstance(self.task_tracker, DualPoolTaskTracker):
            # Dual-pool mode: track in specified pool (default to explore)
            target_pool = pool if pool is not None else "explore"
            self.task_tracker.track_task_creation(task._task_id, target_pool)

            # Check if task was actually tracked
            pool_tracker = self.task_tracker.get_pool_tracker(task._task_id)
            if pool_tracker is None:
                # Task wasn't tracked (pool is full), don't add label
                return

            # Initialize LP score to exploration bonus for new tasks
            pool_tracker.update_lp_score(task._task_id, self.hypers.exploration_bonus)
        else:
            # Single-pool mode: standard tracking
            self.task_tracker.track_task_creation(task._task_id)

            # Check if task was actually tracked (might fail if tracker is full)
            if task._task_id not in self.task_tracker._task_id_to_index:
                # Task wasn't tracked (tracker is full), don't add label
                return

            # Initialize LP score to exploration bonus for new tasks
            self.task_tracker.update_lp_score(task._task_id, self.hypers.exploration_bonus)

        # Handle label tracking (common for both modes)
        label = task.get_label()
        if label:
            # Store label in TaskTracker's shared memory
            if isinstance(self.task_tracker, DualPoolTaskTracker):
                self.task_tracker.set_task_label(task._task_id, label)
            else:
                self.task_tracker.set_task_label(task._task_id, label)

            # If label was inactive, remove it from the inactive queue (reactivating it)
            if label in self._inactive_labels_fifo:
                self._inactive_labels_fifo.remove(label)

            self._active_labels.add(label)

        # Invalidate stats cache when task state changes
        self.invalidate_cache()

    def get_pool_composition_stats(self) -> Dict[str, Dict[str, int]]:
        """Get pool composition and sampling statistics by label.

        Returns:
            Dictionary with 'pool_composition' and 'sampling_counts' keys,
            each containing label->count mappings.
        """
        # Count labels currently in pool from TaskTracker shared memory
        pool_composition = {}
        for task_id in self.task_tracker.get_all_tracked_tasks():
            label = self.task_tracker.get_task_label(task_id)
            if label:
                pool_composition[label] = pool_composition.get(label, 0) + 1

        # Return per-epoch sampling counts (reset each epoch)
        return {
            "pool_composition": pool_composition,
            "sampling_counts": self._label_sampling_counts_this_epoch.copy(),
        }

    def get_base_stats(self) -> Dict[str, float]:
        """Get basic statistics that all algorithms must provide.

        Note: Called per-worker in vectorized environments, so keep lightweight.
        Expensive calculations like Gini are in calculate_gini_coefficients().
        """
        # Start with number of tasks
        stats = {
            "num_tasks": self.num_tasks,
        }

        # Add task tracker global stats with prefix
        tracker_stats = self.task_tracker.get_global_stats()
        for key, value in tracker_stats.items():
            stats[f"tracker/{key}"] = value

        # Add pool composition and sampling statistics
        composition_data = self.get_pool_composition_stats()

        for label, count in composition_data["pool_composition"].items():
            stats[f"pool_composition/{label}"] = float(count)

        for label, count in composition_data["sampling_counts"].items():
            stats[f"sampling_counts/{label}"] = float(count)

        for label, count in self._label_eviction_counts.items():
            stats[f"eviction_counts/{label}"] = float(count)

        return stats

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a distribution.

        Measures inequality in sampling/distribution:
        - 0 = perfect equality (all values equal)
        - 1 = perfect inequality (one value has everything)

        Args:
            values: List of counts/frequencies

        Returns:
            Gini coefficient between 0 and 1
        """
        import logging
        import math

        logger = logging.getLogger(__name__)

        if not values or len(values) == 0:
            return 0.0

        # Handle case with all zeros
        total = sum(values)
        if total == 0:
            return 0.0

        # Check for NaN or inf
        if any(math.isnan(v) or math.isinf(v) for v in values):
            logger.warning(f"Gini calculation received NaN or Inf values: {values[:10]}...")
            return 0.0

        # Sort values in ascending order
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini coefficient using the formula:
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        cumsum = 0.0
        for i, value in enumerate(sorted_values, start=1):
            cumsum += i * value

        gini = (2.0 * cumsum) / (n * total) - (n + 1.0) / n

        # Sanity check result
        if math.isnan(gini) or math.isinf(gini):
            logger.error(f"Gini calculation produced NaN/Inf! cumsum={cumsum}, n={n}, total={total}")
            return 0.0

        return gini

    def calculate_gini_coefficients(self) -> Dict[str, float]:
        """Calculate Gini coefficients at each stage of the LP calculation pipeline.

        This is an expensive operation that should be called once per epoch from
        the centralized stats reporter, not from each worker in vectorized environments.

        This helps diagnose where selectivity is lost in the chain:
        1. Raw LP scores (task-level)
        2. Raw LP scores aggregated by label
        3. Z-scored LP scores (task-level)
        4. Final sampling probabilities (task-level)
        5. Sampling counts aggregated by label
        6. Eviction counts aggregated by label
        7. Pool composition aggregated by label

        Returns:
            Dictionary of Gini coefficients at each pipeline stage
        """
        gini_stats = {}

        # Get all tracked tasks
        all_task_ids = self.task_tracker.get_all_tracked_tasks()

        if not all_task_ids:
            return gini_stats

        # Collect task-level data
        completion_counts = []
        raw_lp_scores = []  # Actual raw LP: |p_fast - p_slow|
        z_scored_lp_scores = []
        sampling_probs = []
        task_labels_list = []

        for task_id in all_task_ids:
            task_stats = self.task_tracker.get_task_stats(task_id)
            if task_stats:
                completion_count = float(task_stats["completion_count"])
                completion_counts.append(completion_count)

                # Get final sampling probability (after all transformations)
                sampling_prob = self.scorer.score_task(task_id, self.task_tracker)
                sampling_probs.append(float(sampling_prob))
                z_scored_lp_scores.append(float(sampling_prob))  # Currently same as sampling_prob

                # Get actual raw learning progress: |p_fast - p_slow|
                # This is the true LP signal before smoothing/normalization
                p_fast = float(task_stats.get("p_fast", 0.0))
                p_slow = float(task_stats.get("p_slow", 0.0))
                raw_lp = abs(p_fast - p_slow)
                raw_lp_scores.append(raw_lp)

                label = self.task_tracker.get_task_label(task_id)
                task_labels_list.append(label if label else "unknown")

                if self.hypers.show_curriculum_troubleshooting_logging:
                    gini_stats[f"task_metrics/{task_id}/completion_count"] = completion_count
                    gini_stats[f"task_metrics/{task_id}/raw_lp"] = raw_lp
                    gini_stats[f"task_metrics/{task_id}/sampling_prob"] = sampling_prob

        if completion_counts:
            gini_stats["curriculum_gini/pool_occupancy"] = self._calculate_gini_coefficient(completion_counts)

        if raw_lp_scores:
            gini_stats["curriculum_gini/raw_lp_scores"] = self._calculate_gini_coefficient(raw_lp_scores)
            # Debug stats for raw LP
            import statistics

            gini_stats["debug/raw_lp_mean"] = float(statistics.mean(raw_lp_scores))
            gini_stats["debug/raw_lp_std"] = float(statistics.stdev(raw_lp_scores)) if len(raw_lp_scores) > 1 else 0.0
            gini_stats["debug/raw_lp_max"] = float(max(raw_lp_scores))
            gini_stats["debug/raw_lp_min"] = float(min(raw_lp_scores))
            gini_stats["debug/raw_lp_nonzero_count"] = float(sum(1 for x in raw_lp_scores if x > 1e-10))

        if raw_lp_scores and task_labels_list:
            label_lp_sums = {}
            for label, lp in zip(task_labels_list, raw_lp_scores, strict=True):
                label_lp_sums[label] = label_lp_sums.get(label, 0.0) + lp

            if label_lp_sums:
                label_lp_values = list(label_lp_sums.values())
                gini_stats["curriculum_gini/raw_lp_by_label"] = self._calculate_gini_coefficient(label_lp_values)

        if z_scored_lp_scores:
            gini_stats["curriculum_gini/zscored_lp_scores"] = self._calculate_gini_coefficient(
                [abs(z) for z in z_scored_lp_scores]
            )

        if sampling_probs:
            gini_stats["curriculum_gini/sampling_probs"] = self._calculate_gini_coefficient(sampling_probs)

        if sampling_probs and task_labels_list:
            label_prob_sums = {}
            for label, prob in zip(task_labels_list, sampling_probs, strict=True):
                label_prob_sums[label] = label_prob_sums.get(label, 0.0) + prob

            if label_prob_sums:
                label_prob_values = list(label_prob_sums.values())
                gini_stats["curriculum_gini/sampling_probs_by_label"] = self._calculate_gini_coefficient(
                    label_prob_values
                )

        if self._label_sampling_counts_this_epoch:
            label_sampling_values = list(self._label_sampling_counts_this_epoch.values())
            gini_stats["curriculum_gini/sampling_by_label"] = self._calculate_gini_coefficient(label_sampling_values)

        if self._label_eviction_counts:
            label_eviction_values = list(self._label_eviction_counts.values())
            if label_eviction_values and sum(label_eviction_values) > 0:
                gini_stats["curriculum_gini/evictions_by_label"] = self._calculate_gini_coefficient(
                    label_eviction_values
                )

        composition_data = self.get_pool_composition_stats()
        if composition_data["pool_composition"]:
            pool_comp_values = list(composition_data["pool_composition"].values())
            gini_stats["curriculum_gini/pool_composition_by_label"] = self._calculate_gini_coefficient(pool_comp_values)

        # Selectivity loss: how much inequality is reduced in the transformation pipeline
        if "curriculum_gini/raw_lp_scores" in gini_stats and "curriculum_gini/sampling_probs" in gini_stats:
            selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_scores"] - gini_stats["curriculum_gini/sampling_probs"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_to_prob"] = selectivity_loss

        if "curriculum_gini/raw_lp_by_label" in gini_stats and "curriculum_gini/sampling_probs_by_label" in gini_stats:
            label_prob_selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_by_label"] - gini_stats["curriculum_gini/sampling_probs_by_label"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_label_to_prob_label"] = label_prob_selectivity_loss

        if "curriculum_gini/raw_lp_by_label" in gini_stats and "curriculum_gini/sampling_by_label" in gini_stats:
            label_selectivity_loss = (
                gini_stats["curriculum_gini/raw_lp_by_label"] - gini_stats["curriculum_gini/sampling_by_label"]
            )
            gini_stats["curriculum_gini/selectivity_loss_lp_label_to_sampling_label"] = label_selectivity_loss

        return gini_stats

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats including learning progress and slice distribution analysis."""
        stats = {}

        # Learning progress stats from scorer with lp/ prefix
        lp_stats = self.scorer.get_stats()
        for key, value in lp_stats.items():
            stats[f"lp/{key}"] = value

        # Dual-pool statistics
        if self.hypers.use_dual_pool and isinstance(self.task_tracker, DualPoolTaskTracker):
            # Pool sizes
            num_explore = len(self.task_tracker.get_all_explore_tasks())
            num_exploit = len(self.task_tracker.get_all_exploit_tasks())
            stats["dual_pool/num_explore_tasks"] = float(num_explore)
            stats["dual_pool/num_exploit_tasks"] = float(num_exploit)

            # Explore-Exploit Ratio
            stats["dual_pool/explore_exploit_ratio"] = self._explore_exploit_ratio

            # Promotion statistics
            stats["dual_pool/num_promotions"] = float(self._num_promotions)
            stats["dual_pool/num_promotion_attempts"] = float(self._num_promotion_attempts)
            if self._num_promotion_attempts > 0:
                stats["dual_pool/promotion_success_rate"] = self._num_promotions / self._num_promotion_attempts
            else:
                stats["dual_pool/promotion_success_rate"] = 0.0

            # Recent promotion rate (from sliding window)
            if len(self._promotion_window) > 0:
                stats["dual_pool/recent_promotion_rate"] = sum(self._promotion_window) / len(self._promotion_window)
            else:
                stats["dual_pool/recent_promotion_rate"] = 0.0

            # Phase indicator
            stats["dual_pool/is_bootstrap_phase"] = 1.0 if self._current_phase == "bootstrap" else 0.0
            stats["dual_pool/is_steady_state_phase"] = 1.0 if self._current_phase == "steady_state" else 0.0

        return stats

    # ========== Dual-Pool Public Methods ==========

    def is_dual_pool_mode(self) -> bool:
        """Check if dual-pool mode is enabled."""
        return self.hypers.use_dual_pool

    def get_current_phase(self) -> str:
        """Get current dual-pool phase ('bootstrap' or 'steady_state').

        Returns:
            'bootstrap' if exploit pool is filling, 'steady_state' otherwise.
            Returns 'single_pool' if not in dual-pool mode.
        """
        if not self.hypers.use_dual_pool:
            return "single_pool"
        return self._current_phase

    def select_pool_for_sampling(self) -> str:
        """Select which pool to sample tasks from based on phase and EER.

        Bootstrap Phase: Always returns 'explore' (100% exploration)
        Steady-State Phase: Returns 'explore' with probability ρ, 'exploit' with probability 1-ρ

        Returns:
            'explore' or 'exploit' pool name
        """
        if not self.hypers.use_dual_pool:
            raise RuntimeError("select_pool_for_sampling called in single-pool mode")

        if self._current_phase == "bootstrap":
            # Bootstrap: 100% explore until exploit pool is full
            return "explore"
        else:
            # Steady-state: Sample based on EER
            if random.random() < self._explore_exploit_ratio:
                return "explore"
            else:
                return "exploit"

    def select_pool_for_creation(self) -> str:
        """Select which pool to create new tasks in.

        All new tasks are created in the explore pool. This ensures:
        1. New tasks get sufficient evaluation before promotion
        2. Exploit pool only contains validated tasks
        3. Exploration is always testing new hypotheses

        Returns:
            'explore' - all new tasks go to explore pool
        """
        if not self.hypers.use_dual_pool:
            raise RuntimeError("select_pool_for_creation called in single-pool mode")

        return "explore"

    def get_pool_task_ids(self, pool: str) -> List[int]:
        """Get all task IDs from a specific pool.

        Args:
            pool: 'explore' or 'exploit'

        Returns:
            List of task IDs in the specified pool
        """
        if not self.hypers.use_dual_pool:
            raise RuntimeError("get_pool_task_ids called in single-pool mode")

        assert isinstance(self.task_tracker, DualPoolTaskTracker)

        if pool == "explore":
            return self.task_tracker.get_all_explore_tasks()
        elif pool == "exploit":
            return self.task_tracker.get_all_exploit_tasks()
        else:
            raise ValueError(f"Invalid pool: {pool}. Must be 'explore' or 'exploit'")

    # ========== Dual-Pool Internal Methods ==========

    def _check_promotion_eligibility(self, task_id: int) -> bool:
        """Check if a task in explore pool is eligible for promotion.

        A task is eligible if:
        1. It's in the explore pool
        2. It has at least S_min samples (promotion_min_samples)

        Args:
            task_id: Task ID to check

        Returns:
            True if task is eligible for promotion
        """
        if not self.hypers.use_dual_pool:
            return False

        assert isinstance(self.task_tracker, DualPoolTaskTracker)

        # Check if task is in explore pool
        if self.task_tracker._task_pool_map.get(task_id) != "explore":
            return False

        # Check if task has enough samples
        stats = self.task_tracker.get_task_stats(task_id)
        if stats is None:
            return False

        return stats["completion_count"] >= self.hypers.promotion_min_samples

    def _attempt_promotion(self, task_id: int) -> bool:
        """Attempt to promote a task from explore to exploit pool.

        Promotion succeeds if:
        1. Task is eligible (has S_min samples)
        2. Task's score is higher than the lowest-scoring task in exploit pool
        3. Exploit pool has space OR lowest-scoring task can be evicted

        Args:
            task_id: Task ID to promote

        Returns:
            True if promotion succeeded, False otherwise
        """
        if not self.hypers.use_dual_pool or not self._check_promotion_eligibility(task_id):
            return False

        assert isinstance(self.task_tracker, DualPoolTaskTracker)

        # Get task score
        task_score = self.scorer.score_task(task_id, self.task_tracker.explore_tracker)

        # Only promote tasks with positive learning progress
        # Tasks with negative/zero LP are not making progress and shouldn't be exploited
        if task_score <= 0.0:
            return False

        # Get all exploit tasks and their scores
        exploit_task_ids = self.task_tracker.get_all_exploit_tasks()

        # If exploit pool is not full, promote task with positive LP
        if len(exploit_task_ids) < self.hypers.num_exploit_tasks:
            success = self.task_tracker.promote_task(task_id)
            if success:
                self._num_promotions += 1
            return success

        # Exploit pool is full - check if task score is higher than minimum
        exploit_scores = {
            tid: self.scorer.score_task(tid, self.task_tracker.exploit_tracker) for tid in exploit_task_ids
        }

        if not exploit_scores:
            return False

        min_exploit_score = min(exploit_scores.values())
        min_exploit_task = min(exploit_scores.items(), key=lambda x: x[1])[0]

        # Only promote if score is higher than minimum exploit task
        if task_score > min_exploit_score:
            # Evict lowest-scoring exploit task
            self.task_tracker.remove_task(min_exploit_task)

            # Promote task
            success = self.task_tracker.promote_task(task_id)
            if success:
                self._num_promotions += 1
            return success

        return False

    def _update_explore_exploit_ratio(self) -> None:
        """Update Explore-Exploit Ratio (ρ) based on recent promotion rate.

        The EER is updated using an exponential moving average:
            ρ(k+1) = α_EER * ρ(k) + (1 - α_EER) * r_promote(k)

        where r_promote is the promotion rate from the sliding window.

        The updated ρ is clipped to [ρ_min, ρ_max] bounds.
        """
        if not self.hypers.use_dual_pool or len(self._promotion_window) == 0:
            return

        # Calculate promotion rate from sliding window
        num_promotions_in_window = sum(self._promotion_window)
        window_size = len(self._promotion_window)

        r_promote = num_promotions_in_window / window_size if window_size > 0 else 0.0

        # EMA update
        alpha_eer = self.hypers.explore_exploit_ratio_alpha
        self._explore_exploit_ratio = alpha_eer * self._explore_exploit_ratio + (1 - alpha_eer) * r_promote

        # Clip to bounds
        self._explore_exploit_ratio = max(
            self.hypers.explore_exploit_ratio_min,
            min(self._explore_exploit_ratio, self.hypers.explore_exploit_ratio_max),
        )

    def _update_phase(self) -> None:
        """Update phase from bootstrap to steady_state when exploit pool is full."""
        if not self.hypers.use_dual_pool:
            return

        assert isinstance(self.task_tracker, DualPoolTaskTracker)

        if self._current_phase == "bootstrap":
            exploit_tasks = self.task_tracker.get_all_exploit_tasks()
            if len(exploit_tasks) >= self.hypers.num_exploit_tasks:
                self._current_phase = "steady_state"

    def get_state(self) -> Dict[str, Any]:
        """Get learning progress algorithm state for checkpointing."""
        state = {
            "type": self.hypers.algorithm_type(),
            "hypers": self.hypers.model_dump(),
            "task_tracker": self.task_tracker.get_state(),
            "scorer": self.scorer.get_state(),
            "label_tracking": {
                # Labels are now stored in TaskTracker shared memory
                # Only save sampling/eviction counts and active label metadata
                "label_sampling_counts": self._label_sampling_counts,
                "label_eviction_counts": self._label_eviction_counts,
                "active_labels": list(self._active_labels),
                "inactive_labels_fifo": self._inactive_labels_fifo,
            },
        }

        # Add dual-pool state if enabled
        if self.hypers.use_dual_pool:
            state["dual_pool"] = {
                "explore_exploit_ratio": self._explore_exploit_ratio,
                "promotion_window": list(self._promotion_window),
                "num_promotions": self._num_promotions,
                "num_promotion_attempts": self._num_promotion_attempts,
                "current_phase": self._current_phase,
            }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load learning progress algorithm state from checkpoint."""
        import logging

        logger = logging.getLogger(__name__)

        # Restore task tracker
        self.task_tracker.load_state(state["task_tracker"])

        # Log what was restored
        num_tasks = len(self.task_tracker.get_all_tracked_tasks())
        total_completions = self.task_tracker._total_completions
        logger.info(
            f"LP Algorithm: Loaded {num_tasks} tasks from checkpoint with {total_completions} total completions"
        )

        # Restore scorer state
        if "scorer" in state:
            self.scorer.load_state(state["scorer"])

        # Restore label tracking state (if available, for backward compatibility)
        # Labels themselves are now in TaskTracker shared memory
        if "label_tracking" in state:
            label_data = state["label_tracking"]
            self._label_sampling_counts = label_data.get("label_sampling_counts", {})
            self._label_eviction_counts = label_data.get("label_eviction_counts", {})
            self._active_labels = set(label_data.get("active_labels", []))
            self._inactive_labels_fifo = label_data.get("inactive_labels_fifo", [])

        # Restore dual-pool state if available
        if self.hypers.use_dual_pool and "dual_pool" in state:
            dual_pool_data = state["dual_pool"]
            self._explore_exploit_ratio = dual_pool_data.get(
                "explore_exploit_ratio", self.hypers.explore_exploit_ratio_init
            )
            promotion_window_list = dual_pool_data.get("promotion_window", [])
            self._promotion_window = deque(promotion_window_list, maxlen=self.hypers.promotion_rate_window)
            self._num_promotions = dual_pool_data.get("num_promotions", 0)
            self._num_promotion_attempts = dual_pool_data.get("num_promotion_attempts", 0)
            self._current_phase = dual_pool_data.get("current_phase", "bootstrap")
            logger.info(
                f"LP Algorithm: Restored dual-pool state - "
                f"EER={self._explore_exploit_ratio:.3f}, "
                f"phase={self._current_phase}, "
                f"promotions={self._num_promotions}"
            )

        # Fix LP scores for tasks loaded from old checkpoints
        # Tasks with 0 completions should have exploration_bonus, not 0.0
        fixed_count = 0
        for task_id in self.task_tracker.get_all_tracked_tasks():
            stats = self.task_tracker.get_task_stats(task_id)
            if stats and stats["completion_count"] == 0 and stats["lp_score"] == 0.0:
                self.task_tracker.update_lp_score(task_id, self.hypers.exploration_bonus)
                fixed_count += 1

        if fixed_count > 0:
            bonus = self.hypers.exploration_bonus
            logger.info(f"LP Algorithm: Fixed {fixed_count} tasks with 0 completions to exploration_bonus={bonus}")

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources with better error handling."""
        task_tracker = getattr(self, "task_tracker", None)
        if task_tracker is None:
            return

        try:
            task_tracker.cleanup_shared_memory()
        except Exception as e:
            # Log but don't raise - cleanup should be best-effort
            import logging

            logging.warning(f"Failed to cleanup shared memory: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        hypers = getattr(self, "hypers", None)
        if hypers is not None and getattr(hypers, "use_shared_memory", False):
            self.cleanup_shared_memory()

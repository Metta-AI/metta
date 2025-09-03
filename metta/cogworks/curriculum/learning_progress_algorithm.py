"""
Learning Progress Curriculum Algorithm for Curriculum.

This module implements the learning progress algorithm as a CurriculumAlgorithm
that can be used with Curriculum nodes to adaptively sample tasks based on
bidirectional learning progress tracking with local task memory pool.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gym.spaces import Discrete
from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumTask,
)

logger = logging.getLogger(__name__)

DEFAULT_SUCCESS_RATE = 0.5


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Hyperparameters for LearningProgressAlgorithm."""

    type: str = "learning_progress"
    ema_timescale: float = Field(default=0.001, description="EMA timescale for learning progress")
    pool_size: int = Field(default=16, description="Size of the task pool")
    sample_size: int = Field(default=8, description="Number of tasks to sample")
    max_samples: int = Field(default=10, description="Maximum samples before eviction")
    exploration_bonus: float = Field(default=0.1, description="Exploration bonus for sampling")

    # Performance optimization settings
    stats_update_frequency: int = Field(default=100, description="Update stats every N task completions")
    debug_log_frequency: int = Field(default=1000, description="Log debug info every N operations")
    max_bucket_axes_for_logging: int = Field(
        default=3, description="Maximum number of bucket axes to track for logging (reduces overhead)"
    )

    # Logging verbosity control
    enable_detailed_bucket_logging: bool = Field(default=False, description="Enable detailed bucket statistics")
    enable_learning_progress_logging: bool = Field(default=True, description="Enable learning progress statistics")

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)

    model_config: ConfigDict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """
    Learning progress algorithm that manages a unified pool of tasks.

    Performance optimizations:
    - Stats are cached and only updated every 100 task completions
    - Debug logging is limited to every 1000 operations (minimal overhead)
    - Only tracks the first 3 bucket axes (default) to reduce overhead
    - Verbose debug logging removed for maximum performance
    - Basic bucket density information is always logged for real-time insight
    - Expensive bucket statistics are cached to reduce computation overhead
    - Lightweight bucket stats focus on summary metrics rather than per-bin details
    """

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        # Don't initialize weights since this algorithm uses its own sampling strategy
        super().__init__(num_tasks, hypers, initialize_weights=False)

        self.hypers = hypers
        self._curriculum = None  # Reference to base curriculum for unified task tracking

        # Initialize bidirectional learning progress tracker
        # Search space should be exactly the pool size - no expansion needed
        search_space_size = hypers.pool_size
        self._lp_tracker = BidirectionalLearningProgress(
            search_space=search_space_size,
            ema_timescale=hypers.ema_timescale,
            num_active_tasks=hypers.pool_size,
        )

        # Task management - now unified with base curriculum
        self._task_memory: Dict[int, Tuple[int, str, int, float, float, float]] = {}
        self._task_id_to_index: Dict[int, int] = {}
        self._next_index = 0

        # Index recycling - maintain a pool of available indices
        self._available_indices = set(range(search_space_size))
        self._used_indices = set()

        # Bucket tracking for completion density analysis
        self._bucket_tracking: Dict[str, Dict[int, Any]] = {}  # bucket_name -> task_id -> value
        self._bucket_completion_counts: Dict[str, Dict[int, int]] = {}  # bucket_name -> bin_index -> count
        self._bucket_bins: Dict[str, List[float]] = {}  # bucket_name -> bin_edges
        self._bucket_is_discrete: Dict[str, bool] = {}  # bucket_name -> is_discrete
        self._bucket_completion_history: Dict[str, List[float]] = {}  # bucket_name -> completion_density_over_time

        # Performance optimization: batch logging
        self._stats_update_counter = 0
        self._stats_update_frequency = hypers.stats_update_frequency
        self._debug_log_frequency = hypers.debug_log_frequency
        self._max_bucket_axes = hypers.max_bucket_axes_for_logging
        self._cached_stats = {}
        self._last_stats_update = 0

        # Track which buckets we're actually monitoring
        self._monitored_buckets: set = set()

    def set_curriculum_reference(self, curriculum):
        """Set reference to base curriculum for stats updates."""
        self._curriculum = curriculum

    # Core task management methods
    def get_task_from_pool(self, task_generator, rng) -> CurriculumTask:
        """Get a task from the unified pool, creating or evicting as needed."""
        # Create new task if pool not full
        if len(self._task_memory) < self.hypers.pool_size:
            return self._create_task(task_generator, rng)

        # Pool is full - we need to evict a task and create a new one
        # Force eviction to make room
        self._evict_task()

        # Create new task
        return self._create_task(task_generator, rng)

    def _create_task(self, task_generator, rng) -> CurriculumTask:
        """Create a new task and add it to the pool."""
        if len(self._task_memory) >= self.hypers.pool_size:
            self._evict_task()

        # Generate a unique task ID
        task_id = self._generate_task_id(rng)
        env_cfg = task_generator.get_task(task_id)
        task = CurriculumTask(task_id, env_cfg)

        # Get an available index from the pool (recycle indices)
        if not self._available_indices:
            # This should never happen if we're properly evicting tasks
            raise RuntimeError("No available indices in search space - pool management error")

        task_index = self._available_indices.pop()
        self._used_indices.add(task_index)
        self._task_id_to_index[task_id] = task_index

        # Add to memory for algorithm tracking
        self._task_memory[task_id] = (task_id, "default", 0, 0.0, 0.0, 0.0)

        # Add to base curriculum's task storage for unified tracking
        if self._curriculum is not None:
            self._curriculum._tasks[task_id] = task
            self._curriculum._task_ids.add(task_id)
            self._curriculum._num_created += 1

        # Extract and track bucket values for this task
        bucket_values = self._extract_bucket_values(task)
        self._initialize_bucket_tracking(bucket_values)
        self._update_bucket_tracking(task_id, bucket_values)

        # Store the bucket values in the task for later use
        if hasattr(task, "_bucket_values"):
            task._bucket_values = bucket_values
            logger.debug(f"Stored bucket values for task {task_id}: {bucket_values}")

        return task

    def _choose_task(self) -> int:
        """Choose a task from the pool based on learning progress scores."""
        if not self._task_memory:
            raise ValueError("No tasks in pool to sample from")

        # Get learning progress scores
        self._lp_tracker._update()
        lp_scores = self._lp_tracker._learning_progress()

        # Create sampling probabilities
        task_ids = list(self._task_memory.keys())
        task_probs = []

        for task_id in task_ids:
            task_index = self._task_id_to_index.get(task_id, 0)
            lp_score = lp_scores[task_index] if task_index < len(lp_scores) else 0.0
            prob = lp_score + self.hypers.exploration_bonus
            task_probs.append(prob)

        # Normalize probabilities
        total_prob = sum(task_probs)
        if total_prob > 0:
            task_probs = [p / total_prob for p in task_probs]
        else:
            task_probs = [1.0 / len(task_ids)] * len(task_ids)

        return np.random.choice(task_ids, p=task_probs)

    def _evict_task(self):
        """Evict a task from the pool based on learning progress and sample count."""
        if not self._task_memory:
            return

        # If pool is full, we need to evict a task to make room
        if len(self._task_memory) >= self.hypers.pool_size:
            # Find task with lowest LP score that has enough samples
            worst_task_id = None
            worst_score = float("inf")

            for task_id, (_, _, sample_count, _, _, lp_score) in self._task_memory.items():
                # Evict tasks that have enough samples OR if we need to make room
                if sample_count >= self.hypers.max_samples or len(self._task_memory) >= self.hypers.pool_size:
                    if lp_score < worst_score:
                        worst_score = lp_score
                        worst_task_id = task_id

            # If no task meets the criteria, evict the one with lowest LP score
            if worst_task_id is None:
                worst_task_id = min(
                    self._task_memory.keys(), key=lambda tid: self._task_memory[tid][5]
                )  # Index 5 is lp_score

            if worst_task_id is not None:
                # Get the index to recycle
                task_index = self._task_id_to_index.get(worst_task_id)

                # Remove from task memory
                if worst_task_id in self._task_memory:
                    del self._task_memory[worst_task_id]
                    del self._task_id_to_index[worst_task_id]

                    # Recycle the index back to available pool
                    if task_index is not None:
                        self._used_indices.discard(task_index)
                        self._available_indices.add(task_index)

                        # Clear the learning progress data for this index
                        if task_index < len(self._lp_tracker._outcomes):
                            self._lp_tracker._outcomes[task_index] = []
                        if task_index < len(self._lp_tracker._task_success_rate):
                            self._lp_tracker._task_success_rate[task_index] = 0.0

                    # Remove bucket tracking for this task
                    for bucket_name in self._bucket_tracking:
                        if worst_task_id in self._bucket_tracking[bucket_name]:
                            del self._bucket_tracking[bucket_name][worst_task_id]

                    # Remove from base curriculum's task storage for unified tracking
                    if self._curriculum is not None:
                        if worst_task_id in self._curriculum._tasks:
                            del self._curriculum._tasks[worst_task_id]
                        if worst_task_id in self._curriculum._task_ids:
                            self._curriculum._task_ids.remove(worst_task_id)
                        self._curriculum._num_evicted += 1

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance and learning progress."""
        if task_id not in self._task_memory:
            return

        # Update task memory
        seed, family, sample_count, current_score, recent_score, lp_score = self._task_memory[task_id]
        self._task_memory[task_id] = (seed, family, sample_count + 1, recent_score, score, lp_score)

        # Update base curriculum's task completion tracking
        if self._curriculum is not None and task_id in self._curriculum._tasks:
            task = self._curriculum._tasks[task_id]
            task._num_completions += 1
            task._num_scheduled += 1

        # Update learning progress tracker
        task_index = self._task_id_to_index.get(task_id, 0)

        # Safety check: ensure task_index is within bounds of _outcomes
        if task_index >= len(self._lp_tracker._outcomes):
            # This should never happen with proper index recycling
            logger.error(f"Task index {task_index} out of bounds for outcomes size {len(self._lp_tracker._outcomes)}")
            logger.error(f"Available indices: {self._available_indices}, Used indices: {self._used_indices}")
            return

        self._lp_tracker._outcomes[task_index].append(score)
        self._lp_tracker._update()

        # Update LP score in memory
        raw_lp_scores = self._lp_tracker._learning_progress()
        if task_index < len(raw_lp_scores):
            new_lp_score = float(raw_lp_scores[task_index])
            self._task_memory[task_id] = (seed, family, sample_count + 1, recent_score, score, new_lp_score)

        # Update bucket completion density tracking
        if self._curriculum and task_id in self._curriculum._tasks:
            self._update_bucket_completion_density(task_id, score)

        # Increment stats counter for batching
        self._stats_update_counter += 1

    # Helper methods for task management
    def _generate_task_id(self, rng) -> int:
        """Generate a unique task ID."""
        while True:
            task_id = rng.randint(0, 1000000)
            if task_id not in self._task_memory:
                return task_id

    def _get_task_lp_score(self, task_id: int) -> float:
        """Get the learning progress score for a specific task."""
        if task_id not in self._task_memory:
            return 0.0

        # Force update of learning progress calculation
        self._lp_tracker._update()

        # Get the raw learning progress score, not the distribution value
        task_index = self._task_id_to_index.get(task_id, 0)
        raw_lp_scores = self._lp_tracker._learning_progress()

        # Safety check: ensure task_index is within bounds
        if task_index >= len(raw_lp_scores):
            return 0.0

        return float(raw_lp_scores[task_index])

    def _validate_index_management(self) -> dict[str, Any]:
        """Validate that index management is working correctly."""
        validation = {
            "pool_size": self.hypers.pool_size,
            "active_tasks": len(self._task_memory),
            "available_indices": len(self._available_indices),
            "used_indices": len(self._used_indices),
            "total_indices": len(self._available_indices) + len(self._used_indices),
            "search_space_size": len(self._lp_tracker._outcomes),
        }

        # Validate invariants
        validation["indices_match_pool"] = validation["total_indices"] == validation["search_space_size"]
        validation["pool_not_overfull"] = validation["active_tasks"] <= validation["pool_size"]
        validation["indices_available"] = validation["available_indices"] >= 0

        return validation

    def _validate_completion_densities(self) -> dict[str, Any]:
        """Validate that completion densities are properly normalized."""
        validation = {}

        for bucket_name in self._bucket_tracking:
            if bucket_name not in self._bucket_completion_counts:
                continue

            completion_counts = self._bucket_completion_counts[bucket_name]
            if not completion_counts:
                continue

            total_completions = sum(completion_counts.values())
            if total_completions == 0:
                continue

            # Calculate densities for validation
            densities = []
            if self._bucket_is_discrete[bucket_name]:
                for bin_index, count in completion_counts.items():
                    if bin_index < len(self._bucket_bins[bucket_name]):
                        density = count / total_completions
                        densities.append(density)
            else:
                for bin_index, count in completion_counts.items():
                    if bin_index < len(self._bucket_bins[bucket_name]) - 1:
                        density = count / total_completions
                        densities.append(density)

            if densities:
                density_sum = sum(densities)
                validation[f"{bucket_name}/density_sum"] = density_sum
                validation[f"{bucket_name}/density_sum_error"] = abs(density_sum - 1.0)
                validation[f"{bucket_name}/num_density_bins"] = len(densities)
                validation[f"{bucket_name}/total_completions"] = total_completions

        return validation

    # Statistics and logging methods
    def stats(self) -> dict[str, float]:
        """Return unified statistics for the pool including bucket completion density."""
        stats = {}

        # Add learning progress stats if enabled
        if self.hypers.enable_learning_progress_logging:
            base_stats = self._lp_tracker.add_stats()
            stats.update(base_stats)

        # Add index management validation stats
        validation_stats = self._validate_index_management()
        for key, value in validation_stats.items():
            if isinstance(value, (int, float)):
                stats[f"index_management/{key}"] = float(value)

        # Add completion density validation stats
        density_validation = self._validate_completion_densities()
        for key, value in density_validation.items():
            if isinstance(value, (int, float)):
                stats[f"density_validation/{key}"] = float(value)

        # Performance optimization: only recalculate expensive bucket stats periodically
        if self._stats_update_counter < self._stats_update_frequency:
            # Return cached bucket stats but always include basic bucket info
            basic_bucket_stats = self._get_basic_bucket_stats()
            stats.update(basic_bucket_stats)
            stats.update(self._cached_stats)
        else:
            # Update expensive bucket stats and cache them
            self._stats_update_counter = 0

            # Add bucket completion density statistics (always detailed, restored from 4b37673)
            bucket_stats = self._get_lightweight_bucket_stats()

            # Cache the expensive results
            self._cached_stats = bucket_stats
            stats.update(bucket_stats)

        return stats

    # Bucket tracking methods
    def _extract_bucket_values(self, task: CurriculumTask) -> Dict[str, Any]:
        """Extract bucket parameter values from a task's environment configuration."""
        bucket_values = {}

        try:
            # First, try to get stored bucket values from the task
            stored_bucket_values = task.get_bucket_values()
            if stored_bucket_values:
                return stored_bucket_values

            # Fallback: try to extract from env_cfg (for backward compatibility)

            # Access the environment configuration to extract bucket values
            env_cfg = task.get_env_cfg()

            # Get the flattened config as a dict to search for bucket values
            config_dict = env_cfg.model_dump()

            # Common bucket paths that might exist in different environments
            potential_bucket_paths = [
                # Arena-style paths
                "game.level_map.num_agents",
                "game.level_map.width",
                "game.level_map.height",
                "game.actions.attack.consumed_resources.laser",
                # Navigation-style paths
                "game.agent.rewards.inventory.heart",
                "game.agent.rewards.inventory.heart_max",
                "game.map_builder.instance_map.dir",
                "game.map_builder.instance_map.objects.altar",
                "game.map_builder.width",
                "game.map_builder.height",
                "game.map_builder.objects.altar",
            ]

            # Extract values for potential bucket paths
            for path in potential_bucket_paths:
                value = self._get_nested_value(config_dict, path)
                if value is not None:
                    bucket_values[path] = value

            # Also look for any other numeric/string values in the config that might be buckets
            # This catches any buckets we didn't explicitly listed above
            self._extract_potential_buckets_recursive(config_dict, "", bucket_values)

        except Exception:
            pass  # Silently handle extraction errors

        return bucket_values

    def _get_nested_value(self, config_dict: dict, path: str) -> Any:
        """Get a nested value from a config dict using dot notation."""
        try:
            keys = path.split(".")
            value = config_dict
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except (KeyError, AttributeError, TypeError):
            return None

    def _extract_potential_buckets_recursive(self, config_dict: dict, current_path: str, bucket_values: dict):
        """Recursively extract potential bucket values from config dict."""
        if not isinstance(config_dict, dict):
            return

        for key, value in config_dict.items():
            path = f"{current_path}.{key}" if current_path else key

            # Skip very deep nesting to avoid noise
            if path.count(".") > 5:
                continue

            # Look for values that could be bucket parameters
            if isinstance(value, (int, float, str)) and value is not None:
                # Skip very long strings (likely not bucket values)
                if isinstance(value, str) and len(value) > 100:
                    continue

                # Skip paths that are likely not bucket parameters
                if any(skip in path.lower() for skip in ["id", "uuid", "hash", "timestamp", "url", "path"]):
                    continue

                # Add as potential bucket value
                bucket_values[path] = value

            elif isinstance(value, dict):
                # Recursively search nested dicts
                self._extract_potential_buckets_recursive(value, path, bucket_values)

    def _initialize_bucket_tracking(self, bucket_values: Dict[str, Any]):
        """Initialize bucket tracking for newly discovered bucket parameters."""
        # Only track the first N bucket axes to reduce overhead
        bucket_names = list(bucket_values.keys())[: self._max_bucket_axes]

        for bucket_name in bucket_names:
            if bucket_name not in self._bucket_tracking:
                self._bucket_tracking[bucket_name] = {}
                self._bucket_completion_counts[bucket_name] = {}
                self._bucket_completion_history[bucket_name] = []
                self._monitored_buckets.add(bucket_name)

                # Determine if bucket is discrete or continuous
                value = bucket_values[bucket_name]
                if isinstance(value, (int, str)) or (isinstance(value, float) and value.is_integer()):
                    self._bucket_is_discrete[bucket_name] = True
                    # For discrete values, create bins for each unique value
                    self._bucket_bins[bucket_name] = []
                else:
                    self._bucket_is_discrete[bucket_name] = False
                    # For continuous values, create 10 histogram bins
                    self._bucket_bins[bucket_name] = []
            else:
                pass  # Bucket already exists

        # Track which buckets we're monitoring (no logging)
        pass

    def _update_bucket_tracking(self, task_id: int, bucket_values: Dict[str, Any]):
        """Update bucket tracking with new task values."""
        for bucket_name, value in bucket_values.items():
            # Only process buckets we're monitoring
            if bucket_name not in self._monitored_buckets:
                continue

            if bucket_name in self._bucket_tracking:
                self._bucket_tracking[bucket_name][task_id] = value

                # Update bins for continuous parameters
                if not self._bucket_is_discrete[bucket_name]:
                    if not self._bucket_bins[bucket_name]:
                        # Initialize bins for continuous parameter
                        # We'll need to collect some data first to determine range
                        pass

                # Collect values for continuous bin initialization
                if not self._bucket_is_discrete[bucket_name]:
                    # Store values to initialize bins later when we have enough data
                    if not hasattr(self, "_continuous_values_buffer"):
                        self._continuous_values_buffer = {}
                    if bucket_name not in self._continuous_values_buffer:
                        self._continuous_values_buffer[bucket_name] = []
                    self._continuous_values_buffer[bucket_name].append(float(value))

                    # Initialize bins when we have enough data points (reduced from 10 to 3)
                    if len(self._continuous_values_buffer[bucket_name]) >= 3:
                        self._initialize_continuous_bins(bucket_name, self._continuous_values_buffer[bucket_name])
                        # Clear buffer after initialization
                        self._continuous_values_buffer[bucket_name] = []

    def _update_bucket_completion_density(self, task_id: int, score: float):
        """Update bucket completion density tracking when a task completes."""
        if task_id not in self._task_memory:
            return

        # Get bucket values for this task from base curriculum
        if self._curriculum and task_id in self._curriculum._tasks:
            task = self._curriculum._tasks[task_id]
            bucket_values = self._extract_bucket_values(task)

            # Update completion density for each bucket
            for bucket_name, value in bucket_values.items():
                # Only process buckets we're monitoring
                if bucket_name not in self._monitored_buckets:
                    continue

                if bucket_name in self._bucket_tracking:
                    # Find the appropriate bin for this value
                    bin_index = self._get_bin_index(bucket_name, value)

                    if bin_index is not None:
                        # Increment completion count for this bin
                        if bin_index not in self._bucket_completion_counts[bucket_name]:
                            self._bucket_completion_counts[bucket_name][bin_index] = 0
                        self._bucket_completion_counts[bucket_name][bin_index] += 1

                        # Update completion density history
                        self._update_completion_density_history(bucket_name)

    def _get_bin_index(self, bucket_name: str, value: Any) -> Optional[int]:
        """Get the bin index for a given bucket value."""
        if bucket_name not in self._bucket_tracking:
            logger.debug(f"Bucket {bucket_name} not in tracking")
            return None

        if self._bucket_is_discrete[bucket_name]:
            # For discrete values, find or create bin for this value
            if value not in self._bucket_bins[bucket_name]:
                self._bucket_bins[bucket_name].append(value)
                logger.debug(f"Added new discrete bin for {bucket_name}: {value}")
            bin_index = self._bucket_bins[bucket_name].index(value)
            logger.debug(
                f"Discrete bucket {bucket_name}, value {value} -> bin {bin_index} "
                f"(bins: {self._bucket_bins[bucket_name]})"
            )
            return bin_index
        else:
            # For continuous values, we need to have bins initialized
            if not self._bucket_bins[bucket_name] or len(self._bucket_bins[bucket_name]) < 2:
                logger.debug(f"Continuous bucket {bucket_name} bins not initialized: {self._bucket_bins[bucket_name]}")
                return None

            # Find the appropriate bin
            try:
                for i in range(len(self._bucket_bins[bucket_name]) - 1):
                    min_val = self._bucket_bins[bucket_name][i]
                    max_val = self._bucket_bins[bucket_name][i + 1]
                    if min_val <= value < max_val:
                        logger.debug(
                            f"Continuous bucket {bucket_name}, value {value} -> bin {i} [{min_val}, {max_val})"
                        )
                        return i
                # Handle edge case where value equals the last bin edge
                if value == self._bucket_bins[bucket_name][-1]:
                    bin_index = len(self._bucket_bins[bucket_name]) - 2
                    logger.debug(f"Continuous bucket {bucket_name}, value {value} -> edge bin {bin_index}")
                    return bin_index
                logger.debug(f"Continuous bucket {bucket_name}, value {value} not in any bin range")
            except (IndexError, TypeError) as e:
                logger.debug(f"Error finding bin for continuous bucket {bucket_name}, value {value}: {e}")
                return None
            return None

    def _update_completion_density_history(self, bucket_name: str):
        """Update the completion density history for a bucket."""
        if bucket_name not in self._bucket_completion_counts:
            return

        # Calculate current completion density across all bins
        completion_counts = self._bucket_completion_counts[bucket_name]
        if not completion_counts:
            return

        total_completions = sum(completion_counts.values())
        if total_completions == 0:
            return

        # Calculate density for each bin that has completions
        densities = []
        if self._bucket_is_discrete[bucket_name]:
            # For discrete, only normalize bins that have completions
            for bin_index, count in completion_counts.items():
                if bin_index < len(self._bucket_bins[bucket_name]):
                    density = count / total_completions
                    densities.append(density)
        else:
            # For continuous, we need bins to be initialized
            if len(self._bucket_bins[bucket_name]) < 2:
                return

            # Only include bins that have completions
            for bin_index, count in completion_counts.items():
                if bin_index < len(self._bucket_bins[bucket_name]) - 1:
                    density = count / total_completions
                    densities.append(density)

        # Store the current density distribution
        if densities:
            # Now densities should sum to 1.0 (or very close due to floating point)
            self._bucket_completion_history[bucket_name].append(np.mean(densities))

    def _initialize_continuous_bins(self, bucket_name: str, values: List[float]):
        """Initialize histogram bins for continuous bucket parameters."""
        if not values:
            logger.debug(f"No values provided for continuous bin initialization of {bucket_name}")
            return

        min_val = min(values)
        max_val = max(values)

        logger.debug(f"Initializing continuous bins for {bucket_name}: min={min_val}, max={max_val}, values={values}")

        # Create 10 bins for continuous parameters
        bin_edges = np.linspace(min_val, max_val, 11)  # 11 edges = 10 bins
        self._bucket_bins[bucket_name] = bin_edges.tolist()

        logger.debug(f"Created {len(bin_edges) - 1} bins for {bucket_name}: {self._bucket_bins[bucket_name]}")

    # Statistics generation methods
    def _get_basic_bucket_stats(self) -> dict[str, float]:
        """Get consolidated bucket statistics to reduce logging overhead."""
        stats = {}

        for bucket_name in self._bucket_tracking:
            if bucket_name in self._bucket_completion_counts:
                completion_counts = self._bucket_completion_counts[bucket_name]
                if completion_counts:
                    total_completions = sum(completion_counts.values())
                    if total_completions > 0:
                        # Log essential bucket completion info
                        stats[f"bucket/{bucket_name}/total_completions"] = float(total_completions)
                        stats[f"bucket/{bucket_name}/num_bins"] = float(len(completion_counts))

                        # For discrete buckets, always show completion density (restored from 4b37673)
                        if self._bucket_is_discrete[bucket_name]:
                            for bin_index, count in completion_counts.items():
                                if bin_index < len(self._bucket_bins[bucket_name]):
                                    value = self._bucket_bins[bucket_name][bin_index]
                                    stats[f"bucket/{bucket_name}/value_{value}_completions"] = float(count)
                                    density = float(count / total_completions)
                                    stats[f"bucket/{bucket_name}/value_{value}_density"] = density
                        else:
                            # For continuous buckets, only log essential summary stats
                            if len(self._bucket_bins[bucket_name]) >= 2:
                                bin_counts = list(completion_counts.values())
                                if bin_counts:
                                    stats[f"bucket/{bucket_name}/completion_variance"] = float(np.var(bin_counts))

        return stats

    def _get_lightweight_bucket_stats(self) -> dict[str, float]:
        """Get lightweight bucket completion density statistics for performance."""
        stats = {}

        for bucket_name in self._bucket_tracking:
            if bucket_name not in self._bucket_completion_counts:
                continue

            # Get completion counts for this bucket
            completion_counts = self._bucket_completion_counts[bucket_name]
            if not completion_counts:
                continue

            # Calculate basic statistics
            total_completions = sum(completion_counts.values())
            if total_completions == 0:
                continue

            # Add bucket completion statistics (lightweight version)
            stats[f"bucket/{bucket_name}/total_completions"] = float(total_completions)
            stats[f"bucket/{bucket_name}/num_bins"] = float(len(completion_counts))

            # For discrete buckets, always log top 3 most completed values (restored from 4b37673)
            if self._bucket_is_discrete[bucket_name]:
                # Sort by completion count and take top 3
                sorted_bins = sorted(completion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for bin_index, count in sorted_bins:
                    if bin_index < len(self._bucket_bins[bucket_name]):
                        value = self._bucket_bins[bucket_name][bin_index]
                        stats[f"bucket/{bucket_name}/value_{value}_completions"] = float(count)
                        density = float(count / total_completions)
                        stats[f"bucket/{bucket_name}/value_{value}_density"] = density
            else:
                # For continuous buckets, only log summary stats to reduce overhead
                if len(self._bucket_bins[bucket_name]) >= 2:
                    bin_counts = list(completion_counts.values())
                    if bin_counts:
                        stats[f"bucket/{bucket_name}/max_bin_completions"] = float(max(bin_counts))
                        stats[f"bucket/{bucket_name}/min_bin_completions"] = float(min(bin_counts))
                        mean_completions = float(sum(bin_counts) / len(bin_counts))
                        stats[f"bucket/{bucket_name}/mean_bin_completions"] = mean_completions

            # Add completion density evolution statistics (lightweight)
            if self._bucket_completion_history[bucket_name]:
                history = self._bucket_completion_history[bucket_name]
                stats[f"bucket/{bucket_name}/completion_density_mean"] = float(np.mean(history))
                # Only add trend if we have enough history
                if len(history) > 1:
                    stats[f"bucket/{bucket_name}/completion_density_trend"] = float(history[-1] - history[0])

        return stats

    # Configuration and utility methods
    def get_real_time_bucket_density(self) -> dict[str, float]:
        """Get real-time bucket density information without any caching."""
        return self._get_basic_bucket_stats()

    def set_logging_frequency(self, debug_frequency: int, stats_frequency: int):
        """Dynamically adjust logging frequency for different environments."""
        self._debug_log_frequency = debug_frequency
        self._stats_update_frequency = stats_frequency
        logger.info(f"Updated logging frequencies: debug={debug_frequency}, stats={stats_frequency}")

    def set_logging_verbosity(self, enable_detailed_bucket: bool = None, enable_learning_progress: bool = None):
        """Dynamically adjust logging verbosity for different environments.

        Args:
            enable_detailed_bucket: Whether to enable detailed bucket statistics
            enable_learning_progress: Whether to enable learning progress statistics
        """
        if enable_detailed_bucket is not None:
            self.hypers.enable_detailed_bucket_logging = enable_detailed_bucket
        if enable_learning_progress is not None:
            self.hypers.enable_learning_progress_logging = enable_learning_progress

        logger.info(
            f"Updated logging verbosity: detailed_bucket={self.hypers.enable_detailed_bucket_logging}, "
            f"learning_progress={self.hypers.enable_learning_progress_logging}"
        )

    def set_monitored_bucket_axes(self, max_axes: int):
        """Dynamically change how many bucket axes to monitor for logging."""
        old_max = self._max_bucket_axes
        self._max_bucket_axes = max_axes

        # If reducing the number of monitored axes, remove excess buckets
        if max_axes < old_max:
            current_buckets = list(self._monitored_buckets)
            for bucket_name in current_buckets[max_axes:]:
                self._monitored_buckets.discard(bucket_name)
                # Note: We keep the data but stop updating it
                logger.info(f"Stopped monitoring bucket: {bucket_name}")

        logger.info(f"Updated monitored bucket axes: {old_max} -> {max_axes}")

    def disable_debug_logging(self):
        """Disable all debug logging for maximum performance."""
        self._debug_log_frequency = float("inf")  # Never log debug
        logger.info("Debug logging disabled for maximum performance")

    def get_bucket_summary(self) -> dict[str, dict]:
        """Get a summary of bucket tracking for debugging and analysis."""
        summary = {}

        for bucket_name in self._bucket_tracking:
            summary[bucket_name] = {
                "is_discrete": self._bucket_is_discrete.get(bucket_name, False),
                "num_tasks": len(self._bucket_tracking[bucket_name]),
                "num_completions": sum(self._bucket_completion_counts.get(bucket_name, {}).values()),
                "bins": self._bucket_bins.get(bucket_name, []),
                "completion_history_length": len(self._bucket_completion_history.get(bucket_name, [])),
            }

            # Add sample values for continuous buckets
            if not self._bucket_is_discrete.get(bucket_name, True):
                values = list(self._bucket_tracking[bucket_name].values())
                if values:
                    summary[bucket_name]["sample_values"] = values[:5]  # First 5 values
                    summary[bucket_name]["value_range"] = (min(values), max(values))

        return summary

    # Legacy methods (kept for compatibility)
    def _update_weights(self, child_idx: int, score: float):
        """Update weights - not used in this implementation."""
        pass


class BidirectionalLearningProgress:
    """Tracks bidirectional learning progress using fast and slow exponential moving averages."""

    def __init__(
        self,
        search_space: int | Discrete,
        ema_timescale: float = 0.001,
        progress_smoothing: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
    ) -> None:
        if isinstance(search_space, int):
            search_space = Discrete(search_space)
        assert isinstance(search_space, Discrete), (
            f"search_space must be a Discrete space or int, got {type(search_space)}"
        )
        self._search_space = search_space
        self._num_tasks = max_num_levels = search_space.n
        self._ema_timescale = ema_timescale
        self.progress_smoothing = progress_smoothing
        self.num_active_tasks = int(num_active_tasks)
        self._rand_task_rate = rand_task_rate
        self._sample_threshold = sample_threshold
        self._memory = int(memory)
        self._outcomes = {}
        for i in range(max_num_levels):
            self._outcomes[i] = []
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self._random_baseline = None
        self._task_success_rate = np.zeros(max_num_levels)
        self._mean_samples_per_eval = []
        self._num_nans = []
        self._update_mask = np.ones(max_num_levels).astype(bool)
        # Initialize _sample_levels properly - start with first N tasks
        self._sample_levels = np.arange(min(self.num_active_tasks, max_num_levels)).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        self._task_dist = None
        self._stale_dist = True

    def add_stats(self) -> Dict[str, float]:
        """Return learning progress statistics for logging."""
        stats = {}

        # Ensure _sample_levels is properly initialized
        if self._sample_levels is not None and len(self._sample_levels) > 0:
            stats["lp/num_active_tasks"] = len(self._sample_levels)
        else:
            # Fallback to the configured number of active tasks
            stats["lp/num_active_tasks"] = self.num_active_tasks

        stats["lp/mean_sample_prob"] = float(np.mean(self._task_dist)) if self._task_dist is not None else 0.0
        stats["lp/num_zeros_lp_dist"] = int(np.sum(self._task_dist == 0)) if self._task_dist is not None else 0
        stats["lp/task_1_success_rate"] = float(self._task_success_rate[0]) if len(self._task_success_rate) > 0 else 0.0
        stats[f"lp/task_{self._num_tasks // 2}_success_rate"] = (
            float(self._task_success_rate[self._num_tasks // 2])
            if len(self._task_success_rate) > self._num_tasks // 2
            else 0.0
        )
        stats["lp/last_task_success_rate"] = (
            float(self._task_success_rate[-1]) if len(self._task_success_rate) > 0 else 0.0
        )
        stats["lp/task_success_rate"] = (
            float(np.mean(self._task_success_rate)) if len(self._task_success_rate) > 0 else 0.0
        )
        stats["lp/mean_evals_per_task"] = float(self._mean_samples_per_eval[-1]) if self._mean_samples_per_eval else 0.0
        stats["lp/num_nan_tasks"] = int(self._num_nans[-1]) if self._num_nans else 0
        return stats

    def _update(self):
        """Update learning progress tracking with current task outcome sequences."""
        # Calculate learning progress for each task individually based on their outcome sequences
        task_lp_scores = np.zeros(self._num_tasks)

        for task_id in range(self._num_tasks):
            outcomes = self._outcomes[task_id]
            if len(outcomes) < 2:
                task_lp_scores[task_id] = 0.0
                continue

            # Convert outcomes to numpy array
            outcomes_array = np.array(outcomes)

            # Calculate fast and slow EMAs for this task's outcome sequence
            if self._p_fast is None:
                # Initialize EMAs
                if self._p_fast is None:
                    self._p_fast = np.zeros(self._num_tasks)
                    self._p_slow = np.zeros(self._num_tasks)
                    self._p_true = np.zeros(self._num_tasks)

                # Initialize with current outcome
                self._p_fast[task_id] = outcomes_array[-1]
                self._p_slow[task_id] = outcomes_array[-1]
                self._p_true[task_id] = outcomes_array[-1]
            else:
                # Update EMAs for this task
                current_outcome = outcomes_array[-1]
                self._p_fast[task_id] = (current_outcome * self._ema_timescale) + (
                    self._p_fast[task_id] * (1.0 - self._ema_timescale)
                )
                # Use a slower timescale for the slow EMA (e.g., 10x slower)
                slow_timescale = self._ema_timescale * 0.1
                self._p_slow[task_id] = (current_outcome * slow_timescale) + (
                    self._p_slow[task_id] * (1.0 - slow_timescale)
                )
                self._p_true[task_id] = (current_outcome * self._ema_timescale) + (
                    self._p_true[task_id] * (1.0 - self._ema_timescale)
                )

            # Calculate learning progress as the absolute difference between fast and slow EMAs
            task_lp_scores[task_id] = abs(self._p_fast[task_id] - self._p_slow[task_id])

        # Update task success rates for statistics
        task_success_rates = np.array(
            [
                np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE
                for i in range(self._num_tasks)
            ]
        )
        task_success_rates = np.nan_to_num(task_success_rates, nan=DEFAULT_SUCCESS_RATE)

        # Update statistics
        self._task_success_rate = task_success_rates
        self._num_nans.append(sum(np.isnan(task_success_rates)))
        self._mean_samples_per_eval.append(np.mean([len(self._outcomes[i]) for i in range(self._num_tasks)]))

        # Mark distribution as stale so it gets recalculated when needed
        self._stale_dist = True
        self._task_dist = None

        return task_success_rates

    def collect_data(self, infos):
        """Collect task outcome data for learning progress tracking."""
        if not bool(infos):
            return

        for k, v in infos.items():
            if "tasks" in k:
                task_id = int(k.split("/")[1])
                for res in v:
                    self._outcomes[task_id].append(res)
                    if task_id in self._sample_levels:
                        self._counter[task_id] += 1

    def _learning_progress(self, reweight: bool = False) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages for each task."""
        if self._p_fast is None or self._p_slow is None:
            return np.zeros(self._num_tasks)

        # Calculate learning progress for each task individually
        lp_scores = np.zeros(self._num_tasks)
        for task_id in range(self._num_tasks):
            fast = self._p_fast[task_id]
            slow = self._p_slow[task_id]

            if reweight:
                # Apply reweighting to individual task scores
                fast_reweighted = self._reweight_single(fast)
                slow_reweighted = self._reweight_single(slow)
                lp_scores[task_id] = abs(fast_reweighted - slow_reweighted)
            else:
                lp_scores[task_id] = abs(fast - slow)

        return lp_scores

    def _reweight_single(self, prob: float) -> float:
        """Apply progress smoothing reweighting to a single probability value."""
        numerator = prob * (1.0 - self.progress_smoothing)
        denominator = prob + self.progress_smoothing * (1.0 - 2.0 * prob)

        # Handle division by zero
        if denominator <= 0:
            return 1.0

        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self._num_tasks) / self._num_tasks
        learning_progress = self._learning_progress()

        # Only include tasks with actual learning progress (lp > 0)
        # Remove the condition that includes tasks with _p_true > 0 but lp = 0
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0]

        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        # Apply sigmoid with scaling to make it less heavy-handed
        # Scale the learning progress scores to make sigmoid less aggressive
        scale_factor = 100.0  # Increase this to make sigmoid less aggressive
        scaled_probs = subprobs * scale_factor
        subprobs = self._sigmoid(scaled_probs)

        # Normalize to sum to 1, handling zero sum case
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            # If all probabilities are zero, use uniform distribution
            subprobs = np.ones_like(subprobs) / len(subprobs)

        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False

        # Truncate outcomes to respect memory limit
        for i in range(self._num_tasks):
            self._outcomes[i] = self._outcomes[i][-self._memory :]
        return self._task_dist

    def _sample_tasks(self):
        """Sample active tasks based on current task distribution."""
        sample_levels = []
        self._update_mask = np.zeros(self._num_tasks).astype(bool)

        # Ensure task_dist is valid
        if self._task_dist is None or len(self._task_dist) == 0:
            # Use uniform distribution if task_dist is not available
            task_dist = np.ones(self._num_tasks) / self._num_tasks
        else:
            task_dist = self._task_dist.copy()

        # Ensure task_dist sums to 1
        sum_dist = np.sum(task_dist)
        if sum_dist <= 0:
            task_dist = np.ones(self._num_tasks) / self._num_tasks
        else:
            task_dist = task_dist / sum_dist

        for _i in range(self.num_active_tasks):
            if np.random.rand() < self._rand_task_rate:
                level = np.random.choice(range(self._num_tasks))
            else:
                try:
                    level = np.random.choice(range(self._num_tasks), p=task_dist)
                except ValueError as e:
                    logger.warning(f"Error in np.random.choice: {e}, using uniform distribution")
                    level = np.random.choice(range(self._num_tasks))
            sample_levels.append(level)
            self._update_mask[level] = True
        self._sample_levels = np.array(sample_levels).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        return self._sample_levels

    def calculate_dist(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate task distribution and sample levels based on learning progress."""
        if all([v < self._sample_threshold for k, v in self._counter.items()]) and self._random_baseline is not None:
            # Ensure we have valid task_dist and sample_levels
            if self._task_dist is None or len(self._task_dist) == 0:
                self._task_dist = np.ones(self._num_tasks) / self._num_tasks
            if self._sample_levels is None or len(self._sample_levels) == 0:
                self._sample_levels = np.arange(self._num_tasks).astype(np.int32)
            return self._task_dist, self._sample_levels

        self._task_success_rate = self._update()
        task_dist = self._sample_distribution()
        tasks = self._sample_tasks()

        return task_dist, tasks

    # No _expand_search_space method needed - search space is fixed at pool size

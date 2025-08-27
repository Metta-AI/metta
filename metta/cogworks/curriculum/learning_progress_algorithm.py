"""
Learning Progress Curriculum Algorithm for Curriculum.

This module implements the learning progress algorithm as a CurriculumAlgorithm
that can be used with Curriculum nodes to adaptively sample tasks based on
bidirectional learning progress tracking with local task memory pool.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from gym.spaces import Discrete
from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.curriculum import CurriculumAlgorithm, CurriculumAlgorithmHypers, CurriculumTask

logger = logging.getLogger(__name__)

DEFAULT_SUCCESS_RATE = 0.5


class LearningProgressHypers(CurriculumAlgorithmHypers):
    """Hyperparameters for LearningProgressAlgorithm."""

    type: str = "learning_progress"
    ema_timescale: float = Field(default=0.001, description="EMA timescale for learning progress")
    pool_size: int = Field(default=16, description="Size of the task pool")
    sample_size: int = Field(default=8, description="Number of tasks to sample")
    max_samples: int = Field(default=10, description="Maximum samples before eviction")
    exploration_bonus: float = Field(default=0.1, description="Exploration bonus for sampling")

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
    """Learning progress algorithm that manages a unified pool of tasks."""

    def __init__(self, num_tasks: int, hypers: LearningProgressHypers):
        # Don't initialize weights since this algorithm uses its own sampling strategy
        super().__init__(num_tasks, hypers, initialize_weights=False)

        self.hypers = hypers

        # Initialize bidirectional learning progress tracker
        self._lp_tracker = BidirectionalLearningProgress(
            search_space=num_tasks,
            ema_timescale=hypers.ema_timescale,
            num_active_tasks=hypers.pool_size,
        )

        # Task management
        self._task_memory: Dict[int, Tuple[int, str, int, float, float, float]] = {}
        self._task_objects: Dict[int, CurriculumTask] = {}
        self._task_id_to_index: Dict[int, int] = {}
        self._next_index: int = 0

    def _update_weights(self, child_idx: int, score: float):
        """Update weights - not used in this implementation."""
        pass

    def get_task_from_pool(self, task_generator, rng) -> CurriculumTask:
        """Get a task from the unified pool, creating or evicting as needed."""
        # Create new task if pool not full
        if len(self._task_memory) < self.hypers.pool_size:
            task_id = self._generate_task_id(rng)
            env_cfg = task_generator.get_task(task_id)
            task = CurriculumTask(task_id, env_cfg)
            self._add_task_to_pool(task_id, task)
            return task

        # Sample from existing pool
        selected_task_id = self._sample_from_pool()
        if selected_task_id in self._task_objects:
            return self._task_objects[selected_task_id]

        # Fallback: create new task and evict one
        self._evict_from_pool()
        task_id = self._generate_task_id(rng)
        env_cfg = task_generator.get_task(task_id)
        task = CurriculumTask(task_id, env_cfg)
        self._add_task_to_pool(task_id, task)
        return task

    def _generate_task_id(self, rng) -> int:
        """Generate a unique task ID."""
        while True:
            task_id = rng.randint(0, 1000000)
            if task_id not in self._task_memory:
                return task_id

    def _add_task_to_pool(self, task_id: int, task: CurriculumTask):
        """Add a task to the unified pool."""
        if len(self._task_memory) >= self.hypers.pool_size:
            self._evict_from_pool()

        # Assign sequential index for learning progress tracking
        self._task_id_to_index[task_id] = self._next_index
        self._next_index += 1

        # Add to memory and objects
        self._task_memory[task_id] = (task_id, "default", 0, 0.0, 0.0, 0.0)
        self._task_objects[task_id] = task

    def _sample_from_pool(self) -> int:
        """Sample a task from the pool based on learning progress scores."""
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

    def _evict_from_pool(self):
        """Evict a task from the pool based on learning progress and sample count."""
        if not self._task_memory:
            return

        # Find task with lowest LP score that has enough samples
        worst_task_id = None
        worst_score = float("inf")

        for task_id, (_, _, sample_count, _, _, lp_score) in self._task_memory.items():
            if sample_count >= self.hypers.max_samples and lp_score < worst_score:
                worst_score = lp_score
                worst_task_id = task_id

        if worst_task_id is not None:
            self._evict_task(worst_task_id)

    def _evict_task(self, task_id: int):
        """Evict a task from the pool."""
        if task_id in self._task_memory:
            del self._task_memory[task_id]
            if task_id in self._task_objects:
                del self._task_objects[task_id]

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance and learning progress."""
        if task_id not in self._task_memory:
            return

        # Update task memory
        seed, family, sample_count, current_score, recent_score, lp_score = self._task_memory[task_id]
        self._task_memory[task_id] = (seed, family, sample_count + 1, recent_score, score, lp_score)

        # Update learning progress tracker
        task_index = self._task_id_to_index.get(task_id, 0)
        self._lp_tracker._outcomes[task_index].append(score)
        self._lp_tracker._update()

        # Update LP score in memory
        raw_lp_scores = self._lp_tracker._learning_progress()
        if task_index < len(raw_lp_scores):
            new_lp_score = float(raw_lp_scores[task_index])
            self._task_memory[task_id] = (seed, family, sample_count + 1, recent_score, score, new_lp_score)

    def _get_task_lp_score(self, task_id: int) -> float:
        """Get the learning progress score for a specific task."""
        if task_id not in self._task_memory:
            return 0.0

        # Force update of learning progress calculation
        self._lp_tracker._update()

        # Get the raw learning progress score, not the distribution value
        task_index = self._task_id_to_index.get(task_id, 0)
        raw_lp_scores = self._lp_tracker._learning_progress()

        if task_index < len(raw_lp_scores):
            return float(raw_lp_scores[task_index])
        else:
            return 0.0

    def stats(self) -> dict[str, float]:
        """Return unified statistics for the pool."""
        # Get base learning progress stats
        base_stats = self._lp_tracker.add_stats()

        return base_stats


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
        self._sample_levels = np.arange(max_num_levels).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        self._task_dist = None
        self._stale_dist = True

    def add_stats(self) -> Dict[str, float]:
        """Return learning progress statistics for logging."""
        stats = {}
        stats["lp/num_active_tasks"] = len(self._sample_levels) if self._sample_levels is not None else 0
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

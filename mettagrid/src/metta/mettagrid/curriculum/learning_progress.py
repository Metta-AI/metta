from __future__ import annotations

import logging
from multiprocessing import Array, Manager, Value
from typing import Dict, Tuple

import numpy as np
from gymnasium.spaces import Discrete
from omegaconf import DictConfig

from metta.mettagrid.curriculum.random import RandomCurriculum

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75


class LearningProgressCurriculum(RandomCurriculum):
    """Curriculum that adaptively samples tasks based on learning progress."""

    def __init__(
        self,
        tasks: Dict[str, float] | DictConfig[str, float],
        env_overrides: DictConfig | None = None,
        ema_timescale: float = 0.001,
        progress_smoothing: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
    ):
        super().__init__(tasks, env_overrides)

        # Initialize learning progress tracker
        search_space_size = len(tasks)
        self._lp_tracker = BidirectionalLearningProgress(
            search_space=search_space_size,
            ema_timescale=ema_timescale,
            progress_smoothing=progress_smoothing,
            num_active_tasks=num_active_tasks,
            rand_task_rate=rand_task_rate,
            sample_threshold=sample_threshold,
            memory=memory,
        )

        logger.info(f"LearningProgressCurriculum initialized with {search_space_size} tasks")

    def complete_task(self, id: str, score: float):
        """Complete a task and update learning progress tracking."""
        # Convert score to success rate (assuming score is between 0 and 1)
        success_rate = max(0.0, min(1.0, score))

        # Get task index for learning progress tracking
        task_idx = list(self._curricula.keys()).index(id)

        # Collect data for learning progress
        self._lp_tracker.collect_data({f"tasks/{task_idx}": [success_rate]})

        # Update task weights based on learning progress
        lp_weights, _ = self._lp_tracker.calculate_dist()

        # Update weights based on learning progress
        for i, task_id in enumerate(self._curricula.keys()):
            if i < len(lp_weights):
                self._task_weights[task_id] = lp_weights[i]

        # Normalize weights
        total_weight = sum(self._task_weights.values())
        if total_weight > 0:
            self._task_weights = {k: v / total_weight for k, v in self._task_weights.items()}

        super().complete_task(id, score)

    def stats(self) -> Dict[str, float]:
        """Get learning progress statistics for logging."""
        return self._lp_tracker.add_stats()


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
        max_num_levels = int(search_space.n)  # Convert to int for Array

        # Create multiprocessing manager for shared data structures
        self._manager = Manager()

        # Store simple values in shared memory
        self._num_tasks = Value("i", max_num_levels)
        self._ema_timescale = Value("d", ema_timescale)
        self.progress_smoothing = Value("d", progress_smoothing)
        self.num_active_tasks = Value("i", int(num_active_tasks))
        self._rand_task_rate = Value("d", rand_task_rate)
        self._sample_threshold = Value("i", sample_threshold)
        self._memory = Value("i", int(memory))
        self._stale_dist = Value("b", True)

        # Use managed dict for outcomes
        self._outcomes = self._manager.dict()
        for i in range(max_num_levels):
            self._outcomes[i] = self._manager.list()

        # Create shared arrays for numpy arrays
        # Initialize as None first, will create shared memory when needed
        self._p_fast_shm = None
        self._p_slow_shm = None
        self._p_true_shm = None
        self._random_baseline_shm = None
        self._task_dist_shm = None

        # Create shared arrays that are always initialized
        self._task_success_rate = Array("d", max_num_levels)
        self._update_mask = Array("b", max_num_levels)

        # Initialize update mask to all True
        with self._update_mask.get_lock():
            for i in range(max_num_levels):
                self._update_mask[i] = True

        # Use managed lists for variable-length data
        self._mean_samples_per_eval = self._manager.list()
        self._num_nans = self._manager.list()
        self._sample_levels = Array("i", max_num_levels)

        # Initialize sample levels
        with self._sample_levels.get_lock():
            for i in range(max_num_levels):
                self._sample_levels[i] = i

        # Use managed dict for counter
        self._counter = self._manager.dict({i: 0 for i in range(max_num_levels)})

        # Track dtypes for shared arrays
        self._shared_array_dtypes = {}

    def _get_num_tasks(self) -> int:
        """Get the number of tasks from shared memory."""
        return self._num_tasks.value

    def _get_numpy_array_from_shared(self, shared_array, dtype=None, name=None):
        """Convert shared array to numpy array."""
        if shared_array is None:
            return None

        if dtype is None:
            # Try to get stored dtype if name is provided
            if name and name in self._shared_array_dtypes:
                dtype = self._shared_array_dtypes[name]
            else:
                dtype = np.float64  # default

        with shared_array.get_lock():
            return np.frombuffer(shared_array.get_obj(), dtype=dtype)

    def _create_or_update_shared_array(self, name: str, data: np.ndarray | None):
        """Create or update a shared array."""
        if data is None:
            return None

        # Get the shared memory attribute name
        shm_attr = f"_{name}_shm"
        existing_shared_array = getattr(self, shm_attr)

        # If shared memory doesn't exist or size mismatch, create new array
        if existing_shared_array is None or len(existing_shared_array) != len(data):
            if data.dtype == np.float64:
                shared_array = Array("d", len(data))
                self._shared_array_dtypes[name] = np.float64
            elif data.dtype == np.float32:
                shared_array = Array("f", len(data))
                self._shared_array_dtypes[name] = np.float32
            else:
                shared_array = Array("d", len(data))
                self._shared_array_dtypes[name] = np.float64
            setattr(self, shm_attr, shared_array)
        else:
            shared_array = existing_shared_array

        # Update the shared array with data
        with shared_array.get_lock():
            # Use the stored dtype
            array_dtype = self._shared_array_dtypes.get(name, np.float64)

            np_array = np.frombuffer(shared_array.get_obj(), dtype=array_dtype)
            # Convert data to the correct dtype before assignment
            np_array[:] = data.astype(array_dtype)

        return shared_array

    def add_stats(self) -> Dict[str, float]:
        """Return learning progress statistics for logging."""
        stats = {}

        # Get sample levels as numpy array
        sample_levels = np.array(self._sample_levels[:])
        stats["lp/num_active_tasks"] = len(sample_levels)

        # Get task_dist as numpy array if it exists
        if self._task_dist_shm is not None:
            task_dist = self._get_numpy_array_from_shared(self._task_dist_shm, dtype=np.float32, name="task_dist")
            if task_dist is not None:
                stats["lp/mean_sample_prob"] = np.mean(task_dist)
                stats["lp/num_zeros_lp_dist"] = np.sum(task_dist == 0)
            else:
                stats["lp/mean_sample_prob"] = 0.0
                stats["lp/num_zeros_lp_dist"] = 0
        else:
            stats["lp/mean_sample_prob"] = 0.0
            stats["lp/num_zeros_lp_dist"] = 0

        # Get task success rate as numpy array
        task_success_rate = np.array(self._task_success_rate[:])
        stats["lp/task_1_success_rate"] = task_success_rate[0]
        stats[f"lp/task_{self._get_num_tasks() // 2}_success_rate"] = task_success_rate[self._get_num_tasks() // 2]
        stats["lp/last_task_success_rate"] = task_success_rate[-1]
        stats["lp/task_success_rate"] = np.mean(task_success_rate)

        if len(self._mean_samples_per_eval) > 0:
            stats["lp/mean_evals_per_task"] = self._mean_samples_per_eval[-1]
        else:
            stats["lp/mean_evals_per_task"] = 0.0

        if len(self._num_nans) > 0:
            stats["lp/num_nan_tasks"] = self._num_nans[-1]
        else:
            stats["lp/num_nan_tasks"] = 0

        return stats

    def _update(self):
        """Update learning progress tracking with current task success rates."""
        num_tasks = self._get_num_tasks()
        task_success_rates = np.array(
            [
                np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE
                for i in range(num_tasks)
            ]
        )
        # Handle NaN values in task success rates (empty lists)
        task_success_rates = np.nan_to_num(task_success_rates, nan=DEFAULT_SUCCESS_RATE)

        # Get arrays from shared memory
        random_baseline = self._get_numpy_array_from_shared(self._random_baseline_shm, name="random_baseline")
        update_mask_raw = np.array(self._update_mask[:])

        # Ensure update_mask has the right size and only has True values for valid indices
        if len(update_mask_raw) != num_tasks:
            update_mask = np.ones(num_tasks).astype(bool)
        else:
            # Ensure no True values beyond num_tasks and convert to bool
            update_mask = update_mask_raw[:num_tasks].astype(bool)

        if random_baseline is None:
            random_baseline = np.minimum(task_success_rates, RANDOM_BASELINE_CAP)
            self._random_baseline_shm = self._create_or_update_shared_array("random_baseline", random_baseline)
        elif len(random_baseline) != num_tasks:
            # If size mismatch, recreate random_baseline
            random_baseline = np.minimum(task_success_rates, RANDOM_BASELINE_CAP)
            self._random_baseline_shm = self._create_or_update_shared_array("random_baseline", random_baseline)

        # Ensure update_mask only has True values for indices that exist in random_baseline
        if len(random_baseline) < num_tasks:
            # Only keep True values for indices that exist
            update_mask = update_mask & (np.arange(len(update_mask)) < len(random_baseline))

        # Handle division by zero in normalization
        denominator = 1.0 - random_baseline[update_mask]
        denominator = np.where(denominator <= 0, 1.0, denominator)

        normalized_task_success_rates = (
            np.maximum(
                task_success_rates[update_mask] - random_baseline[update_mask],
                np.zeros(task_success_rates[update_mask].shape),
            )
            / denominator
        )

        # Get current values from shared memory
        p_fast = self._get_numpy_array_from_shared(self._p_fast_shm, name="p_fast")
        p_slow = self._get_numpy_array_from_shared(self._p_slow_shm, name="p_slow")
        p_true = self._get_numpy_array_from_shared(self._p_true_shm, name="p_true")

        if p_fast is None:
            # First time initialization - create arrays with size matching update_mask
            update_indices = np.where(update_mask)[0]
            self._p_fast_shm = self._create_or_update_shared_array("p_fast", normalized_task_success_rates)
            self._p_slow_shm = self._create_or_update_shared_array("p_slow", normalized_task_success_rates)
            self._p_true_shm = self._create_or_update_shared_array("p_true", task_success_rates[update_mask])
        else:
            # Create arrays matching current sizes
            update_indices = np.where(update_mask)[0]

            # If sizes don't match, recreate arrays
            if len(p_fast) != len(update_indices):
                self._p_fast_shm = self._create_or_update_shared_array("p_fast", normalized_task_success_rates)
                self._p_slow_shm = self._create_or_update_shared_array("p_slow", normalized_task_success_rates)
                self._p_true_shm = self._create_or_update_shared_array("p_true", task_success_rates[update_mask])
            else:
                # Update shared arrays
                p_fast[:] = (normalized_task_success_rates * self._ema_timescale.value) + (
                    p_fast * (1.0 - self._ema_timescale.value)
                )
                p_slow[:] = (p_fast * self._ema_timescale.value) + (p_slow * (1.0 - self._ema_timescale.value))
                p_true[:] = (task_success_rates[update_mask] * self._ema_timescale.value) + (
                    p_true * (1.0 - self._ema_timescale.value)
                )

                # Write back to shared memory
                self._create_or_update_shared_array("p_fast", p_fast)
                self._create_or_update_shared_array("p_slow", p_slow)
                self._create_or_update_shared_array("p_true", p_true)

        self._stale_dist.value = True
        self._task_dist_shm = None

        return task_success_rates

    def collect_data(self, infos):
        """Collect task outcome data for learning progress tracking."""
        if not bool(infos):
            return

        # Get sample levels as a list for membership checking
        sample_levels_list = list(self._sample_levels[: self.num_active_tasks.value])

        for k, v in infos.items():
            if "tasks" in k:
                task_id = int(k.split("/")[1])
                for res in v:
                    self._outcomes[task_id].append(res)
                    if task_id in sample_levels_list:
                        # Ensure task_id exists in counter before incrementing
                        if task_id not in self._counter:
                            self._counter[task_id] = 0
                        self._counter[task_id] += 1

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as the difference between fast and slow moving averages."""
        p_fast = self._get_numpy_array_from_shared(self._p_fast_shm, name="p_fast")
        p_slow = self._get_numpy_array_from_shared(self._p_slow_shm, name="p_slow")

        if p_fast is None or p_slow is None:
            return np.zeros(self._get_num_tasks())

        # If arrays don't match task count, return zeros for missing tasks
        num_tasks = self._get_num_tasks()
        if len(p_fast) < num_tasks:
            # Pad with zeros
            fast_padded = np.zeros(num_tasks)
            slow_padded = np.zeros(num_tasks)
            fast_padded[: len(p_fast)] = p_fast
            slow_padded[: len(p_slow)] = p_slow
            p_fast = fast_padded
            p_slow = slow_padded

        fast = self._reweight(p_fast) if reweight else p_fast
        slow = self._reweight(p_slow) if reweight else p_slow
        return abs(fast - slow)

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting to probability values."""
        smoothing = self.progress_smoothing.value
        numerator = probs * (1.0 - smoothing)
        denominator = probs + smoothing * (1.0 - 2.0 * probs)

        # Handle division by zero
        denominator = np.where(denominator <= 0, 1.0, denominator)
        result = numerator / denominator

        return result

    def _sigmoid(self, x: np.ndarray):
        """Apply sigmoid function to array values."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self._get_num_tasks()) / self._get_num_tasks()
        learning_progress = self._learning_progress()

        # Get p_true array from shared memory
        p_true = self._get_numpy_array_from_shared(self._p_true_shm, name="p_true")
        if p_true is None:
            p_true = np.zeros(self._get_num_tasks())
        elif len(p_true) < self._get_num_tasks():
            # Pad with zeros if p_true is smaller than expected
            p_true_padded = np.zeros(self._get_num_tasks())
            p_true_padded[: len(p_true)] = p_true
            p_true = p_true_padded

        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        std = np.std(subprobs)
        if std > 0:
            subprobs = (subprobs - np.mean(subprobs)) / std
        else:
            # If all values are the same, keep them as is
            subprobs = subprobs - np.mean(subprobs)

        subprobs = self._sigmoid(subprobs)

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

        self._task_dist_shm = self._create_or_update_shared_array("task_dist", task_dist.astype(np.float32))
        self._stale_dist.value = False

        out_vec = [
            np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE
            for i in range(self._get_num_tasks())
        ]
        out_vec = [DEFAULT_SUCCESS_RATE if np.isnan(x) else x for x in out_vec]  # Handle NaN in outcomes
        self._num_nans.append(sum(np.isnan(out_vec)))

        # Update task success rate shared array
        with self._task_success_rate.get_lock():
            task_success_rate_array = np.frombuffer(self._task_success_rate.get_obj(), dtype=np.float64)
            task_success_rate_array[:] = out_vec

        self._mean_samples_per_eval.append(np.mean([len(self._outcomes[i]) for i in range(self._get_num_tasks())]))

        # Trim outcomes to memory limit
        memory_limit = self._memory.value
        for i in range(self._get_num_tasks()):
            if len(self._outcomes[i]) > memory_limit:
                # Convert to list, slice, and reassign
                self._outcomes[i] = self._manager.list(list(self._outcomes[i])[-memory_limit:])

        # Get task distribution as numpy array
        task_dist_array = self._get_numpy_array_from_shared(self._task_dist_shm, dtype=np.float32, name="task_dist")
        return task_dist_array

    def _sample_tasks(self):
        """Sample active tasks based on current task distribution."""
        sample_levels = []
        num_tasks = self._get_num_tasks()

        # Create new update mask array
        update_mask = np.zeros(num_tasks).astype(bool)

        # Get task distribution from shared memory
        task_dist = self._get_numpy_array_from_shared(self._task_dist_shm, dtype=np.float32, name="task_dist")

        # Ensure task_dist is valid
        if task_dist is None or len(task_dist) == 0:
            # Use uniform distribution if task_dist is not available
            task_dist = np.ones(num_tasks) / num_tasks

        # Ensure task_dist sums to 1
        sum_dist = np.sum(task_dist)
        if sum_dist <= 0:
            task_dist = np.ones(num_tasks) / num_tasks
        else:
            task_dist = task_dist / sum_dist

        # Get values from shared memory
        num_active_tasks = min(self.num_active_tasks.value, num_tasks)  # Can't have more active tasks than total tasks
        rand_task_rate = self._rand_task_rate.value

        for _i in range(num_active_tasks):
            if np.random.rand() < rand_task_rate:
                level = np.random.choice(range(num_tasks))
            else:
                try:
                    level = np.random.choice(range(num_tasks), p=task_dist)
                except ValueError as e:
                    logger.warning(f"Error in np.random.choice: {e}, using uniform distribution")
                    level = np.random.choice(range(num_tasks))
            sample_levels.append(level)
            update_mask[level] = True

        # Update shared arrays
        with self._sample_levels.get_lock():
            sample_levels_array = np.frombuffer(self._sample_levels.get_obj(), dtype=np.int32)
            # Only update up to the size of the array
            num_to_update = min(len(sample_levels), len(sample_levels_array))
            sample_levels_array[:num_to_update] = sample_levels[:num_to_update]

        with self._update_mask.get_lock():
            update_mask_array = np.frombuffer(self._update_mask.get_obj(), dtype=bool)
            update_mask_array[:] = update_mask

        # Reset counters only for newly sampled tasks, preserving others
        # This prevents KeyError in multiprocessing scenarios
        for level in sample_levels:
            self._counter[level] = 0
        return np.array(sample_levels).astype(np.int32)

    def calculate_dist(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate task distribution and sample levels based on learning progress."""
        # Check if we should update based on sample threshold
        sample_threshold = self._sample_threshold.value
        should_update = any([v >= sample_threshold for k, v in self._counter.items()])

        if not should_update and self._random_baseline_shm is not None:
            # Ensure we have valid task_dist and sample_levels
            task_dist = self._get_numpy_array_from_shared(self._task_dist_shm, dtype=np.float32, name="task_dist")
            if task_dist is None or len(task_dist) == 0:
                task_dist = np.ones(self._get_num_tasks()) / self._get_num_tasks()

            sample_levels = np.array(self._sample_levels[: self.num_active_tasks.value])
            return task_dist, sample_levels

        # Update without overwriting self._task_success_rate
        self._update()
        task_dist = self._sample_distribution()
        tasks = self._sample_tasks()

        return task_dist, tasks

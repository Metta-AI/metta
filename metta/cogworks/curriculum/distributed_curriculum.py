"""Distributed Curriculum Management with Shared Memory.

This module provides a distributed curriculum manager that allows multiple workers
to share learning progress information through shared memory with minimal locking.
"""

import multiprocessing as mp
from typing import Union

import numpy as np

from .learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressHypers


class SharedMemoryArray:
    """A wrapper around shared memory arrays for easy access and management."""

    def __init__(self, shape: tuple, dtype: Union[np.dtype, type] = np.float32):
        """Initialize a shared memory array.

        Args:
            shape: Shape of the array
            dtype: Data type of the array
        """
        self.shape = shape
        self.dtype = dtype
        self.size = int(np.prod(shape))

        # Create shared memory
        if dtype == np.int32:
            self.shared_memory = mp.RawArray("i", self.size)
        else:
            self.shared_memory = mp.RawArray("f", self.size)
        self.array = np.frombuffer(self.shared_memory, dtype=dtype).reshape(shape)

    def __getitem__(self, key):
        """Get item from shared array."""
        return self.array[key]

    def __setitem__(self, key, value):
        """Set item in shared array."""
        self.array[key] = value

    def copy(self) -> np.ndarray:
        """Return a copy of the current array state."""
        return self.array.copy()


class DistributedCurriculumManager:
    """Manages distributed curriculum learning progress across multiple workers.

    This class provides a hybrid approach:
    - Lock-free local updates for maximum performance
    - Periodic global aggregation with minimal locking
    - Shared memory for efficient data access
    """

    def __init__(self, num_tasks: int, num_workers: int, worker_id: int, aggregation_interval: int = 10):
        """Initialize the distributed curriculum manager.

        Args:
            num_tasks: Number of tasks in the curriculum
            num_workers: Number of workers sharing the curriculum
            worker_id: ID of this worker (0 to num_workers-1)
            aggregation_interval: How often to perform global aggregation
        """
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f"worker_id must be between 0 and {num_workers - 1}")

        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.aggregation_interval = aggregation_interval

        # Shared memory arrays (lock-free updates)
        self.shared_counts = SharedMemoryArray((num_workers, num_tasks), dtype=np.int32)
        self.shared_success_sums = SharedMemoryArray((num_workers, num_tasks), dtype=np.float32)

        # Single lock for global aggregation
        self.global_lock = mp.Lock()

        # Local learning progress algorithm with default hypers
        default_hypers = LearningProgressHypers()
        self.lp_algorithm = LearningProgressAlgorithm(num_tasks, hypers=default_hypers)

        # Local aggregation counter
        self.aggregation_counter = 0

        # Track which tasks have been updated since last aggregation
        self.updated_tasks = set()

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance (lock-free local update).

        Args:
            task_id: ID of the task that was completed
            score: Success rate (0.0 to 1.0) for the task
        """
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"task_id must be between 0 and {self.num_tasks - 1}")

        # Lock-free local update
        self.shared_counts[self.worker_id, task_id] += 1
        self.shared_success_sums[self.worker_id, task_id] += score

        # Track that this task was updated
        self.updated_tasks.add(task_id)

        # Increment aggregation counter
        self.aggregation_counter += 1

        # Periodic global aggregation
        if self.aggregation_counter >= self.aggregation_interval:
            self._perform_global_aggregation()

    def _perform_global_aggregation(self):
        """Perform global aggregation with lock."""
        with self.global_lock:
            # Only update tasks that have been modified
            for task_id in self.updated_tasks:
                global_success_rate = self._calculate_global_success_rate(task_id)
                self.lp_algorithm.update(task_id, global_success_rate)

            # Reset counters
            self.aggregation_counter = 0
            self.updated_tasks.clear()

    def _calculate_global_success_rate(self, task_id: int) -> float:
        """Calculate global success rate for a task across all workers.

        Args:
            task_id: ID of the task

        Returns:
            Global success rate (0.0 to 1.0)
        """
        total_count = np.sum(self.shared_counts.array[:, task_id])
        total_success = np.sum(self.shared_success_sums.array[:, task_id])

        if total_count > 0:
            return float(total_success / total_count)
        else:
            return 0.0

    def get_task_weights(self) -> np.ndarray:
        """Get current task weights from the learning progress algorithm.

        Returns:
            Array of task weights
        """
        return self.lp_algorithm.weights

    def get_task_probabilities(self) -> np.ndarray:
        """Get current task sampling probabilities.

        Returns:
            Array of task probabilities
        """
        return self.lp_algorithm.probabilities

    def sample_task(self) -> int:
        """Sample a task based on current learning progress.

        Returns:
            Task ID to sample
        """
        return self.lp_algorithm.sample_idx()

    def get_stats(self) -> dict:
        """Get statistics about the distributed curriculum.

        Returns:
            Dictionary containing statistics
        """
        # Get local stats
        local_stats = {
            "worker_id": self.worker_id,
            "num_tasks": self.num_tasks,
            "num_workers": self.num_workers,
            "aggregation_interval": self.aggregation_interval,
            "aggregation_counter": self.aggregation_counter,
            "updated_tasks_count": len(self.updated_tasks),
        }

        # Get learning progress stats
        lp_stats = self.lp_algorithm.stats(prefix="distributed_lp/")

        # Get global stats (with lock to ensure consistency)
        with self.global_lock:
            total_episodes = int(np.sum(self.shared_counts.array))
            total_success = float(np.sum(self.shared_success_sums.array))
            global_stats = {
                "global_total_episodes": total_episodes,
                "global_mean_success_rate": total_success / max(1, total_episodes),
            }

        return {**local_stats, **lp_stats, **global_stats}

    def force_global_aggregation(self):
        """Force immediate global aggregation (useful for debugging or final stats)."""
        self._perform_global_aggregation()


class DistributedCurriculumConfig:
    """Configuration for distributed curriculum management."""

    def __init__(self, num_tasks: int, num_workers: int, worker_id: int, aggregation_interval: int = 10):
        """Initialize distributed curriculum configuration.

        Args:
            num_tasks: Number of tasks in the curriculum
            num_workers: Number of workers sharing the curriculum
            worker_id: ID of this worker
            aggregation_interval: How often to perform global aggregation
        """
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.aggregation_interval = aggregation_interval

    def create(self) -> DistributedCurriculumManager:
        """Create a distributed curriculum manager instance.

        Returns:
            Configured distributed curriculum manager
        """
        return DistributedCurriculumManager(
            num_tasks=self.num_tasks,
            num_workers=self.num_workers,
            worker_id=self.worker_id,
            aggregation_interval=self.aggregation_interval,
        )

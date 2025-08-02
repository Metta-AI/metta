"""Curriculum Manager for distributed training - maintains a pool of tasks in shared memory."""

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Shared memory representation of a task."""

    task_id: int
    score: float = 0.0
    num_runs: int = 0
    last_update: float = 0.0  # timestamp
    reward_mean: float = 0.0
    reward_var: float = 0.0


class CurriculumManager:
    """
    Server-side curriculum manager that maintains a pool of tasks in shared memory.

    The manager maintains a fixed-size pool of tasks, where each task has:
    - task_id: unique identifier used to seed task generation
    - score: difficulty score (lower = easier/less interesting)
    - num_runs: number of times the task has been completed
    - reward statistics: mean and variance of episode rewards

    Tasks with low scores (easy tasks) are replaced with new random tasks
    to maintain curriculum progression.

        The default shared memory name is "metta/curriculum/tasks". If you need
    multiple curriculums, use different names like:
    - "metta/curriculum/navigation"
    - "metta/curriculum/combat"
    - "metta/curriculum/exploration"

    Note: Special characters (/, .) are automatically converted to underscores
    for compatibility with shared memory naming restrictions.
    """

    # TaskState fields in shared memory array
    TASK_ID_IDX = 0
    SCORE_IDX = 1
    NUM_RUNS_IDX = 2
    LAST_UPDATE_IDX = 3
    REWARD_MEAN_IDX = 4
    REWARD_VAR_IDX = 5
    FIELDS_PER_TASK = 6

    def __init__(
        self,
        pool_size: int = 1024,
        min_runs: int = 10,
        name: Optional[str] = None,
    ):
        """
        Initialize the curriculum manager.

        Args:
            pool_size: Number of tasks in the pool
            min_runs: Minimum runs before a task can be replaced
            name: Name for the shared memory (defaults to "metta/curriculum/tasks")
        """
        self.pool_size = pool_size
        self.min_runs = min_runs

        # Use default name if not provided
        if name is None:
            name = "metta/curriculum/tasks"
        self.name = name

        # Create named shared memory for pool - 6 floats per TaskState
        shm_size = pool_size * self.FIELDS_PER_TASK * 8  # 8 bytes per float
        # Ensure name is valid for shared memory (no special chars that might be interpreted as paths)
        safe_name = name.replace("/", "_").replace(".", "_")

        # Try to clean up any existing shared memory with this name
        try:
            existing_shm = shared_memory.SharedMemory(name=f"{safe_name}_pool")
            existing_shm.close()
            existing_shm.unlink()
            logger.debug(f"Cleaned up existing shared memory: {safe_name}_pool")
        except FileNotFoundError:
            pass  # No existing shared memory, which is fine

        self._shm = shared_memory.SharedMemory(create=True, size=shm_size, name=f"{safe_name}_pool")
        self._pool = np.ndarray((pool_size * self.FIELDS_PER_TASK,), dtype=np.float64, buffer=self._shm.buf)

        # Create named lock
        self._lock = mp.Lock()

        # Create named shared memory for statistics
        try:
            existing_stats_shm = shared_memory.SharedMemory(name=f"{safe_name}_stats")
            existing_stats_shm.close()
            existing_stats_shm.unlink()
            logger.debug(f"Cleaned up existing shared memory: {safe_name}_stats")
        except FileNotFoundError:
            pass  # No existing shared memory, which is fine

        self._stats_shm = shared_memory.SharedMemory(create=True, size=16, name=f"{safe_name}_stats")  # 2 * 8 bytes
        self._stats = np.ndarray((2,), dtype=np.int64, buffer=self._stats_shm.buf)
        self._stats[0] = 0  # total_completions
        self._stats[1] = 0  # total_replacements

        # Initialize pool with random task IDs
        self._initialize_pool()

        logger.info(f"CurriculumManager initialized with pool_size={pool_size}, min_runs={min_runs}, name={name}")

    def _initialize_pool(self):
        """Initialize pool with random task IDs."""
        with self._lock:
            for i in range(self.pool_size):
                self._set_task_state(i, TaskState(task_id=np.random.randint(0, 2**31 - 1), last_update=time.time()))

    def _get_task_state(self, slot_id: int) -> TaskState:
        """Get TaskState from shared memory."""
        base_idx = slot_id * self.FIELDS_PER_TASK
        return TaskState(
            task_id=int(self._pool[base_idx + self.TASK_ID_IDX]),
            score=self._pool[base_idx + self.SCORE_IDX],
            num_runs=int(self._pool[base_idx + self.NUM_RUNS_IDX]),
            last_update=self._pool[base_idx + self.LAST_UPDATE_IDX],
            reward_mean=self._pool[base_idx + self.REWARD_MEAN_IDX],
            reward_var=self._pool[base_idx + self.REWARD_VAR_IDX],
        )

    def _set_task_state(self, slot_id: int, task_state: TaskState):
        """Set TaskState in shared memory."""
        base_idx = slot_id * self.FIELDS_PER_TASK
        self._pool[base_idx + self.TASK_ID_IDX] = float(task_state.task_id)
        self._pool[base_idx + self.SCORE_IDX] = task_state.score
        self._pool[base_idx + self.NUM_RUNS_IDX] = float(task_state.num_runs)
        self._pool[base_idx + self.LAST_UPDATE_IDX] = task_state.last_update
        self._pool[base_idx + self.REWARD_MEAN_IDX] = task_state.reward_mean
        self._pool[base_idx + self.REWARD_VAR_IDX] = task_state.reward_var

    def get_shared_memory_names(self) -> str:
        """Get the name for clients to connect to shared memory."""
        return self.name

    def get_stats(self) -> Dict[str, float]:
        """Get curriculum statistics."""
        with self._lock:
            all_tasks = [self._get_task_state(i) for i in range(self.pool_size)]

            scores = [ts.score for ts in all_tasks]
            runs = [ts.num_runs for ts in all_tasks]
            rewards = [ts.reward_mean for ts in all_tasks if ts.num_runs > 0]

            stats = {
                "pool_size": self.pool_size,
                "avg_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "std_score": np.std(scores),
                "avg_runs": np.mean(runs),
                "total_runs": np.sum(runs),
                "tasks_with_runs": sum(1 for r in runs if r > 0),
                "total_completions": int(self._stats[0]),
                "total_replacements": int(self._stats[1]),
            }

            if rewards:
                stats.update(
                    {
                        "avg_reward": np.mean(rewards),
                        "std_reward": np.std(rewards),
                    }
                )

            # Score distribution
            score_hist, score_bins = np.histogram(scores, bins=10, range=(0, 1))
            for i, count in enumerate(score_hist):
                stats[f"score_bin_{i}"] = count

            return stats

    def save_state(self, filepath: str):
        """Save pool state to disk for recovery."""
        with self._lock:
            state = {
                "pool": np.array(self._pool[:]),
                "total_completions": int(self._stats[0]),
                "total_replacements": int(self._stats[1]),
                "config": {
                    "pool_size": self.pool_size,
                    "min_runs": self.min_runs,
                },
            }
            np.savez(filepath, **state)
            logger.info(f"Saved curriculum state to {filepath}")

    def load_state(self, filepath: str):
        """Load pool state from disk."""
        with self._lock:
            state_file = np.load(filepath, allow_pickle=True)
            state = {
                "pool": state_file["pool"],
                "total_completions": state_file["total_completions"].item(),
                "total_replacements": state_file["total_replacements"].item(),
                "config": state_file["config"].item(),
            }

            # Verify configuration matches
            config = state["config"]
            if config["pool_size"] != self.pool_size:
                raise ValueError(f"Pool size mismatch: saved={config['pool_size']}, current={self.pool_size}")

            # Load pool data
            self._pool[:] = state["pool"]
            self._stats[0] = state["total_completions"]
            self._stats[1] = state["total_replacements"]

            logger.info(f"Loaded curriculum state from {filepath}")

    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            self._shm.close()
            self._shm.unlink()
            self._stats_shm.close()
            self._stats_shm.unlink()
            logger.info(f"Cleaned up shared memory for curriculum {self.name}")
        except Exception as e:
            logger.warning(f"Error cleaning up shared memory: {e}")

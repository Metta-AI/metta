"""Task tracking component for curriculum algorithms.

Provides unified TaskTracker implementation with configurable memory backends:
- LocalMemoryBackend: In-memory tracking for single-process use
- SharedMemoryBackend: Shared memory tracking for multi-process use
"""

import time
from typing import Any, Dict, List, Optional

from metta.cogworks.curriculum.shared_memory_backend import LocalMemoryBackend, SharedMemoryBackend, TaskMemoryBackend


class TaskTracker:
    """Unified task tracker using configurable memory backend.

    Works with both local (single-process) and shared (multi-process) memory backends.
    All task tracking logic is unified - no conditional branching based on backend type.
    """

    def __init__(
        self,
        max_memory_tasks: int = 1000,
        ema_alpha: float = 0.1,
        backend: Optional[TaskMemoryBackend] = None,
        session_id: Optional[str] = None,
        use_shared_memory: bool = False,
        task_struct_size: int = 13,
        completion_history_size: int = 1000,
        default_success_threshold: float = 0.5,
        default_generator_type: float = 0.0,
    ):
        """Initialize task tracker with configurable backend.

        Args:
            max_memory_tasks: Maximum number of tasks to track
            ema_alpha: Alpha parameter for exponential moving average
            backend: Optional pre-configured backend. If None, creates based on use_shared_memory
            session_id: Unique identifier for shared memory session (only for shared memory)
            use_shared_memory: If True and backend is None, creates SharedMemoryBackend
            task_struct_size: Size of each task's data structure (default: 13)
            completion_history_size: Size of completion history array (default: 1000)
            default_success_threshold: Default success threshold for new tasks (default: 0.5)
            default_generator_type: Default generator type identifier (default: 0.0)
        """
        self.max_memory_tasks = max_memory_tasks
        self.ema_alpha = ema_alpha
        self.default_success_threshold = default_success_threshold
        self.default_generator_type = default_generator_type

        # Initialize or use provided backend
        if backend is None:
            if use_shared_memory:
                backend = SharedMemoryBackend(
                    max_tasks=max_memory_tasks,
                    session_id=session_id,
                    task_struct_size=task_struct_size,
                    completion_history_size=completion_history_size,
                )
            else:
                backend = LocalMemoryBackend(
                    max_tasks=max_memory_tasks,
                    task_struct_size=task_struct_size,
                    completion_history_size=completion_history_size,
                )

        self._backend: TaskMemoryBackend = backend
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0

        # Rebuild mapping from existing memory
        self._rebuild_task_mapping()

    def _rebuild_task_mapping(self) -> None:
        """Rebuild task ID to array index mapping by scanning backend memory."""
        with self._backend.acquire_lock():
            self._task_id_to_index.clear()
            self._next_free_index = 0

            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                task_id = int(task_data[0])
                is_active = bool(task_data[12])

                if is_active and task_id > 0:
                    self._task_id_to_index[task_id] = i

                if task_id == 0:
                    self._next_free_index = i
                    break
            else:
                self._next_free_index = self._backend.max_tasks

    def track_task_creation(
        self,
        task_id: int,
        success_threshold: Optional[float] = None,
        seed: Optional[float] = None,
        generator_type: Optional[float] = None,
    ) -> None:
        """Track when a task is created with metadata."""
        with self._backend.acquire_lock():
            timestamp = time.time()
            if seed is None:
                seed = hash(str(task_id) + str(timestamp)) % (2**31)
            if success_threshold is None:
                success_threshold = self.default_success_threshold
            if generator_type is None:
                generator_type = self.default_generator_type

            # Check if task already exists
            if task_id in self._task_id_to_index:
                return

            # Find slot in backend memory
            if self._next_free_index >= self._backend.max_tasks:
                return  # No space available

            index = self._next_free_index
            self._task_id_to_index[task_id] = index

            # Write to backend memory
            task_data = self._backend.get_task_data(index)
            task_data[0] = float(task_id)
            task_data[1] = timestamp
            task_data[2] = 0.0  # completion_count
            task_data[3] = 0.0  # reward_ema
            task_data[4] = 0.0  # lp_score
            task_data[5] = 0.0  # success_rate_ema
            task_data[6] = 0.0  # total_score
            task_data[7] = 0.0  # last_score
            task_data[8] = success_threshold
            task_data[9] = float(seed)
            task_data[10] = generator_type
            task_data[11] = 0.0  # ema_squared (for variance calculation)
            task_data[12] = 1.0  # is_active

            self._next_free_index += 1

    def update_task_performance(
        self,
        task_id: int,
        score: float,
        lp_score: Optional[float] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Update task performance with new completion score."""
        # Create task if needed (outside the main lock to avoid deadlock)
        if task_id not in self._task_id_to_index:
            self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)

        with self._backend.acquire_lock():
            # Task should exist now
            if task_id not in self._task_id_to_index:
                # Race condition - another process might have removed it
                return

            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)

            # Read current values
            completion_count = int(task_data[2])
            reward_ema = task_data[3]
            old_lp_score = task_data[4]
            success_rate_ema = task_data[5]
            total_score = task_data[6]
            task_success_threshold = task_data[8]
            ema_squared = task_data[11]

            # Update counts and totals
            new_completion_count = completion_count + 1
            new_total_score = total_score + score

            # Update reward EMA
            if completion_count == 0:
                new_reward_ema = score
            else:
                new_reward_ema = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

            # Update EMA of squared scores (for variance calculation)
            score_squared = score * score
            if completion_count == 0:
                new_ema_squared = score_squared
            else:
                new_ema_squared = (1 - self.ema_alpha) * ema_squared + self.ema_alpha * score_squared

            # Update LP score if provided
            new_lp_score = lp_score if lp_score is not None else old_lp_score

            # Update success rate EMA
            current_threshold = success_threshold if success_threshold is not None else task_success_threshold
            is_success = float(score >= current_threshold)
            if completion_count == 0:
                new_success_rate_ema = is_success
            else:
                new_success_rate_ema = (1 - self.ema_alpha) * success_rate_ema + self.ema_alpha * is_success

            # Write updated values
            task_data[2] = float(new_completion_count)
            task_data[3] = new_reward_ema
            task_data[4] = new_lp_score
            task_data[5] = new_success_rate_ema
            task_data[6] = new_total_score
            task_data[7] = score
            task_data[8] = current_threshold
            task_data[11] = new_ema_squared

            # Add score to completion history
            history = self._backend.get_completion_history()
            for i in range(len(history)):
                if history[i] == 0.0:
                    history[i] = score
                    break
            else:
                # Shift array if full
                history[:-1] = history[1:]
                history[-1] = score

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        with self._backend.acquire_lock():
            if task_id not in self._task_id_to_index:
                return

            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)
            task_data[4] = lp_score

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task.

        Note: No locking - may read slightly stale data, but that's acceptable
        for statistics queries to avoid lock contention.
        """
        if task_id not in self._task_id_to_index:
            return None

        index = self._task_id_to_index[task_id]
        task_data = self._backend.get_task_data(index)

        if task_data[12] == 0:  # not active
            return None

        creation_time = task_data[1]
        completion_count = int(task_data[2])
        reward_ema = task_data[3]
        lp_score = task_data[4]
        success_rate_ema = task_data[5]
        total_score = task_data[6]
        last_score = task_data[7]
        success_threshold = task_data[8]
        seed = task_data[9]
        generator_type = task_data[10]
        ema_squared = task_data[11]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "reward_ema": 0.0,
                "ema_squared": 0.0,
                "lp_score": 0.0,
                "success_rate_ema": 0.0,
                "last_score": 0.0,
                "success_threshold": success_threshold,
                "seed": seed,
                "generator_type": generator_type,
                "age_seconds": time.time() - creation_time,
            }

        return {
            "completion_count": completion_count,
            "mean_score": total_score / completion_count,
            "reward_ema": reward_ema,
            "ema_squared": ema_squared,
            "lp_score": lp_score,
            "success_rate_ema": success_rate_ema,
            "last_score": last_score,
            "success_threshold": success_threshold,
            "seed": seed,
            "generator_type": generator_type,
            "age_seconds": time.time() - creation_time,
        }

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs.

        Note: No locking - returns snapshot which may be slightly stale.
        """
        return list(self._task_id_to_index.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        with self._backend.acquire_lock():
            if task_id in self._task_id_to_index:
                index = self._task_id_to_index[task_id]
                task_data = self._backend.get_task_data(index)
                task_data[12] = 0.0  # is_active = False
                del self._task_id_to_index[task_id]

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics.

        Note: No locking - statistics may be slightly inconsistent but acceptable
        for monitoring purposes. Avoids lock contention on frequent stat queries.
        """
        # Get completion history
        history = self._backend.get_completion_history()
        completion_history = [score for score in history if score != 0.0]

        if not completion_history:
            return {
                "mean_recent_score": 0.0,
            }

        return {
            "mean_recent_score": sum(completion_history) / len(completion_history),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing.

        Note: No locking - checkpoint may have minor inconsistencies if captured
        during updates, but this is acceptable as checkpoints are infrequent.
        """
        task_memory = {}
        for task_id, index in self._task_id_to_index.items():
            task_data = self._backend.get_task_data(index)
            if task_data[12] > 0:  # is_active
                task_memory[task_id] = {
                    "creation_time": task_data[1],
                    "completion_count": int(task_data[2]),
                    "reward_ema": task_data[3],
                    "lp_score": task_data[4],
                    "success_rate_ema": task_data[5],
                    "total_score": task_data[6],
                    "last_score": task_data[7],
                    "success_threshold": task_data[8],
                    "seed": task_data[9],
                    "generator_type": task_data[10],
                    "ema_squared": task_data[11],
                }

        # Get completion history
        history = self._backend.get_completion_history()
        completion_history = [score for score in history if score != 0.0]

        total_completions = sum(int(self._backend.get_task_data(idx)[2]) for idx in self._task_id_to_index.values())

        # Determine tracker type based on backend
        tracker_type = "centralized" if isinstance(self._backend, SharedMemoryBackend) else "local"
        session_id = getattr(self._backend, "session_id", None)

        return {
            "max_memory_tasks": self.max_memory_tasks,
            "tracker_type": tracker_type,
            "session_id": session_id,
            "task_memory": task_memory,
            "completion_history": completion_history,
            "task_creation_order": [],  # Not used with backend approach
            "cached_total_completions": total_completions,
            "cache_valid": True,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        with self._backend.acquire_lock():
            self.max_memory_tasks = state["max_memory_tasks"]

            # Clear backend memory
            self._backend.clear()
            self._task_id_to_index.clear()

            # Restore tasks
            for i, (task_id, task_data) in enumerate(state["task_memory"].items()):
                if i >= self._backend.max_tasks:
                    break

                self._task_id_to_index[int(task_id)] = i
                data = self._backend.get_task_data(i)
                data[0] = float(task_id)
                data[1] = task_data.get("creation_time", time.time())
                data[2] = float(task_data.get("completion_count", 0))
                data[3] = task_data.get("reward_ema", 0.0)
                data[4] = task_data.get("lp_score", 0.0)
                data[5] = task_data.get("success_rate_ema", 0.0)
                data[6] = task_data.get("total_score", 0.0)
                data[7] = task_data.get("last_score", 0.0)
                data[8] = task_data.get("success_threshold", 0.5)
                data[9] = task_data.get("seed", 0.0)
                data[10] = task_data.get("generator_type", 0.0)
                data[11] = task_data.get("ema_squared", 0.0)
                data[12] = 1.0  # is_active

            self._next_free_index = len(state["task_memory"])

            # Restore completion history
            history = self._backend.get_completion_history()
            completion_history = state.get("completion_history", [])
            for i, score in enumerate(completion_history[: len(history)]):
                history[i] = score

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources (only relevant for shared memory backend)."""
        self._backend.cleanup()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        # Only close, don't unlink in destructor
        pass  # Backend handles its own cleanup


# Backwards compatibility factory functions
def LocalTaskTracker(max_memory_tasks: int = 1000, ema_alpha: float = 0.1) -> TaskTracker:
    """Create a local (single-process) task tracker.

    Factory function for backwards compatibility with existing code.

    Args:
        max_memory_tasks: Maximum number of tasks to track
        ema_alpha: Alpha parameter for exponential moving average

    Returns:
        TaskTracker instance with LocalMemoryBackend
    """
    return TaskTracker(max_memory_tasks=max_memory_tasks, ema_alpha=ema_alpha, use_shared_memory=False)


def CentralizedTaskTracker(
    max_memory_tasks: int = 1000,
    session_id: Optional[str] = None,
    ema_alpha: float = 0.1,
    task_struct_size: int = 12,
    completion_history_size: int = 1000,
) -> TaskTracker:
    """Create a centralized (multi-process) task tracker with shared memory.

    Factory function for backwards compatibility with existing code.

    Args:
        max_memory_tasks: Maximum number of tasks to track
        session_id: Unique identifier for shared memory session
        ema_alpha: Alpha parameter for exponential moving average
        task_struct_size: Size of each task's data structure (default: 12)
        completion_history_size: Size of completion history array (default: 1000)

    Returns:
        TaskTracker instance with SharedMemoryBackend
    """
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        session_id=session_id,
        use_shared_memory=True,
        task_struct_size=task_struct_size,
        completion_history_size=completion_history_size,
    )

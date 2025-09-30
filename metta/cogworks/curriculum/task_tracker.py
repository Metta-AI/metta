"""Task tracking component for curriculum algorithms.

Provides abstract interface and two implementations:
- LocalTaskTracker: In-memory tracking for single-process use
- CentralizedTaskTracker: Shared memory tracking for multi-process use
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class TaskTracker(ABC):
    """Abstract base class for task tracking."""

    def __init__(self, max_memory_tasks: int = 1000, ema_alpha: float = 0.1):
        self.max_memory_tasks = max_memory_tasks
        self.ema_alpha = ema_alpha  # Learning rate for EMA updates

    @abstractmethod
    def track_task_creation(
        self, task_id: int, success_threshold: float = 0.5, seed: Optional[float] = None, generator_type: float = 0.0
    ) -> None:
        """Track when a task is created with metadata."""
        pass

    @abstractmethod
    def update_task_performance(
        self, task_id: int, score: float, lp_score: Optional[float] = None, success_threshold: Optional[float] = None
    ) -> None:
        """Update task performance with new completion score."""
        pass

    @abstractmethod
    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        pass

    @abstractmethod
    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task."""
        pass

    @abstractmethod
    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs."""
        pass

    @abstractmethod
    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        pass

    @abstractmethod
    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        pass


class LocalTaskTracker(TaskTracker):
    """In-memory task tracker for single-process use."""

    def __init__(self, max_memory_tasks: int = 1000, ema_alpha: float = 0.1):
        super().__init__(max_memory_tasks, ema_alpha)

        # Task memory: task_id -> (creation_time, completion_count, reward_ema, lp_score,
        #                          success_rate_ema, total_score, last_score, success_threshold,
        #                          seed, generator_type)
        self._task_memory: Dict[int, Tuple[float, int, float, float, float, float, float, float, float, float]] = {}

        # Task creation order for efficient cleanup
        self._task_creation_order = deque()  # (timestamp, task_id) pairs

        # Performance tracking
        self._completion_history = deque(maxlen=1000)

        # Cached values
        self._cached_total_completions = 0
        self._cache_valid = False

    def track_task_creation(
        self,
        task_id: int,
        success_threshold: float = 0.5,
        seed: Optional[float] = None,
        generator_type: float = 0.0,
    ) -> None:
        """Track when a task is created with metadata."""
        timestamp = time.time()
        if seed is None:
            seed = hash(str(task_id) + str(timestamp)) % (2**31)

        self._task_memory[task_id] = (
            timestamp,
            0,  # completion_count
            0.0,  # reward_ema
            0.0,  # lp_score
            0.0,  # success_rate_ema
            0.0,  # total_score
            0.0,  # last_score
            success_threshold,
            float(seed),
            generator_type,
        )
        self._task_creation_order.append((timestamp, task_id))

        # Cleanup old tasks if needed
        if len(self._task_memory) > self.max_memory_tasks:
            self._cleanup_old_tasks()

    def update_task_performance(
        self,
        task_id: int,
        score: float,
        lp_score: Optional[float] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Update task performance with new completion score."""
        # Ensure task exists (create if needed, then continue to process this score)
        if task_id not in self._task_memory:
            self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)

        task_data = self._task_memory[task_id]
        (
            creation_time,
            completion_count,
            reward_ema,
            old_lp_score,
            success_rate_ema,
            total_score,
            _,
            task_success_threshold,
            seed,
            generator_type,
        ) = task_data

        # Update counts and totals
        new_completion_count = completion_count + 1
        new_total_score = total_score + score

        # Update reward EMA
        if completion_count == 0:
            new_reward_ema = score
        else:
            new_reward_ema = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

        # Update LP score if provided
        new_lp_score = lp_score if lp_score is not None else old_lp_score

        # Update success rate EMA
        current_threshold = success_threshold if success_threshold is not None else task_success_threshold
        is_success = float(score >= current_threshold)
        if completion_count == 0:
            new_success_rate_ema = is_success
        else:
            new_success_rate_ema = (1 - self.ema_alpha) * success_rate_ema + self.ema_alpha * is_success

        # Update task data
        self._task_memory[task_id] = (
            creation_time,
            new_completion_count,
            new_reward_ema,
            new_lp_score,
            new_success_rate_ema,
            new_total_score,
            score,
            current_threshold,
            seed,
            generator_type,
        )
        self._completion_history.append(score)

        # Update cached total incrementally
        if self._cache_valid:
            self._cached_total_completions += 1
        else:
            self._cache_valid = False

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        if task_id not in self._task_memory:
            return

        task_data = self._task_memory[task_id]
        (
            creation_time,
            completion_count,
            reward_ema,
            _,
            success_rate_ema,
            total_score,
            last_score,
            success_threshold,
            seed,
            generator_type,
        ) = task_data

        self._task_memory[task_id] = (
            creation_time,
            completion_count,
            reward_ema,
            lp_score,
            success_rate_ema,
            total_score,
            last_score,
            success_threshold,
            seed,
            generator_type,
        )

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task."""
        if task_id not in self._task_memory:
            return None

        (
            creation_time,
            completion_count,
            reward_ema,
            lp_score,
            success_rate_ema,
            total_score,
            last_score,
            success_threshold,
            seed,
            generator_type,
        ) = self._task_memory[task_id]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "reward_ema": 0.0,
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
            "lp_score": lp_score,
            "success_rate_ema": success_rate_ema,
            "last_score": last_score,
            "success_threshold": success_threshold,
            "seed": seed,
            "generator_type": generator_type,
            "age_seconds": time.time() - creation_time,
        }

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs."""
        return list(self._task_memory.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        if task_id in self._task_memory:
            if self._cache_valid:
                task_data = self._task_memory[task_id]
                completion_count = task_data[1]
                self._cached_total_completions -= completion_count
            self._task_memory.pop(task_id, None)
            self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics."""
        if not self._completion_history:
            return {
                "mean_recent_score": 0.0,
                "total_tracked_tasks": 0,
                "total_completions": 0,
            }

        if not self._cache_valid:
            self._cached_total_completions = sum(
                completion_count for (_, completion_count, *_) in self._task_memory.values()
            )
            self._cache_valid = True

        return {
            "mean_recent_score": sum(self._completion_history) / len(self._completion_history),
            "total_tracked_tasks": len(self._task_memory),
            "total_completions": self._cached_total_completions,
        }

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks when memory limit is exceeded."""
        cleanup_count = min(100, len(self._task_memory) - self.max_memory_tasks + 100)

        removed_count = 0
        while self._task_creation_order and removed_count < cleanup_count:
            _, task_id = self._task_creation_order.popleft()
            if task_id in self._task_memory:
                if self._cache_valid:
                    task_data = self._task_memory[task_id]
                    completion_count = task_data[1]
                    self._cached_total_completions -= completion_count
                del self._task_memory[task_id]
                removed_count += 1

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing."""
        return {
            "max_memory_tasks": self.max_memory_tasks,
            "tracker_type": "local",
            "task_memory": {
                task_id: {
                    "creation_time": creation_time,
                    "completion_count": completion_count,
                    "reward_ema": reward_ema,
                    "lp_score": lp_score,
                    "success_rate_ema": success_rate_ema,
                    "total_score": total_score,
                    "last_score": last_score,
                    "success_threshold": success_threshold,
                    "seed": seed,
                    "generator_type": generator_type,
                }
                for task_id, (
                    creation_time,
                    completion_count,
                    reward_ema,
                    lp_score,
                    success_rate_ema,
                    total_score,
                    last_score,
                    success_threshold,
                    seed,
                    generator_type,
                ) in self._task_memory.items()
            },
            "task_creation_order": list(self._task_creation_order),
            "completion_history": list(self._completion_history),
            "cached_total_completions": self._cached_total_completions,
            "cache_valid": self._cache_valid,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        self.max_memory_tasks = state["max_memory_tasks"]

        self._task_memory.clear()
        for task_id, task_data in state["task_memory"].items():
            self._task_memory[int(task_id)] = (
                task_data.get("creation_time", time.time()),
                task_data.get("completion_count", 0),
                task_data.get("reward_ema", 0.0),
                task_data.get("lp_score", 0.0),
                task_data.get("success_rate_ema", 0.0),
                task_data.get("total_score", 0.0),
                task_data.get("last_score", 0.0),
                task_data.get("success_threshold", 0.5),
                task_data.get("seed", 0.0),
                task_data.get("generator_type", 0.0),
            )

        self._task_creation_order = deque(state.get("task_creation_order", []))
        self._completion_history = deque(state.get("completion_history", []), maxlen=1000)
        self._cached_total_completions = state.get("cached_total_completions", 0)
        self._cache_valid = state.get("cache_valid", False)


class CentralizedTaskTracker(TaskTracker):
    """Shared memory task tracker for multi-process use.

    Uses SharedMemoryBackend for cross-process communication.
    """

    def __init__(
        self,
        max_memory_tasks: int = 1000,
        session_id: Optional[str] = None,
        ema_alpha: float = 0.1,
        task_struct_size: int = 12,
        completion_history_size: int = 1000,
    ):
        """Initialize centralized task tracker with shared memory.

        Args:
            max_memory_tasks: Maximum number of tasks to track
            session_id: Unique identifier for shared memory session
            ema_alpha: Alpha parameter for exponential moving average
            task_struct_size: Size of each task's data structure (default: 12)
                - Configurable to allow different learning progress algorithms
                - Current structure: [task_id, creation_time, completion_count,
                  reward_ema, lp_score, success_rate_ema, total_score, last_score,
                  success_threshold, seed, generator_type, is_active]
            completion_history_size: Size of completion history array (default: 1000)
        """
        super().__init__(max_memory_tasks, ema_alpha)

        from metta.cogworks.curriculum.shared_memory_backend import SharedMemoryBackend

        # Use max_memory_tasks for shared memory size to ensure consistency
        self._backend = SharedMemoryBackend(
            max_tasks=max_memory_tasks,
            session_id=session_id,
            task_struct_size=task_struct_size,
            completion_history_size=completion_history_size,
        )
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0

        # Rebuild mapping from existing shared memory
        self._rebuild_task_mapping()

    def _rebuild_task_mapping(self):
        """Rebuild task ID to array index mapping by scanning shared memory."""
        with self._backend.acquire_lock():
            self._task_id_to_index.clear()
            self._next_free_index = 0

            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                task_id = int(task_data[0])
                is_active = bool(task_data[11])

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
        success_threshold: float = 0.5,
        seed: Optional[float] = None,
        generator_type: float = 0.0,
    ) -> None:
        """Track when a task is created with metadata."""
        with self._backend.acquire_lock():
            timestamp = time.time()
            if seed is None:
                seed = hash(str(task_id) + str(timestamp)) % (2**31)

            # Check if task already exists
            if task_id in self._task_id_to_index:
                return

            # Find slot in shared memory
            if self._next_free_index >= self._backend.max_tasks:
                return  # No space available

            index = self._next_free_index
            self._task_id_to_index[task_id] = index

            # Write to shared memory
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
            task_data[11] = 1.0  # is_active

            self._next_free_index += 1

    def update_task_performance(
        self,
        task_id: int,
        score: float,
        lp_score: Optional[float] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Update task performance with new completion score."""
        with self._backend.acquire_lock():
            # Ensure task exists (create if needed, then continue to process this score)
            if task_id not in self._task_id_to_index:
                # Release lock, create task, then reacquire
                pass  # Will create task in track_task_creation which has its own lock

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

            # Update counts and totals
            new_completion_count = completion_count + 1
            new_total_score = total_score + score

            # Update reward EMA
            if completion_count == 0:
                new_reward_ema = score
            else:
                new_reward_ema = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

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

        if task_data[11] == 0:  # not active
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

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "reward_ema": 0.0,
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
                task_data[11] = 0.0  # is_active = False
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
                "total_tracked_tasks": 0,
                "total_completions": 0,
            }

        # Calculate total completions
        total_completions = 0
        for index in self._task_id_to_index.values():
            task_data = self._backend.get_task_data(index)
            if task_data[11] > 0:  # is_active
                total_completions += int(task_data[2])

        return {
            "mean_recent_score": sum(completion_history) / len(completion_history),
            "total_tracked_tasks": len(self._task_id_to_index),
            "total_completions": total_completions,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing.

        Note: No locking - checkpoint may have minor inconsistencies if captured
        during updates, but this is acceptable as checkpoints are infrequent.
        """
        task_memory = {}
        for task_id, index in self._task_id_to_index.items():
            task_data = self._backend.get_task_data(index)
            if task_data[11] > 0:  # is_active
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
                }

        # Get completion history
        history = self._backend.get_completion_history()
        completion_history = [score for score in history if score != 0.0]

        total_completions = sum(int(self._backend.get_task_data(idx)[2]) for idx in self._task_id_to_index.values())

        return {
            "max_memory_tasks": self.max_memory_tasks,
            "tracker_type": "centralized",
            "session_id": self._backend.session_id,
            "task_memory": task_memory,
            "completion_history": completion_history,
            "task_creation_order": [],  # Not used in centralized mode
            "cached_total_completions": total_completions,
            "cache_valid": True,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        with self._backend.acquire_lock():
            self.max_memory_tasks = state["max_memory_tasks"]

            # Clear shared memory
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
                data[11] = 1.0  # is_active

            self._next_free_index = len(state["task_memory"])

            # Restore completion history
            history = self._backend.get_completion_history()
            completion_history = state.get("completion_history", [])
            for i, score in enumerate(completion_history[: len(history)]):
                history[i] = score

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources."""
        if hasattr(self, "_backend"):
            self._backend.cleanup()

    def __del__(self):
        """Cleanup on destruction."""
        # Only close, don't unlink in destructor
        pass  # SharedTaskMemory handles its own cleanup

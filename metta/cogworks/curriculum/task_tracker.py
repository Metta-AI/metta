"""
Task tracking component for curriculum algorithms.

Handles task memory, performance history, and basic task metadata
without mixing in learning progress calculations or bucket analysis.
"""

import time
from collections import deque
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple


class TaskTracker:
    """Tracks task metadata, performance history, and completion statistics."""

    def __init__(self, max_memory_tasks: int = 1000, use_shared_memory: bool = True):
        self.max_memory_tasks = max_memory_tasks
        self.use_shared_memory = use_shared_memory

        if use_shared_memory:
            # Use multiprocessing.Lock for consistency with other components
            from multiprocessing import Lock

            self._lock = Lock()
            # Shared memory for cross-process task tracking
            self._init_shared_memory()
        else:
            # Standard in-process memory - use a simple object for compatibility
            class DummyLock:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False

            self._lock = DummyLock()
            self._init_local_memory()

    def _init_shared_memory(self):
        """Initialize shared memory structures for cross-thread access."""
        # Use shared memory arrays for direct random access
        # Define the structure: each task gets a fixed slot with:
        # [task_id, creation_time, completion_count, reward_ema, lp_score, success_rate_ema,
        #  total_score, last_score, success_threshold, seed, generator_type, is_active]
        self.max_shared_tasks = 10000  # Maximum tasks we can track in shared memory
        self.task_struct_size = 12  # 12 floats per task

        # EMA decay factor for reward and success rate tracking
        self.ema_alpha = 0.1  # Learning rate for EMA updates

        # Initialize creation tracking
        self._created_shared_memory = False

        # Create unique names for shared memory to avoid conflicts in parallel tests
        import os

        pid = os.getpid()
        self._task_array_name = f"task_tracker_array_{pid}"
        self._completion_history_name = f"completion_history_array_{pid}"

        try:
            # Try to connect to existing shared memory
            self._task_array_shm = shared_memory.SharedMemory(name=self._task_array_name, create=False)
            self._completion_history_shm = shared_memory.SharedMemory(name=self._completion_history_name, create=False)

            # Check if existing buffer is large enough for new structure
            expected_size = self.max_shared_tasks * self.task_struct_size * 8  # 8 bytes per float64
            if self._task_array_shm.size < expected_size:
                # Buffer too small, recreate with new size
                self._task_array_shm.close()
                self._task_array_shm.unlink()
                self._completion_history_shm.close()
                self._completion_history_shm.unlink()
                raise FileNotFoundError("Recreating shared memory with new size")

        except FileNotFoundError:
            # Create new shared memory arrays
            self._create_shared_memory_arrays()

        # Create numpy-like views of the shared memory
        import numpy as np

        self._task_array = np.ndarray(
            (self.max_shared_tasks, self.task_struct_size), dtype=np.float64, buffer=self._task_array_shm.buf
        )
        self._completion_history_array = np.ndarray((1000,), dtype=np.float64, buffer=self._completion_history_shm.buf)

        # Task ID to array index mapping for fast lookup
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0

        # Initialize the mapping by scanning existing tasks
        self._rebuild_task_mapping()

    def _rebuild_task_mapping(self):
        """Rebuild the task ID to array index mapping by scanning shared memory."""
        if not self.use_shared_memory:
            return

        self._task_id_to_index.clear()
        self._next_free_index = 0

        for i in range(self.max_shared_tasks):
            task_id = int(self._task_array[i, 0])
            is_active = bool(self._task_array[i, 11])  # is_active at index 11

            if is_active and task_id > 0:
                self._task_id_to_index[task_id] = i

            if task_id == 0:  # Found first empty slot
                self._next_free_index = i
                break
        else:
            # No empty slots found
            self._next_free_index = self.max_shared_tasks

    def _init_local_memory(self):
        """Initialize standard local memory structures."""
        # For local memory mode, keep the dictionary-based approach
        # Task memory: task_id -> (creation_time, completion_count, reward_ema, lp_score, success_rate_ema,
        #                          total_score, last_score, success_threshold, seed, generator_type)
        self._task_memory: Dict[int, Tuple[float, int, float, float, float, float, float, float, float, float]] = {}

        # EMA decay factor for reward and success rate tracking
        self.ema_alpha = 0.1  # Learning rate for EMA updates

        # Task creation order for efficient cleanup
        self._task_creation_order = deque()  # (timestamp, task_id) pairs

        # Performance tracking
        self._completion_history = deque(maxlen=1000)  # Recent completion scores

        # Cached values to avoid expensive recomputation
        self._cached_total_completions = 0
        self._cache_valid = False

    def _create_shared_memory_arrays(self):
        """Create shared memory arrays for task tracking."""
        import numpy as np

        # Calculate sizes
        task_array_size = self.max_shared_tasks * self.task_struct_size * 8  # 8 bytes per float64
        completion_history_size = 1000 * 8  # 8 bytes per float64

        # Track if we created the shared memory
        self._created_shared_memory = False

        try:
            self._task_array_shm = shared_memory.SharedMemory(
                name=self._task_array_name, create=True, size=task_array_size
            )
            self._completion_history_shm = shared_memory.SharedMemory(
                name=self._completion_history_name, create=True, size=completion_history_size
            )
            # We successfully created the shared memory
            self._created_shared_memory = True
        except FileExistsError:
            # Another process created it, connect to existing
            self._task_array_shm = shared_memory.SharedMemory(name=self._task_array_name, create=False)
            self._completion_history_shm = shared_memory.SharedMemory(name=self._completion_history_name, create=False)
            self._created_shared_memory = False
            return

        # Initialize arrays to zero only if we created them
        if self._created_shared_memory:
            task_array = np.ndarray(
                (self.max_shared_tasks, self.task_struct_size), dtype=np.float64, buffer=self._task_array_shm.buf
            )
            completion_array = np.ndarray((1000,), dtype=np.float64, buffer=self._completion_history_shm.buf)

            task_array.fill(0.0)
            completion_array.fill(0.0)

    def track_task_creation(
        self,
        task_id: int,
        success_threshold: float = 0.5,
        seed: Optional[float] = None,
        generator_type: float = 0.0,  # Can encode different generator types as numbers
    ) -> None:
        """Track when a task is created with metadata for regeneration."""
        with self._lock:
            timestamp = time.time()
            if seed is None:
                seed = hash(str(task_id) + str(timestamp)) % (2**31)  # Generate reproducible seed

            if self.use_shared_memory:
                # Check if task already exists
                if task_id in self._task_id_to_index:
                    return  # Task already exists

                # Find or create a slot in shared memory
                if self._next_free_index >= self.max_shared_tasks:
                    # No space available - could implement cleanup here
                    return

                index = self._next_free_index
                self._task_id_to_index[task_id] = index

                # Write directly to shared memory
                self._task_array[index, 0] = float(task_id)
                self._task_array[index, 1] = timestamp
                self._task_array[index, 2] = 0.0  # completion_count
                self._task_array[index, 3] = 0.0  # reward_ema
                self._task_array[index, 4] = 0.0  # lp_score
                self._task_array[index, 5] = 0.0  # success_rate_ema
                self._task_array[index, 6] = 0.0  # total_score
                self._task_array[index, 7] = 0.0  # last_score
                self._task_array[index, 8] = success_threshold
                self._task_array[index, 9] = float(seed)
                self._task_array[index, 10] = generator_type
                self._task_array[index, 11] = 1.0  # is_active

                # Update next free index
                self._next_free_index += 1

            else:
                # Use local memory
                self._task_memory[task_id] = (
                    timestamp,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    success_threshold,
                    float(seed),
                    generator_type,
                )
                self._task_creation_order.append((timestamp, task_id))

                # Cleanup old tasks if we exceed memory limit
                if len(self._task_memory) > self.max_memory_tasks:
                    self._cleanup_old_tasks()

    def update_task_performance(
        self, task_id: int, score: float, lp_score: Optional[float] = None, success_threshold: Optional[float] = None
    ) -> None:
        """Update task performance with new completion score and EMA calculations."""
        with self._lock:
            if self.use_shared_memory:
                # Ensure task exists in shared memory
                if task_id not in self._task_id_to_index:
                    self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)
                    return

                index = self._task_id_to_index[task_id]

                # Read current values from shared memory
                completion_count = int(self._task_array[index, 2])
                reward_ema = self._task_array[index, 3]
                old_lp_score = self._task_array[index, 4]
                success_rate_ema = self._task_array[index, 5]
                total_score = self._task_array[index, 6]
                task_success_threshold = self._task_array[index, 8]

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

                # Write updated values directly to shared memory
                self._task_array[index, 2] = float(new_completion_count)
                self._task_array[index, 3] = new_reward_ema
                self._task_array[index, 4] = new_lp_score
                self._task_array[index, 5] = new_success_rate_ema
                self._task_array[index, 6] = new_total_score
                self._task_array[index, 7] = score  # last_score
                self._task_array[index, 8] = current_threshold  # success_threshold

                # Add score to completion history (find next empty slot)
                for i in range(1000):
                    if self._completion_history_array[i] == 0.0:
                        self._completion_history_array[i] = score
                        break
                else:
                    # If no empty slot, shift array and add at end (simple FIFO)
                    self._completion_history_array[:-1] = self._completion_history_array[1:]
                    self._completion_history_array[-1] = score
            else:
                # Use local memory mode
                # Ensure task exists in memory with atomic operation
                if task_id not in self._task_memory:
                    self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)
                    return

                # Use get() with default to handle race conditions in multiprocessing
                task_data = self._task_memory.get(task_id)
                if task_data is None:
                    # Task was removed between check and access - recreate it
                    self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)
                    return

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

                # Update cached total completions incrementally
                if hasattr(self, "_cache_valid") and self._cache_valid:
                    self._cached_total_completions += 1
                else:
                    if hasattr(self, "_cache_valid"):
                        self._cache_valid = False

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        with self._lock:
            if self.use_shared_memory:
                if task_id not in self._task_id_to_index:
                    return  # Task doesn't exist

                index = self._task_id_to_index[task_id]
                # Update LP score directly in shared memory
                self._task_array[index, 4] = lp_score
            else:
                if task_id not in self._task_memory:
                    return  # Task doesn't exist

                task_data = self._task_memory.get(task_id)
                if task_data is None:
                    return  # Task was removed

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

                # Update with new LP score
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
        with self._lock:
            if self.use_shared_memory:
                # Check shared memory
                if task_id not in self._task_id_to_index:
                    return None

                index = self._task_id_to_index[task_id]
                task_data = self._task_array[index]

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

            else:
                # Check local memory
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

    def get_all_tracked_tasks(self) -> list[int]:
        """Get all currently tracked task IDs."""
        with self._lock:
            if self.use_shared_memory:
                return list(self._task_id_to_index.keys())
            else:
                return list(self._task_memory.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        with self._lock:
            if self.use_shared_memory:
                if task_id in self._task_id_to_index:
                    index = self._task_id_to_index[task_id]
                    # Mark as inactive in shared memory
                    self._task_array[index, 11] = 0.0  # is_active = False
                    # Remove from mapping
                    del self._task_id_to_index[task_id]
            else:
                if task_id in self._task_memory:
                    # Update cached total before removal
                    if hasattr(self, "_cache_valid") and self._cache_valid:
                        task_data = self._task_memory[task_id]
                        completion_count = task_data[1]  # completion_count is at index 1
                        self._cached_total_completions -= completion_count

                    self._task_memory.pop(task_id, None)
                    # Note: We don't remove from creation_order for performance - cleanup handles this

                    # Invalidate cache if removal makes it invalid
                    if hasattr(self, "_cache_valid") and self._cache_valid:
                        self._cache_valid = False

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics."""
        with self._lock:
            if self.use_shared_memory:
                # Get completion history from shared memory
                completion_history = []
                for i in range(1000):
                    score = self._completion_history_array[i]
                    if score != 0.0:
                        completion_history.append(score)

                if not completion_history:
                    return {
                        "mean_recent_score": 0.0,
                        "total_tracked_tasks": 0,
                        "total_completions": 0,
                    }

                # Calculate total completions from shared memory
                total_completions = 0
                for index in self._task_id_to_index.values():
                    if self._task_array[index, 11] > 0:  # is_active
                        total_completions += int(self._task_array[index, 2])  # completion_count

                return {
                    "mean_recent_score": sum(completion_history) / len(completion_history),
                    "total_tracked_tasks": len(self._task_id_to_index),
                    "total_completions": total_completions,
                }
            else:
                # Use local memory
                if not self._completion_history:
                    return {
                        "mean_recent_score": 0.0,
                        "total_tracked_tasks": 0,
                        "total_completions": 0,
                    }

                # Use cached total completions if valid, otherwise compute
                if not hasattr(self, "_cache_valid") or not self._cache_valid:
                    self._cached_total_completions = sum(
                        completion_count for (_, completion_count, _, _, _, _, _, _, _, _) in self._task_memory.values()
                    )
                    self._cache_valid = True

                return {
                    "mean_recent_score": sum(self._completion_history) / len(self._completion_history),
                    "total_tracked_tasks": len(self._task_memory),
                    "total_completions": self._cached_total_completions,
                }

    def _cleanup_old_tasks(self) -> None:
        """Remove oldest tasks when memory limit is exceeded."""
        # Note: This is called from within a lock, so no additional locking needed
        cleanup_count = min(100, len(self._task_memory) - self.max_memory_tasks + 100)

        # Remove oldest tasks
        removed_count = 0
        while self._task_creation_order and removed_count < cleanup_count:
            _, task_id = self._task_creation_order.popleft()
            if task_id in self._task_memory:
                # Update cached total before removal
                if self._cache_valid:
                    task_data = self._task_memory[task_id]
                    completion_count = task_data[1]  # completion_count is at index 1
                    self._cached_total_completions -= completion_count

                del self._task_memory[task_id]
                removed_count += 1

        # Cache may still be valid after cleanup if we tracked changes

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing."""
        with self._lock:
            if self.use_shared_memory:
                # Collect data from shared memory
                task_memory = {}
                for task_id, index in self._task_id_to_index.items():
                    task_data = self._task_array[index]
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

                # Get completion history from shared memory
                completion_history = []
                for i in range(1000):
                    score = self._completion_history_array[i]
                    if score != 0.0:
                        completion_history.append(score)

                return {
                    "max_memory_tasks": self.max_memory_tasks,
                    "use_shared_memory": self.use_shared_memory,
                    "task_memory": task_memory,
                    "completion_history": completion_history,
                    "task_creation_order": [],  # Empty for shared memory mode (not needed)
                    "cached_total_completions": sum(
                        int(self._task_array[idx, 2]) for idx in self._task_id_to_index.values()
                    ),
                    "cache_valid": True,  # Always valid in shared memory mode
                }
            else:
                # Use local memory
                return {
                    "max_memory_tasks": self.max_memory_tasks,
                    "use_shared_memory": self.use_shared_memory,
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
        with self._lock:
            self.max_memory_tasks = state["max_memory_tasks"]
            self.use_shared_memory = state.get("use_shared_memory", self.use_shared_memory)

            if self.use_shared_memory:
                # Clear shared memory and rebuild
                self._task_array.fill(0.0)
                self._completion_history_array.fill(0.0)
                self._task_id_to_index.clear()

                # Restore tasks to shared memory
                for i, (task_id, task_data) in enumerate(state["task_memory"].items()):
                    if i >= self.max_shared_tasks:
                        break

                    self._task_id_to_index[int(task_id)] = i
                    self._task_array[i, 0] = float(task_id)
                    self._task_array[i, 1] = task_data.get("creation_time", time.time())
                    self._task_array[i, 2] = float(task_data.get("completion_count", 0))
                    self._task_array[i, 3] = task_data.get("reward_ema", 0.0)
                    self._task_array[i, 4] = task_data.get("lp_score", 0.0)
                    self._task_array[i, 5] = task_data.get("success_rate_ema", 0.0)
                    self._task_array[i, 6] = task_data.get("total_score", 0.0)
                    self._task_array[i, 7] = task_data.get("last_score", 0.0)
                    self._task_array[i, 8] = task_data.get("success_threshold", 0.5)
                    self._task_array[i, 9] = task_data.get("seed", 0.0)
                    self._task_array[i, 10] = task_data.get("generator_type", 0.0)
                    self._task_array[i, 11] = 1.0  # is_active

                self._next_free_index = len(state["task_memory"])

                # Restore completion history
                completion_history = state.get("completion_history", [])
                for i, score in enumerate(completion_history[:1000]):
                    self._completion_history_array[i] = score

            else:
                # Restore local memory
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

                # Restore creation order and other local state
                self._task_creation_order = deque(state.get("task_creation_order", []))
                self._completion_history = deque(state.get("completion_history", []), maxlen=1000)
                self._cached_total_completions = state.get("cached_total_completions", 0)
                self._cache_valid = state.get("cache_valid", False)

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources."""
        if self.use_shared_memory:
            try:
                if hasattr(self, "_task_array_shm"):
                    self._task_array_shm.close()
                    # Only unlink if we're the original creator
                    if getattr(self, "_created_shared_memory", False):
                        try:
                            self._task_array_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already unlinked
                if hasattr(self, "_completion_history_shm"):
                    self._completion_history_shm.close()
                    # Only unlink if we're the original creator
                    if getattr(self, "_created_shared_memory", False):
                        try:
                            self._completion_history_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already unlinked
            except Exception:
                pass  # Ignore cleanup errors

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "use_shared_memory") and self.use_shared_memory:
            try:
                # Only close, don't unlink in destructor to avoid race conditions
                if hasattr(self, "_task_array_shm"):
                    self._task_array_shm.close()
                if hasattr(self, "_completion_history_shm"):
                    self._completion_history_shm.close()
            except Exception:
                pass

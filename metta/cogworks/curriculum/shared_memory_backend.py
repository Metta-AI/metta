"""Low-level shared memory backend for curriculum task tracking.

Provides a clean separation between shared memory management and task tracking logic.
"""

from multiprocessing import shared_memory
from typing import Optional

import numpy as np


class SharedTaskMemory:
    """Pure shared memory data structure for task tracking.

    Handles low-level shared memory operations without mixing in logic.
    Designed for cross-process access with proper synchronization.
    """

    # Task structure: [task_id, creation_time, completion_count, reward_ema, lp_score,
    #                  success_rate_ema, total_score, last_score, success_threshold,
    #                  seed, generator_type, is_active]
    TASK_STRUCT_SIZE = 12
    COMPLETION_HISTORY_SIZE = 1000

    def __init__(self, max_tasks: int = 10000, session_id: Optional[str] = None):
        """Initialize shared memory backend.

        Args:
            max_tasks: Maximum number of tasks to track in shared memory
            session_id: Unique identifier for this shared memory session.
                       All processes sharing state must use the same session_id.
                       If None, creates a unique session (not shared).
        """
        self.max_tasks = max_tasks

        # Generate session ID if not provided
        if session_id is None:
            import uuid

            session_id = f"cur_{uuid.uuid4().hex[:6]}"
        self.session_id = session_id

        # Create shared memory names based on session
        # POSIX shared memory names have strict length limits (31 chars on macOS)
        # Use short prefixes to stay within limits
        self._task_array_name = f"ta_{session_id}"  # task array
        self._completion_history_name = f"ch_{session_id}"  # completion history
        self._lock_name = f"lk_{session_id}"  # lock

        # Track if we created the shared memory (for cleanup)
        self._created_shared_memory = False

        # Initialize shared structures
        self._init_shared_memory()

    def _init_shared_memory(self):
        """Initialize shared memory structures."""
        # Calculate sizes
        task_array_size = self.max_tasks * self.TASK_STRUCT_SIZE * 8  # 8 bytes per float64
        completion_history_size = self.COMPLETION_HISTORY_SIZE * 8

        try:
            # Try to connect to existing shared memory
            self._task_array_shm = shared_memory.SharedMemory(name=self._task_array_name)
            self._completion_history_shm = shared_memory.SharedMemory(name=self._completion_history_name)
            self._created_shared_memory = False

            # Verify size is sufficient (OS may round up to page boundaries)
            if self._task_array_shm.size < task_array_size:
                # Size too small - this is an error
                raise ValueError(
                    f"Existing shared memory is too small. "
                    f"Need {task_array_size} bytes, found {self._task_array_shm.size} bytes. "
                    f"Ensure all processes use the same max_memory_tasks value."
                )

        except FileNotFoundError:
            # Create new shared memory
            self._task_array_shm = shared_memory.SharedMemory(
                name=self._task_array_name, create=True, size=task_array_size
            )
            self._completion_history_shm = shared_memory.SharedMemory(
                name=self._completion_history_name, create=True, size=completion_history_size
            )
            self._created_shared_memory = True

            # Initialize to zero only if we created it
            task_array = np.ndarray(
                (self.max_tasks, self.TASK_STRUCT_SIZE), dtype=np.float64, buffer=self._task_array_shm.buf
            )
            completion_array = np.ndarray(
                (self.COMPLETION_HISTORY_SIZE,), dtype=np.float64, buffer=self._completion_history_shm.buf
            )
            task_array.fill(0.0)
            completion_array.fill(0.0)

        # Create numpy views
        self._task_array = np.ndarray(
            (self.max_tasks, self.TASK_STRUCT_SIZE), dtype=np.float64, buffer=self._task_array_shm.buf
        )
        self._completion_history = np.ndarray(
            (self.COMPLETION_HISTORY_SIZE,), dtype=np.float64, buffer=self._completion_history_shm.buf
        )

        # Use multiprocessing RLock for synchronization
        # This works with both fork and spawn multiprocessing contexts
        from multiprocessing import RLock

        self._lock = RLock()

    def acquire_lock(self):
        """Acquire the shared lock."""
        return self._lock

    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view)."""
        return self._task_array[index]

    def set_task_data(self, index: int, data: np.ndarray):
        """Set task data at given index."""
        self._task_array[index] = data

    def get_completion_history(self) -> np.ndarray:
        """Get completion history array (raw view)."""
        return self._completion_history

    def clear(self):
        """Clear all shared memory data."""
        with self._lock:
            self._task_array.fill(0.0)
            self._completion_history.fill(0.0)

    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            if hasattr(self, "_task_array_shm"):
                self._task_array_shm.close()
                if self._created_shared_memory:
                    try:
                        self._task_array_shm.unlink()
                    except FileNotFoundError:
                        pass

            if hasattr(self, "_completion_history_shm"):
                self._completion_history_shm.close()
                if self._created_shared_memory:
                    try:
                        self._completion_history_shm.unlink()
                    except FileNotFoundError:
                        pass
        except Exception:
            pass  # Best-effort cleanup

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, "_task_array_shm"):
                self._task_array_shm.close()
            if hasattr(self, "_completion_history_shm"):
                self._completion_history_shm.close()
        except Exception:
            pass

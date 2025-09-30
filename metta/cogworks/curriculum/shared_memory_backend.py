"""Memory backends for curriculum task tracking.

Provides unified interface for task data storage with both local and shared memory implementations.
The abstraction allows TaskTracker to work identically whether using in-process or multi-process storage.
"""

from abc import ABC, abstractmethod
from multiprocessing import RLock, shared_memory
from typing import Optional

import numpy as np


class TaskMemoryBackend(ABC):
    """Abstract interface for task memory storage.

    This interface hides the implementation details of local vs shared memory,
    allowing the rest of the curriculum system to work identically regardless
    of whether running single-process or multi-process training.
    """

    # Task structure: [task_id, creation_time, completion_count, reward_ema, lp_score,
    #                  success_rate_ema, total_score, last_score, success_threshold,
    #                  seed, generator_type, is_active]
    TASK_STRUCT_SIZE = 12
    COMPLETION_HISTORY_SIZE = 1000

    @abstractmethod
    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view).

        Returns numpy array of length TASK_STRUCT_SIZE that can be read/written.
        """
        pass

    @abstractmethod
    def get_completion_history(self) -> np.ndarray:
        """Get completion history array (raw view).

        Returns numpy array of length COMPLETION_HISTORY_SIZE that can be read/written.
        """
        pass

    @abstractmethod
    def acquire_lock(self):
        """Acquire lock for thread-safe access.

        Returns context manager for use with 'with' statement.
        For local backend, this may be a no-op lock.
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all memory data."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (close files, free memory, etc.)."""
        pass


class LocalMemoryBackend(TaskMemoryBackend):
    """In-memory backend for single-process use.

    Uses simple numpy arrays stored in local process memory.
    No inter-process communication overhead.
    """

    def __init__(self, max_tasks: int = 10000):
        """Initialize local memory backend.

        Args:
            max_tasks: Maximum number of tasks to track
        """
        self.max_tasks = max_tasks

        # Allocate local numpy arrays
        self._task_array = np.zeros((max_tasks, self.TASK_STRUCT_SIZE), dtype=np.float64)
        self._completion_history = np.zeros((self.COMPLETION_HISTORY_SIZE,), dtype=np.float64)

        # No-op lock for local memory (single process)
        from threading import RLock

        self._lock = RLock()

    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view)."""
        return self._task_array[index]

    def get_completion_history(self) -> np.ndarray:
        """Get completion history array (raw view)."""
        return self._completion_history

    def acquire_lock(self):
        """Acquire lock (no-op for local memory)."""
        return self._lock

    def clear(self):
        """Clear all memory data."""
        self._task_array.fill(0.0)
        self._completion_history.fill(0.0)

    def cleanup(self):
        """Clean up resources (no-op for local memory)."""
        pass

    def __del__(self):
        """Cleanup on destruction (no-op for local memory)."""
        pass


class SharedMemoryBackend(TaskMemoryBackend):
    """Shared memory backend for multi-process use.

    Uses multiprocessing.shared_memory for cross-process data sharing.
    Multiple processes can read/write the same task data concurrently.
    """

    def __init__(self, max_tasks: int = 10000, session_id: Optional[str] = None):
        """Initialize shared memory backend.

        Args:
            max_tasks: Maximum number of tasks to track in shared memory
            session_id: Unique identifier for this shared memory session.
                       All processes sharing state must use the same session_id.
                       If None, creates a unique session (not shared across processes).
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
        self._lock = RLock()

    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view)."""
        return self._task_array[index]

    def get_completion_history(self) -> np.ndarray:
        """Get completion history array (raw view)."""
        return self._completion_history

    def acquire_lock(self):
        """Acquire the shared lock."""
        return self._lock

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


# Backwards compatibility alias
SharedTaskMemory = SharedMemoryBackend

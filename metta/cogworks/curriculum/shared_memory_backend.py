"""Memory storage backends for task performance data.

This module defines the TaskMemoryBackend interface and provides two implementations:
local memory (numpy arrays) and shared memory (multiprocessing.shared_memory). The unified
interface allows TaskTracker to work identically regardless of whether training is
single-process or multi-process.

Key components:
- TaskState: Pydantic model defining the task data structure
- TaskMemoryBackend: Abstract interface defining storage operations
- LocalMemoryBackend: Fast numpy array storage for single-process use
- SharedMemoryBackend: Cross-process shared memory with multiprocessing.Lock for synchronization

Synchronization: SharedMemoryBackend uses multiprocessing.Manager().Lock() to ensure atomic
multi-field updates across processes. Manager.Lock() returns a proxy object that can be
properly pickled and shared, ensuring all processes synchronize on the same server-backed
lock. This prevents race conditions when updating related values like the 4 bidirectional
EMAs (p_fast, p_slow, p_true, random_baseline).

Why separate file: Memory management is a low-level concern distinct from curriculum logic.
Isolating it here makes it easy to swap implementations, test independently, and reason
about thread safety and resource cleanup without cluttering higher-level code.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import nullcontext
from multiprocessing import Manager, shared_memory
from typing import Any, ContextManager, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TaskState(BaseModel):
    """Task performance data structure.

    This model defines all fields stored for each task. The struct_size is computed
    dynamically from the number of float fields, eliminating magic offsets.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Core task metadata
    task_id: float = Field(default=0.0, description="Unique task identifier")
    creation_time: float = Field(default=0.0, description="Timestamp when task was created")
    is_active: float = Field(default=0.0, description="Whether task is active (1.0) or evicted (0.0)")

    # Performance tracking
    completion_count: float = Field(default=0.0, description="Number of times task was completed")
    reward_ema: float = Field(default=0.0, description="Exponential moving average of rewards")
    success_rate_ema: float = Field(default=0.0, description="EMA of success rate (0-1)")
    ema_squared: float = Field(default=0.0, description="EMA of squared rewards (for variance)")
    total_score: float = Field(default=0.0, description="Sum of all scores")
    last_score: float = Field(default=0.0, description="Most recent score")

    # Learning progress scoring
    lp_score: float = Field(default=0.0, description="Learning progress score (sampling probability)")

    # Bidirectional learning progress EMAs
    p_fast: float = Field(default=0.0, description="Fast EMA for bidirectional LP")
    p_slow: float = Field(default=0.0, description="Slow EMA for bidirectional LP")
    p_true: float = Field(default=0.0, description="True performance EMA (unnormalized)")
    random_baseline: float = Field(default=0.0, description="Random agent baseline performance")

    # Task configuration
    success_threshold: float = Field(default=0.0, description="Success threshold for this task")
    seed: float = Field(default=0.0, description="Random seed for task generation")
    generator_type: float = Field(default=0.0, description="Type of task generator used")
    label_hash: float = Field(default=0.0, description="Hash of task label for grouping")

    @classmethod
    def struct_size(cls) -> int:
        """Compute the struct size dynamically from the number of fields."""
        return len(cls.model_fields)

    def to_array(self) -> np.ndarray:
        """Convert TaskState to numpy array for storage."""
        return np.array(
            [
                self.task_id,
                self.creation_time,
                self.completion_count,
                self.reward_ema,
                self.lp_score,
                self.success_rate_ema,
                self.total_score,
                self.last_score,
                self.success_threshold,
                self.seed,
                self.generator_type,
                self.ema_squared,
                self.is_active,
                self.p_fast,
                self.p_slow,
                self.p_true,
                self.random_baseline,
                self.label_hash,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TaskState":
        """Create TaskState from numpy array."""
        return cls(
            task_id=float(arr[0]),
            creation_time=float(arr[1]),
            completion_count=float(arr[2]),
            reward_ema=float(arr[3]),
            lp_score=float(arr[4]),
            success_rate_ema=float(arr[5]),
            total_score=float(arr[6]),
            last_score=float(arr[7]),
            success_threshold=float(arr[8]),
            seed=float(arr[9]),
            generator_type=float(arr[10]),
            ema_squared=float(arr[11]),
            is_active=float(arr[12]),
            p_fast=float(arr[13]),
            p_slow=float(arr[14]),
            p_true=float(arr[15]),
            random_baseline=float(arr[16]),
            label_hash=float(arr[17]),
        )


class TaskMemoryBackend(ABC):
    """Abstract interface for task memory storage.

    This interface hides the implementation details of local vs shared memory,
    allowing the rest of the curriculum system to work identically regardless
    of whether running single-process or multi-process training.

    The task_struct_size is computed dynamically from TaskState.struct_size().
    """

    # Required attributes (must be set by subclasses)
    max_tasks: int
    task_struct_size: int

    @abstractmethod
    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view).

        Returns numpy array of length task_struct_size that can be read/written directly.
        Prefer get_task_state() for type-safe access.
        """
        ...

    def get_task_state(self, index: int) -> TaskState:
        """Get task state at given index (type-safe).

        Returns TaskState object with typed fields.
        """
        return TaskState.from_array(self.get_task_data(index))

    def set_task_state(self, index: int, state: TaskState) -> None:
        """Set task state at given index (type-safe).

        Args:
            index: Task index
            state: TaskState object to write
        """
        task_data = self.get_task_data(index)
        task_data[:] = state.to_array()

    @abstractmethod
    def acquire_lock(self) -> ContextManager[Any]:
        """Acquire lock for thread-safe access.

        Returns context manager for use with 'with' statement.
        For local backend, this may be a no-op lock.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory data."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (close files, free memory, etc.)."""
        ...


class LocalMemoryBackend(TaskMemoryBackend):
    """In-memory backend for single-process use.

    Uses simple numpy arrays stored in local process memory.
    No inter-process communication overhead.
    """

    def __init__(
        self,
        max_tasks: int,
        task_struct_size: Optional[int] = None,
    ):
        """Initialize local memory backend.

        Args:
            max_tasks: Maximum number of tasks to track (required, set from LearningProgressConfig)
            task_struct_size: Size of each task's data structure (default: computed from TaskState)
        """
        self.max_tasks = max_tasks
        self.task_struct_size = task_struct_size if task_struct_size is not None else TaskState.struct_size()

        # Allocate local numpy arrays
        self._task_array = np.zeros((max_tasks, self.task_struct_size), dtype=np.float64)

        # True no-op lock for local memory (single process, single thread)
        # Using nullcontext avoids threading lock overhead
        self._lock = nullcontext()

    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view)."""
        return self._task_array[index]

    def acquire_lock(self) -> ContextManager[Any]:
        """Acquire lock (true no-op for local memory - returns nullcontext)."""
        return self._lock

    def clear(self):
        """Clear all memory data."""
        self._task_array.fill(0.0)

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

    Synchronization: Uses multiprocessing.Manager().Lock() to ensure atomic multi-field
    updates. This prevents race conditions when updating related values (e.g., bidirectional
    EMAs). Manager.Lock() returns a proxy object that can be properly pickled and shared
    across processes, ensuring all processes synchronize on the same server-backed lock.
    """

    def __init__(
        self,
        max_tasks: int,
        session_id: Optional[str] = None,
        task_struct_size: Optional[int] = None,
    ):
        """Initialize shared memory backend.

        Args:
            max_tasks: Maximum number of tasks to track in shared memory (required, set from LearningProgressConfig)
            session_id: Unique identifier for this shared memory session.
                       All processes sharing state must use the same session_id.
                       If None, creates a unique session (not shared across processes).
                       NOTE: When using LearningProgressConfig with use_shared_memory=True,
                       session_id is auto-generated at config creation time and shared across
                       all processes, so this fallback should rarely be used.
            task_struct_size: Size of each task's data structure (default: computed from TaskState)
        """
        self.max_tasks = max_tasks
        self.task_struct_size = task_struct_size if task_struct_size is not None else TaskState.struct_size()

        # Initialize shared memory handle to None (set by _init_shared_memory)
        self._task_array_shm = None
        self._task_array = None

        # Generate session ID if not provided (fallback for non-config usage)
        if session_id is None:
            session_id = f"cur_{uuid.uuid4().hex[:6]}"
        self.session_id = session_id

        # Create shared memory names based on session
        # POSIX shared memory names have strict length limits (31 chars on macOS)
        # Use short prefixes to stay within limits
        self._task_array_name = f"ta_{session_id}"  # task array

        # Track if we created the shared memory (for cleanup)
        self._created_shared_memory = False

        # Initialize shared structures
        self._init_shared_memory()

        # Create a Manager and its lock for proper cross-process synchronization
        # Manager.Lock() returns a proxy that can be pickled and properly shared
        # across processes, unlike regular Lock() which creates independent locks
        # in each process after unpickling.
        self._manager = Manager()
        self._lock = self._manager.Lock()

    def _init_shared_memory(self):
        """Initialize shared memory structures."""
        # Calculate sizes
        task_array_size = self.max_tasks * self.task_struct_size * 8  # 8 bytes per float64

        try:
            # Try to connect to existing shared memory
            self._task_array_shm = shared_memory.SharedMemory(name=self._task_array_name)
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
            self._created_shared_memory = True

            # Initialize to zero only if we created it
            task_array = np.ndarray(
                (self.max_tasks, self.task_struct_size), dtype=np.float64, buffer=self._task_array_shm.buf
            )
            task_array.fill(0.0)

        # Create numpy views
        self._task_array = np.ndarray(
            (self.max_tasks, self.task_struct_size), dtype=np.float64, buffer=self._task_array_shm.buf
        )

    def get_task_data(self, index: int) -> np.ndarray:
        """Get task data at given index (raw array view)."""
        return self._task_array[index]

    def acquire_lock(self) -> ContextManager[Any]:
        """Acquire lock for thread-safe access.

        Returns a multiprocessing.Lock to ensure atomic multi-field updates.
        This prevents race conditions when updating multiple related values
        (e.g., bidirectional EMAs: p_fast, p_slow, p_true, random_baseline).
        """
        return self._lock

    def clear(self):
        """Clear all shared memory data."""
        self._task_array.fill(0.0)

    def cleanup(self):
        """Clean up shared memory resources."""
        if self._task_array_shm is None:
            return

        try:
            self._task_array_shm.close()
            if self._created_shared_memory:
                try:
                    self._task_array_shm.unlink()
                except FileNotFoundError:
                    # Already unlinked by another process - this is fine
                    pass
        except Exception as e:
            logger.warning(
                f"Failed to cleanup shared memory (session={self.session_id}, name={self._task_array_name}): {e}"
            )

        # Shutdown manager if we have a reference to it (only in main process)
        if hasattr(self, "_manager") and self._manager is not None:
            try:
                self._manager.shutdown()
                self._manager = None
            except Exception as e:
                logger.debug(f"Failed to shutdown manager (may already be shutdown): {e}")

    def __del__(self):
        """Cleanup on destruction."""
        # Only access _task_array_shm if it was initialized
        if not hasattr(self, "_task_array_shm") or self._task_array_shm is None:
            return

        session_id = getattr(self, "session_id", "unknown")
        name = getattr(self, "_task_array_name", "unknown")

        try:
            self._task_array_shm.close()
        except Exception as e:
            logger.warning(f"Failed to close shared memory in destructor (session={session_id}, name={name}): {e}")

    def __getstate__(self):
        """Prepare for pickling - save connection parameters.

        The lock is included in the state because Manager.Lock() returns a proxy
        object that can be properly pickled and shared across processes.
        """
        return {
            "max_tasks": self.max_tasks,
            "task_struct_size": self.task_struct_size,
            "session_id": self.session_id,
            "created_shared_memory": self._created_shared_memory,
            "lock": self._lock,
        }

    def __setstate__(self, state):
        """Restore from pickle - reconnect to existing shared memory.

        The lock is restored from the pickled state rather than recreated. Since
        it's a Manager.Lock() proxy, all processes will reference the same
        server-side lock, ensuring proper synchronization across processes.
        """
        self.max_tasks = state["max_tasks"]
        self.task_struct_size = state["task_struct_size"]
        self.session_id = state["session_id"]

        # Initialize shared memory handle to None (set by _init_shared_memory)
        self._task_array_shm = None
        self._task_array = None

        # Worker processes never created it, only connect
        self._created_shared_memory = False

        # Set shared memory names before reconnecting
        self._task_array_name = f"ta_{self.session_id}"

        # Reconnect to existing shared memory
        self._init_shared_memory()

        # Restore the shared lock from pickled state (not create a new one!)
        # The Manager proxy maintains its connection to the manager server
        self._lock = state["lock"]

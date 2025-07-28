import logging
import os
import pickle
import random
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional

import filelock
import numpy as np

logger = logging.getLogger(__name__)

# Shared memory layout constants
TASK_ID_SIZE = 36  # UUID string length
TASK_NAME_SIZE = 256
ENV_CFG_SIZE = 65536  # 64KB for pickled env config

# Define numpy dtypes for structured access
HEADER_DTYPE = np.dtype(
    [
        ("ready", np.uint32),
        ("max_slots", np.uint32),  # Maximum capacity
        ("num_active_slots", np.uint32),  # Currently active slots
    ]
)

TASK_SLOT_DTYPE = np.dtype(
    [
        ("task_id", f"S{TASK_ID_SIZE}"),
        ("task_name", f"S{TASK_NAME_SIZE}"),
        ("env_cfg", f"S{ENV_CFG_SIZE}"),
        ("num_completed", np.uint32),
        ("mean_score", np.float32),
        ("num_outstanding", np.uint32),
        ("occupied", np.uint8),
    ]
)

HEADER_SIZE = HEADER_DTYPE.itemsize
SLOT_SIZE = TASK_SLOT_DTYPE.itemsize


class CurriculumState:
    """Manages shared memory state for curriculum tasks.

    Uses a dynamic number of task slots that can grow as needed, up to a maximum capacity.
    """

    def __init__(self, max_slots: int = 10000, name: str = "curriculum_server", create: bool = True):
        """Initialize the curriculum state manager.

        Args:
            max_slots: Maximum number of task slots that can be allocated.
            name: Name of the shared memory segment
            create: If True, create new shared memory. If False, attach to existing.
        """
        self.name = name
        # Create a lock file for process-safe synchronization
        self.lock_path = f"/tmp/curriculum_state_{name}.lock"
        self._lock = filelock.FileLock(self.lock_path)

        if create:
            self.max_slots = max_slots
            self.total_size = HEADER_SIZE + (SLOT_SIZE * max_slots)

            # Try to unlink any existing shared memory with the same name
            try:
                existing_shm = shared_memory.SharedMemory(name=name)
                existing_shm.close()
                existing_shm.unlink()
                logger.info("Cleaned up existing shared memory")
            except FileNotFoundError:
                pass  # No existing shared memory, which is fine
            except Exception as e:
                logger.warning(f"Error cleaning up existing shared memory: {e}")

            # Create shared memory
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=name)
                logger.info(
                    f"Created shared memory '{self.shm.name}' with max {max_slots} "
                    f"slots (size: {self.total_size} bytes)"
                )
            except Exception as e:
                logger.error(f"Failed to create shared memory: {e}")
                raise

            # Create structured arrays for header and slots
            self._init_arrays()

            # Initialize header
            self.header["ready"] = 0
            self.header["max_slots"] = max_slots
            self.header["num_active_slots"] = 0  # Start with no active slots

            # Initialize all slots as unoccupied
            self.slots["occupied"] = 0
        else:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name)

            # Create structured arrays
            self._init_arrays()

            # Read max_slots from header
            self.max_slots = int(self.header["max_slots"])

            logger.debug(f"Attached to existing shared memory '{name}' with max {self.max_slots} slots")

    @classmethod
    def connect(
        cls, name: str = "curriculum_server", timeout: float = 30.0, wait_ready: bool = True
    ) -> "CurriculumState":
        """Connect to an existing CurriculumState with retries.

        Args:
            name: Name of the shared memory segment to connect to
            timeout: Timeout in seconds for connecting to the server
            wait_ready: Whether to wait for the ready flag after connecting

        Returns:
            Connected CurriculumState instance

        Raises:
            ConnectionError: If unable to connect within timeout
        """
        start_time = time.time()
        last_error = None
        retry_count = 0

        while time.time() - start_time < timeout:
            try:
                # Try to connect to existing shared memory
                state = cls(name=name, create=False)

                # If we don't need to wait for ready, return immediately
                if not wait_ready:
                    return state

                # Wait for server to signal it's ready
                ready_start = time.time()
                while time.time() - ready_start < 300.0:  # Wait up to 5 minutes
                    if state.is_ready():
                        return state
                    time.sleep(0.1)

                # If we get here, server wasn't ready in time
                state.close()
                raise ConnectionError(
                    "Curriculum server shared memory found but server not ready after 300s. "
                    "The server may still be initializing tasks."
                )

            except FileNotFoundError as e:
                last_error = e
                retry_count += 1
                if retry_count == 1:
                    logger.info(f"Waiting for curriculum server '{name}' to be available...")
                elif retry_count % 10 == 0:  # Log every second
                    elapsed = time.time() - start_time
                    logger.info(f"Still waiting for curriculum server '{name}'... ({elapsed:.1f}s elapsed)")
                time.sleep(0.1)  # Wait 100ms before retrying
            except ConnectionError:
                # Re-raise connection errors (server not ready)
                raise
            except Exception as e:
                # Unexpected error, log and continue retrying
                logger.debug(f"Error connecting to curriculum server: {e}")
                last_error = e
                time.sleep(0.1)

        # If we exhausted the timeout, raise an error
        raise ConnectionError(
            f"Failed to connect to curriculum server '{name}' after {timeout}s. "
            f"Is the server running? Last error: {last_error}"
        )

    def _init_arrays(self):
        """Initialize numpy structured arrays for header and slots."""
        # Create header array
        self.header = np.ndarray(shape=(), dtype=HEADER_DTYPE, buffer=self.shm.buf, offset=0)

        # Determine max number of slots
        if hasattr(self, "max_slots"):
            max_slots = self.max_slots
        else:
            # Read from header to get max_slots
            temp_header = np.ndarray(shape=(), dtype=HEADER_DTYPE, buffer=self.shm.buf, offset=0)
            max_slots = int(temp_header["max_slots"])
            self.max_slots = max_slots

        # Create slots array (for all possible slots)
        self.slots = np.ndarray(shape=(max_slots,), dtype=TASK_SLOT_DTYPE, buffer=self.shm.buf, offset=HEADER_SIZE)

    def set_ready(self, ready: bool = True):
        """Set the ready flag in shared memory."""
        with self._lock:
            self.header["ready"] = 1 if ready else 0
        logger.info(f"Set ready flag to {ready}")

    def is_ready(self) -> bool:
        """Check if the shared memory is marked as ready."""
        with self._lock:
            return bool(self.header["ready"])

    def get_num_active_slots(self) -> int:
        """Get the current number of active slots."""
        with self._lock:
            return int(self.header["num_active_slots"])

    def add_slot(self) -> Optional[int]:
        """Add a new active slot.

        Returns:
            The index of the new slot, or None if at max capacity
        """
        with self._lock:
            current_active = int(self.header["num_active_slots"])
            if current_active >= self.max_slots:
                logger.warning(f"Cannot add slot: at maximum capacity ({self.max_slots})")
                return None

            # The new slot index is the current number of active slots
            new_slot_idx = current_active

            # Increment the active slot count
            self.header["num_active_slots"] = current_active + 1

            # Ensure the slot is marked as unoccupied initially
            self.slots[new_slot_idx]["occupied"] = 0

            logger.debug(f"Added new slot {new_slot_idx} (now {current_active + 1} active slots)")
            return new_slot_idx

    def add_task(self, task_id: str, task_name: str, env_cfg: Any) -> Optional[int]:
        """Add a new task to the curriculum.

        First tries to find an empty slot among existing active slots.
        If none are available and all slots have outstanding work, adds a new slot.

        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            env_cfg: Environment configuration

        Returns:
            Slot index where task was added, or None if no slots available
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])

            # First, try to find an empty slot among active slots
            for slot_idx in range(num_active):
                if not self.slots[slot_idx]["occupied"]:
                    # Found an empty slot, use it
                    self._write_slot_data(slot_idx, task_id, task_name, env_cfg)
                    logger.debug(f"Added task '{task_name}' to existing slot {slot_idx}")
                    return slot_idx

            # No empty slots found. Check if all tasks have outstanding work
            all_have_outstanding = True
            for slot_idx in range(num_active):
                if self.slots[slot_idx]["occupied"] and self.slots[slot_idx]["num_outstanding"] == 0:
                    all_have_outstanding = False
                    break

            # If all tasks have outstanding work, try to add a new slot
            if all_have_outstanding:
                if num_active < self.max_slots:
                    new_slot_idx = num_active
                    self.header["num_active_slots"] = num_active + 1
                    self._write_slot_data(new_slot_idx, task_id, task_name, env_cfg)
                    logger.debug(
                        f"Added task '{task_name}' to new slot {new_slot_idx} (now {num_active + 1} active slots)"
                    )
                    return new_slot_idx
                else:
                    logger.warning(f"Cannot add task: at maximum capacity ({self.max_slots}) and all slots occupied")
                    return None
            else:
                logger.warning("Cannot add task: empty slots exist but tasks without outstanding work remain")
                return None

    def _write_slot_data(self, slot_idx: int, task_id: str, task_name: str, env_cfg: Any):
        """Write task data to a slot (internal helper, assumes lock is held)."""
        # Prepare data
        task_id_bytes = task_id.encode("utf-8")[:TASK_ID_SIZE]
        task_name_bytes = task_name.encode("utf-8")[:TASK_NAME_SIZE]
        env_cfg_pickled = pickle.dumps(env_cfg)

        if len(env_cfg_pickled) > ENV_CFG_SIZE:
            raise ValueError(f"Pickled env_cfg too large: {len(env_cfg_pickled)} > {ENV_CFG_SIZE}")

        # Pad strings to full size
        task_id_padded = task_id_bytes.ljust(TASK_ID_SIZE, b"\x00")
        task_name_padded = task_name_bytes.ljust(TASK_NAME_SIZE, b"\x00")
        env_cfg_padded = env_cfg_pickled.ljust(ENV_CFG_SIZE, b"\x00")

        # Write task data
        self.slots[slot_idx]["task_id"] = task_id_padded
        self.slots[slot_idx]["task_name"] = task_name_padded
        self.slots[slot_idx]["env_cfg"] = env_cfg_padded
        self.slots[slot_idx]["num_completed"] = 0
        self.slots[slot_idx]["mean_score"] = 0.0
        self.slots[slot_idx]["num_outstanding"] = 0
        self.slots[slot_idx]["occupied"] = 1

    def get_task(self) -> Optional[Dict]:
        """Get a task to work on.

        Samples 3 random tasks and returns the one with fewest completions.
        Increments the outstanding count for the selected task.
        Updates local knowledge of active slot count.

        Returns:
            Task dictionary with task details, or None if no tasks available
        """
        with self._lock:
            # Update our knowledge of active slots
            num_active = int(self.header["num_active_slots"])

            # Get all occupied slots from active slots only
            occupied_slots = [i for i in range(num_active) if self.slots[i]["occupied"]]

            if not occupied_slots:
                return None

            # Sample up to 3 tasks
            sample_size = min(3, len(occupied_slots))
            sampled_slots = random.sample(occupied_slots, sample_size)

            # Find the task with fewest completions
            best_slot = None
            min_completions = float("inf")

            for slot_idx in sampled_slots:
                num_completed = self.slots[slot_idx]["num_completed"]
                if num_completed < min_completions:
                    min_completions = num_completed
                    best_slot = slot_idx

            if best_slot is None:
                return None

            # Increment outstanding count
            self.slots[best_slot]["num_outstanding"] += 1

            # Read and return task data
            task_data = self._read_slot(best_slot)

            logger.debug(
                f"Selected task '{task_data['task_name']}' from slot {best_slot} "
                f"(completions: {task_data['num_completed']}, outstanding: {task_data['num_outstanding']}) "
                f"[{num_active} active slots]"
            )

            return task_data

    def complete_task(self, task_id: str, score: float) -> bool:
        """Mark a task as completed with a score.

        Updates the task statistics and decrements the outstanding count.

        Args:
            task_id: ID of the task to complete
            score: Score achieved on the task

        Returns:
            True if task was found and updated, False otherwise
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])

            # Find the task by ID (only search active slots)
            for slot_idx in range(num_active):
                if self.slots[slot_idx]["occupied"]:
                    stored_id = self.slots[slot_idx]["task_id"].decode("utf-8").rstrip("\x00")
                    if stored_id == task_id:
                        # Update statistics
                        slot = self.slots[slot_idx]
                        old_num_completed = slot["num_completed"]
                        old_mean = slot["mean_score"]

                        slot["num_completed"] = old_num_completed + 1
                        slot["mean_score"] = (old_mean * old_num_completed + score) / slot["num_completed"]
                        slot["num_outstanding"] = max(0, slot["num_outstanding"] - 1)

                        task_name = slot["task_name"].decode("utf-8").rstrip("\x00")
                        logger.debug(
                            f"Completed task '{task_name}' with score {score:.2f} "
                            f"(total completions: {slot['num_completed']}, "
                            f"mean: {slot['mean_score']:.2f}, "
                            f"outstanding: {slot['num_outstanding']})"
                        )
                        return True

            logger.warning(f"Task with ID '{task_id}' not found")
            return False

    def increment_outstanding(self, slot_idx: int) -> None:
        """Increment the outstanding count for a task when it's sampled.

        Args:
            slot_idx: The slot index to increment outstanding count for
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])

            # Verify slot is valid and occupied
            if slot_idx >= num_active:
                raise ValueError(f"Invalid slot index {slot_idx}, only {num_active} active slots")

            if not self.slots[slot_idx]["occupied"]:
                raise ValueError(f"Slot {slot_idx} is not occupied")

            # Increment the outstanding count
            current_outstanding = int(self.slots[slot_idx]["num_outstanding"])
            self.slots[slot_idx]["num_outstanding"] = current_outstanding + 1

            # Read task info for logging
            task_name = self.slots[slot_idx]["task_name"].decode("utf-8").rstrip("\x00")
            num_completed = int(self.slots[slot_idx]["num_completed"])
            new_outstanding = int(self.slots[slot_idx]["num_outstanding"])

            logger.debug(
                f"Sampled task '{task_name}' from slot {slot_idx} "
                f"(completions: {num_completed}, outstanding: {new_outstanding})"
            )

    def get_completable_tasks(self) -> List[Dict]:
        """Get all tasks that have no outstanding work.

        Returns:
            List of task dictionaries for tasks with num_outstanding == 0
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])
            completable_tasks = []

            for slot_idx in range(num_active):
                if self.slots[slot_idx]["occupied"] and self.slots[slot_idx]["num_outstanding"] == 0:
                    task_data = self._read_slot(slot_idx)
                    completable_tasks.append(task_data)
            return completable_tasks

    def _read_slot(self, slot_idx: int) -> Dict:
        """Read task data from a slot (internal, no locking)."""
        slot = self.slots[slot_idx]

        # Decode strings
        task_id = slot["task_id"].decode("utf-8").rstrip("\x00")
        task_name = slot["task_name"].decode("utf-8").rstrip("\x00")

        # Unpickle env_cfg
        env_cfg_bytes = bytes(slot["env_cfg"])
        env_cfg_len = len(env_cfg_bytes.rstrip(b"\x00"))
        if env_cfg_len > 0:
            env_cfg = pickle.loads(env_cfg_bytes[:env_cfg_len])
        else:
            env_cfg = None

        return {
            "task_id": task_id,
            "task_name": task_name,
            "env_cfg": env_cfg,
            "num_completed": int(slot["num_completed"]),
            "mean_score": float(slot["mean_score"]),
            "num_outstanding": int(slot["num_outstanding"]),
            "slot_idx": slot_idx,
        }

    def write_task(
        self, slot_idx: int, task_id: str, task_name: str, env_cfg: Any, num_completed: int = 0, mean_score: float = 0.0
    ):
        """Write task data to a specific slot.

        Args:
            slot_idx: Slot index to write to
            task_id: Unique task identifier
            task_name: Human-readable task name
            env_cfg: Environment configuration (will be pickled)
            num_completed: Number of times task has been completed
            mean_score: Mean score across completions
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])
            if slot_idx < 0 or slot_idx >= num_active:
                raise ValueError(f"Invalid slot index: {slot_idx} (only {num_active} active slots)")

            # Use the internal helper
            self._write_slot_data(slot_idx, task_id, task_name, env_cfg)

            # Set the completion stats
            self.slots[slot_idx]["num_completed"] = num_completed
            self.slots[slot_idx]["mean_score"] = mean_score

    def read_task(self, slot_idx: int) -> Dict:
        """Read task data from a specific slot.

        Args:
            slot_idx: Slot index to read from

        Returns:
            Dictionary containing task data
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])
            if slot_idx < 0 or slot_idx >= num_active:
                raise ValueError(f"Invalid slot index: {slot_idx} (only {num_active} active slots)")
            return self._read_slot(slot_idx)

    def update_task_stats(self, slot_idx: int, score: float):
        """Update task statistics by adding a new completion score.

        Args:
            slot_idx: Slot index to update
            score: Score achieved on the task
        """
        with self._lock:
            num_active = int(self.header["num_active_slots"])
            if slot_idx < 0 or slot_idx >= num_active:
                raise ValueError(f"Invalid slot index: {slot_idx} (only {num_active} active slots)")

            # Get current statistics
            slot = self.slots[slot_idx]
            old_num_completed = slot["num_completed"]
            old_mean = slot["mean_score"]

            # Calculate new statistics
            new_num_completed = old_num_completed + 1
            new_mean = (old_mean * old_num_completed + score) / new_num_completed

            # Update slot
            self.slots[slot_idx]["num_completed"] = new_num_completed
            self.slots[slot_idx]["mean_score"] = new_mean

    def close(self):
        """Close the shared memory connection (doesn't unlink)."""
        self.shm.close()

    def cleanup(self):
        """Close and unlink (delete) the shared memory."""
        self.shm.close()
        self.shm.unlink()
        # Clean up lock file
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass
        logger.info(f"Cleaned up shared memory '{self.name}'")

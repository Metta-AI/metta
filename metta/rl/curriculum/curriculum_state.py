import logging
import os
import pickle
import random
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
        ("num_slots", np.uint32),
        ("num_active_tasks", np.uint32),
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

    Pre-allocates a fixed number of task slots in shared memory that can be
    accessed by multiple processes.
    """

    def __init__(self, num_slots: Optional[int] = None, name: str = "curriculum_server", create: bool = True):
        """Initialize the curriculum state manager.

        Args:
            num_slots: Number of task slots to allocate. Required when create=True.
                       When create=False, this will be auto-detected from shared memory header.
            name: Name of the shared memory segment
            create: If True, create new shared memory. If False, attach to existing.
        """
        self.name = name
        # Create a lock file for process-safe synchronization
        self.lock_path = f"/tmp/curriculum_state_{name}.lock"
        self._lock = filelock.FileLock(self.lock_path)

        if create:
            if num_slots is None:
                raise ValueError("num_slots is required when creating shared memory (create=True)")
            self.num_slots = num_slots
            self.total_size = HEADER_SIZE + (SLOT_SIZE * num_slots)

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
                    f"Created shared memory '{self.shm.name}' with {num_slots} slots (size: {self.total_size} bytes)"
                )
            except Exception as e:
                logger.error(f"Failed to create shared memory: {e}")
                raise

            # Create structured arrays for header and slots
            self._init_arrays()

            # Initialize header
            self.header["ready"] = 0
            self.header["num_slots"] = num_slots
            self.header["num_active_tasks"] = 0

            # Initialize all slots as unoccupied
            self.slots["occupied"] = 0
        else:
            # Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name)

            # Create structured arrays
            self._init_arrays()

            # Read num_slots from header
            self.num_slots = int(self.header["num_slots"])

            # If num_slots was provided, validate it matches
            if num_slots is not None and num_slots != self.num_slots:
                logger.warning(
                    f"Provided num_slots ({num_slots}) doesn't match stored slots ({self.num_slots}). "
                    f"Using stored value."
                )

            logger.info(f"Attached to existing shared memory '{name}' with {self.num_slots} slots")

    def _init_arrays(self):
        """Initialize numpy structured arrays for header and slots."""
        # Create header array
        self.header = np.ndarray(shape=(), dtype=HEADER_DTYPE, buffer=self.shm.buf, offset=0)

        # Determine number of slots (for attaching to existing memory)
        if hasattr(self, "num_slots"):
            num_slots = self.num_slots
        else:
            # Read from header to get num_slots
            temp_header = np.ndarray(shape=(), dtype=HEADER_DTYPE, buffer=self.shm.buf, offset=0)
            num_slots = int(temp_header["num_slots"])

        # Create slots array
        self.slots = np.ndarray(shape=(num_slots,), dtype=TASK_SLOT_DTYPE, buffer=self.shm.buf, offset=HEADER_SIZE)

    def set_ready(self, ready: bool = True):
        """Set the ready flag in shared memory."""
        with self._lock:
            self.header["ready"] = 1 if ready else 0
        logger.info(f"Set ready flag to {ready}")

    def is_ready(self) -> bool:
        """Check if the shared memory is marked as ready."""
        with self._lock:
            return bool(self.header["ready"])

    def add_task(self, task_id: str, task_name: str, env_cfg: Any) -> Optional[int]:
        """Add a new task to the curriculum.

        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            env_cfg: Environment configuration

        Returns:
            Slot index where task was added, or None if no slots available
        """
        with self._lock:
            # Find an empty slot
            for slot_idx in range(self.num_slots):
                if not self.slots[slot_idx]["occupied"]:
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

                    # Update active task count
                    self.header["num_active_tasks"] += 1

                    logger.info(f"Added task '{task_name}' to slot {slot_idx}")
                    return slot_idx

            logger.warning("No available slots for new task")
            return None

    def get_task(self) -> Optional[Dict]:
        """Get a task to work on.

        Samples 3 random tasks and returns the one with fewest completions.
        Increments the outstanding count for the selected task.

        Returns:
            Task dictionary with task details, or None if no tasks available
        """
        with self._lock:
            # Get all occupied slots
            occupied_slots = [i for i in range(self.num_slots) if self.slots[i]["occupied"]]

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

            logger.info(
                f"Selected task '{task_data['task_name']}' from slot {best_slot} "
                f"(completions: {task_data['num_completed']}, outstanding: {task_data['num_outstanding']})"
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
            # Find the task by ID
            for slot_idx in range(self.num_slots):
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
                        logger.info(
                            f"Completed task '{task_name}' with score {score:.2f} "
                            f"(total completions: {slot['num_completed']}, "
                            f"mean: {slot['mean_score']:.2f}, "
                            f"outstanding: {slot['num_outstanding']})"
                        )
                        return True

            logger.warning(f"Task with ID '{task_id}' not found")
            return False

    def get_completable_tasks(self) -> List[Dict]:
        """Get all tasks that have no outstanding work.

        Returns:
            List of task dictionaries for tasks with num_outstanding == 0
        """
        with self._lock:
            completable_tasks = []
            for slot_idx in range(self.num_slots):
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
            if slot_idx < 0 or slot_idx >= self.num_slots:
                raise ValueError(f"Invalid slot index: {slot_idx}")

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

            # Write data
            self.slots[slot_idx]["task_id"] = task_id_padded
            self.slots[slot_idx]["task_name"] = task_name_padded
            self.slots[slot_idx]["env_cfg"] = env_cfg_padded
            self.slots[slot_idx]["num_completed"] = num_completed
            self.slots[slot_idx]["mean_score"] = mean_score
            self.slots[slot_idx]["num_outstanding"] = 0
            self.slots[slot_idx]["occupied"] = 1

    def read_task(self, slot_idx: int) -> Dict:
        """Read task data from a specific slot.

        Args:
            slot_idx: Slot index to read from

        Returns:
            Dictionary containing task data
        """
        with self._lock:
            if slot_idx < 0 or slot_idx >= self.num_slots:
                raise ValueError(f"Invalid slot index: {slot_idx}")
            return self._read_slot(slot_idx)

    def update_task_stats(self, slot_idx: int, num_completed: int, mean_score: float):
        """Update just the statistics for a task without rewriting the whole slot.

        Args:
            slot_idx: Slot index to update
            num_completed: New number of completions
            mean_score: New mean score
        """
        with self._lock:
            if slot_idx < 0 or slot_idx >= self.num_slots:
                raise ValueError(f"Invalid slot index: {slot_idx}")

            self.slots[slot_idx]["num_completed"] = num_completed
            self.slots[slot_idx]["mean_score"] = mean_score

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

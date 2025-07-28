import logging
import pickle
import random
import struct
import time
from multiprocessing import shared_memory
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)

# Shared memory layout constants (must match server)
TASK_ID_SIZE = 36  # UUID string length
TASK_NAME_SIZE = 256
ENV_CFG_SIZE = 65536  # 64KB for pickled env config
SLOT_SIZE = (
    TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE + 4 + 4 + 4
)  # +4 for num_completed, +4 for mean_score, +4 for lock
HEADER_SIZE = 4  # 4 bytes for ready flag at the beginning of shared memory


class RemoteTask(Task):
    """A task that represents a remote task from the curriculum server."""

    def __init__(self, task_id: str, task_name: str, env_cfg: DictConfig, client: "CurriculumClient", slot_idx: int):
        # Initialize without calling parent __init__ since we handle things differently
        self._id = task_id
        self._name = task_name
        self._is_complete = False
        self._env_cfg = env_cfg
        self._client = client
        self._slot_idx = slot_idx

    def complete(self, score: float):
        """Complete the task by updating shared memory."""
        if self._is_complete:
            logger.warning(f"Task {self._id} is already complete")
            return

        # Convert score to Python float in case it's a numpy type
        score = float(score)

        # Update shared memory
        success = self._client.complete_task(self._slot_idx, score)
        if success:
            self._is_complete = True
            logger.debug(f"Task {self._name} completed with score {score}")
        else:
            logger.error(f"Failed to complete task {self._name} in shared memory")

    def short_name(self) -> str:
        return self._name.split("/")[-1]


class CurriculumClient(Curriculum):
    """Client that connects to a curriculum server via shared memory.

    The server pre-allocates all task slots, so clients can immediately start
    working without waiting for task creation. Tasks are read without locks
    for maximum performance.
    """

    def __init__(self, num_slots: int = 100, timeout: float = 30.0):
        self.num_slots = num_slots
        self.timeout = timeout

        # Connect to shared memory with retries
        start_time = time.time()
        connected = False
        last_error = None
        retry_count = 0

        while time.time() - start_time < timeout:
            try:
                self.shm = shared_memory.SharedMemory(name="curriculum_server")
                connected = True
                break
            except FileNotFoundError as e:
                last_error = e
                retry_count += 1
                if retry_count == 1:
                    logger.info("Waiting for curriculum server shared memory to be available...")
                elif retry_count % 10 == 0:  # Log every second
                    elapsed = time.time() - start_time
                    logger.info(f"Still waiting for curriculum server... ({elapsed:.1f}s elapsed)")
                time.sleep(0.1)  # Wait 100ms before retrying

        if not connected:
            raise ConnectionError(
                f"Curriculum server shared memory not found after {timeout}s. "
                f"Is the server running? Last error: {last_error}"
            )

        # Create numpy array for easier access
        self.buffer = np.ndarray((self.shm.size,), dtype=np.uint8, buffer=self.shm.buf)

        # Wait for server to signal it's ready
        ready_start = time.time()
        while time.time() - ready_start < 10.0:  # Wait up to 10 seconds
            ready_flag = struct.unpack_from("I", self.buffer.data, 0)[0]
            if ready_flag == 1:
                break
            time.sleep(0.1)
        else:
            raise ConnectionError(
                "Curriculum server shared memory found but server not ready after 10s. "
                "The server may still be initializing tasks."
            )

        # Calculate expected size
        expected_size = HEADER_SIZE + (SLOT_SIZE * num_slots)
        if self.shm.size != expected_size:
            actual_slots = (self.shm.size - HEADER_SIZE) // SLOT_SIZE
            logger.warning(f"Shared memory size mismatch. Expected {num_slots} slots, found {actual_slots}")
            self.num_slots = actual_slots

        # Verify at least some slots have tasks
        logger.info(f"Connected to curriculum server shared memory with {self.num_slots} slots")
        self._verify_connection()

    def _verify_connection(self):
        """Verify that the shared memory contains valid tasks."""
        valid_count = 0
        for i in range(min(5, self.num_slots)):  # Check first 5 slots
            try:
                slot_data = self._read_slot(i)
                if slot_data["task_id"]:
                    valid_count += 1
            except Exception as e:
                logger.debug(f"Error reading slot {i} during verification: {e}")

        if valid_count == 0:
            logger.warning(
                f"No valid tasks found in first {min(5, self.num_slots)} slots. Server may still be initializing."
            )

    def _get_slot_offset(self, slot_idx: int) -> int:
        """Get the byte offset for a specific slot."""
        return HEADER_SIZE + (slot_idx * SLOT_SIZE)

    def _acquire_lock(self, slot_idx: int, timeout: float = 1.0) -> bool:
        """Acquire lock for a slot using compare-and-swap."""
        offset = self._get_slot_offset(slot_idx)
        lock_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE + 8  # +8 for num_completed and mean_score

        start_time = time.time()
        while time.time() - start_time < timeout:
            current = struct.unpack_from("I", self.buffer.data, lock_offset)[0]
            if current == 0:  # Unlocked
                struct.pack_into("I", self.buffer.data, lock_offset, 1)  # Lock it
                return True
            time.sleep(0.001)  # Small delay before retry
        return False

    def _release_lock(self, slot_idx: int):
        """Release lock for a slot."""
        offset = self._get_slot_offset(slot_idx)
        lock_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE + 8
        struct.pack_into("I", self.buffer.data, lock_offset, 0)  # Unlock

    def _read_slot(self, slot_idx: int) -> Dict:
        """Read task info from a slot."""
        offset = self._get_slot_offset(slot_idx)

        # Read task_id
        task_id_bytes = bytes(self.buffer[offset : offset + TASK_ID_SIZE])
        task_id = task_id_bytes.decode("utf-8").rstrip("\x00")

        # Read task_name
        name_offset = offset + TASK_ID_SIZE
        task_name_bytes = bytes(self.buffer[name_offset : name_offset + TASK_NAME_SIZE])
        task_name = task_name_bytes.decode("utf-8").rstrip("\x00")

        # Read env_cfg
        cfg_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE
        env_cfg_bytes = bytes(self.buffer[cfg_offset : cfg_offset + ENV_CFG_SIZE])
        # Find the actual length of pickled data
        env_cfg_len = len(env_cfg_bytes.rstrip(b"\x00"))
        if env_cfg_len > 0:
            env_cfg = pickle.loads(env_cfg_bytes[:env_cfg_len])
        else:
            env_cfg = None

        # Read num_completed
        num_completed_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE
        num_completed = struct.unpack_from("I", self.buffer.data, num_completed_offset)[0]

        # Read mean_score
        mean_score_offset = num_completed_offset + 4
        mean_score = struct.unpack_from("f", self.buffer.data, mean_score_offset)[0]

        return {
            "task_id": task_id,
            "task_name": task_name,
            "env_cfg": env_cfg,
            "num_completed": num_completed,
            "mean_score": mean_score,
        }

    def get_task(self) -> Task:
        """Get a random task from shared memory.

        Tasks are pre-allocated by the server, so this method returns immediately
        without waiting for task creation. Multiple clients can read tasks concurrently
        since no locks are used for reading.
        """
        # Try multiple times to get a task
        max_attempts = 10
        empty_slots = 0
        invalid_slots = 0

        for _ in range(max_attempts):
            # Pick a random slot
            slot_idx = random.randint(0, self.num_slots - 1)

            # Read without acquiring lock
            slot_data = self._read_slot(slot_idx)

            # Make sure we have valid task data
            if slot_data["task_id"] and slot_data["env_cfg"] is not None:
                return RemoteTask(slot_data["task_id"], slot_data["task_name"], slot_data["env_cfg"], self, slot_idx)
            elif not slot_data["task_id"]:
                empty_slots += 1
            else:
                invalid_slots += 1

            # Small delay before retry
            time.sleep(0.1)  # Increased from 0.01 to give server more time

        # Before failing, check one more time with detailed diagnostics
        logger.warning("Failed to find valid task. Checking all slots for diagnostics...")
        valid_count = 0
        for i in range(min(10, self.num_slots)):  # Check first 10 slots
            slot_data = self._read_slot(i)
            if slot_data["task_id"]:
                valid_count += 1
                logger.debug(
                    f"Slot {i}: task_id={slot_data['task_id'][:8]}..., has_cfg={slot_data['env_cfg'] is not None}"
                )
            else:
                logger.debug(f"Slot {i}: EMPTY")

        logger.warning(f"Found {valid_count}/{min(10, self.num_slots)} valid slots in first 10 slots")

        raise RuntimeError(
            f"Failed to get task after {max_attempts} attempts. "
            f"Empty slots: {empty_slots}, Invalid slots: {invalid_slots}. "
            f"The curriculum server may not have finished initializing."
        )

    def complete_task(self, slot_idx: int, score: float) -> bool:
        """Update task completion in shared memory."""
        if not self._acquire_lock(slot_idx, timeout=self.timeout):
            logger.error(f"Failed to acquire lock for slot {slot_idx}")
            return False

        try:
            offset = self._get_slot_offset(slot_idx)

            # Read current values
            num_completed_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE
            num_completed = struct.unpack_from("I", self.buffer.data, num_completed_offset)[0]

            mean_score_offset = num_completed_offset + 4
            current_mean = struct.unpack_from("f", self.buffer.data, mean_score_offset)[0]

            # Update values using incremental mean calculation
            new_num_completed = num_completed + 1
            new_mean = ((current_mean * num_completed) + score) / new_num_completed

            # Write updated values
            struct.pack_into("I", self.buffer.data, num_completed_offset, new_num_completed)
            struct.pack_into("f", self.buffer.data, mean_score_offset, new_mean)

            logger.debug(f"Updated slot {slot_idx}: completions={new_num_completed}, mean_score={new_mean:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return False
        finally:
            self._release_lock(slot_idx)

    def stats(self) -> Dict[str, Any]:
        """Get statistics from shared memory."""
        stats = {"total_completions": 0, "active_tasks": 0, "slot_utilization": []}

        for slot_idx in range(self.num_slots):
            # Read without acquiring lock
            slot_data = self._read_slot(slot_idx)
            if slot_data["task_id"]:
                stats["active_tasks"] += 1
                stats["total_completions"] += slot_data["num_completed"]
                stats["slot_utilization"].append(
                    {
                        "slot": slot_idx,
                        "task_name": slot_data["task_name"],
                        "completions": slot_data["num_completed"],
                        "mean_score": slot_data["mean_score"],
                    }
                )

        return stats

    def __del__(self):
        """Clean up shared memory connection."""
        if hasattr(self, "shm"):
            self.shm.close()

    # These methods are for API compatibility but don't apply to shared memory version
    def completed_tasks(self) -> list[str]:
        """Not implemented for shared memory version."""
        return []

    def get_completion_rates(self) -> Dict[str, float]:
        """Not implemented for shared memory version."""
        return {}

    def get_task_probs(self) -> Dict[str, float]:
        """Not implemented for shared memory version."""
        return {}

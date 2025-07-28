import logging
import pickle
import struct
import threading
import time
import uuid
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)

# Shared memory layout constants
TASK_ID_SIZE = 36  # UUID string length
TASK_NAME_SIZE = 256
ENV_CFG_SIZE = 65536  # 64KB for pickled env config
SLOT_SIZE = (
    TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE + 4 + 4 + 4
)  # +4 for num_completed, +4 for mean_score, +4 for lock
HEADER_SIZE = 4  # 4 bytes for ready flag at the beginning of shared memory


class CurriculumServer:
    """Server that manages curriculum tasks using shared memory.

    Pre-allocates a fixed number of task slots in shared memory that clients can
    access immediately without waiting. Tasks are automatically refreshed when completed.
    """

    def __init__(self, curriculum: Curriculum, num_slots: int = 100, buffer_size: int = 4096, auto_start: bool = True):
        self.curriculum = curriculum
        self.num_slots = num_slots
        self.buffer_size = buffer_size
        self.tasks_created = 0  # Track total tasks created

        # Calculate total shared memory size
        self.total_size = HEADER_SIZE + (SLOT_SIZE * num_slots)

        # Try to unlink any existing shared memory with the same name
        try:
            existing_shm = shared_memory.SharedMemory(name="curriculum_server")
            existing_shm.close()
            existing_shm.unlink()
            logger.info("Cleaned up existing shared memory")
        except FileNotFoundError:
            pass  # No existing shared memory, which is fine
        except Exception as e:
            logger.warning(f"Error cleaning up existing shared memory: {e}")

        # Create shared memory
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name="curriculum_server")
            logger.info(
                f"Created shared memory '{self.shm.name}' with {num_slots} slots (size: {self.total_size} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

        # Create numpy arrays for easier access to shared memory
        self.buffer = np.ndarray((self.total_size,), dtype=np.uint8, buffer=self.shm.buf)

        # Initialize ready flag to 0 (not ready)
        struct.pack_into("I", self.buffer.data, 0, 0)

        # Pre-allocate all slots with tasks before starting monitor
        self._initialize_slots()

        # Set ready flag to 1 after all tasks are allocated
        struct.pack_into("I", self.buffer.data, 0, 1)
        logger.info("Set ready flag - clients can now connect")

        # Mark server as ready
        self._ready = True

        # Background thread for monitoring and refreshing tasks
        self.running = False
        self.monitor_thread = None

        if auto_start:
            self.start()

    def _get_slot_offset(self, slot_idx: int) -> int:
        """Get the byte offset for a specific slot."""
        return HEADER_SIZE + (slot_idx * SLOT_SIZE)

    def _write_slot(self, slot_idx: int, task: Task):
        """Write a task to a specific slot."""
        offset = self._get_slot_offset(slot_idx)

        # Generate task info
        task_id = str(uuid.uuid4())
        task_name = task.name()[: TASK_NAME_SIZE - 1]  # Truncate if needed
        env_cfg_pickled = pickle.dumps(task.env_cfg())

        if len(env_cfg_pickled) > ENV_CFG_SIZE:
            raise ValueError(f"Pickled env_cfg too large: {len(env_cfg_pickled)} > {ENV_CFG_SIZE}")

        # Write task_id (36 bytes)
        task_id_bytes = task_id.encode("utf-8")
        self.buffer[offset : offset + TASK_ID_SIZE] = 0  # Clear first
        self.buffer[offset : offset + len(task_id_bytes)] = np.frombuffer(task_id_bytes, dtype=np.uint8)

        # Write task_name (256 bytes)
        task_name_bytes = task_name.encode("utf-8")
        name_offset = offset + TASK_ID_SIZE
        self.buffer[name_offset : name_offset + TASK_NAME_SIZE] = 0  # Clear first
        self.buffer[name_offset : name_offset + len(task_name_bytes)] = np.frombuffer(task_name_bytes, dtype=np.uint8)

        # Write env_cfg (65536 bytes)
        cfg_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE
        self.buffer[cfg_offset : cfg_offset + ENV_CFG_SIZE] = 0  # Clear first
        self.buffer[cfg_offset : cfg_offset + len(env_cfg_pickled)] = np.frombuffer(env_cfg_pickled, dtype=np.uint8)

        # Write num_completed (4 bytes) - initialized to 0
        num_completed_offset = offset + TASK_ID_SIZE + TASK_NAME_SIZE + ENV_CFG_SIZE
        struct.pack_into("I", self.buffer.data, num_completed_offset, 0)

        # Write mean_score (4 bytes) - initialized to 0.0
        mean_score_offset = num_completed_offset + 4
        struct.pack_into("f", self.buffer.data, mean_score_offset, 0.0)

        # Write lock (4 bytes) - initialized to 0 (unlocked)
        lock_offset = mean_score_offset + 4
        struct.pack_into("I", self.buffer.data, lock_offset, 0)

        # Store task reference for completion
        self._active_tasks[slot_idx] = task

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
        # Find the actual length of pickled data (look for trailing zeros)
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
            "slot_idx": slot_idx,
        }

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

    def _initialize_slots(self):
        """Pre-allocate all task slots with tasks from the curriculum."""
        self._active_tasks = {}
        logger.info(f"Pre-allocating {self.num_slots} tasks in shared memory...")

        start_time = time.time()
        for i in range(self.num_slots):
            task = self.curriculum.get_task()
            self._write_slot(i, task)
            self.tasks_created += 1

            # Log progress for large numbers of slots
            if (i + 1) % 10 == 0 or (i + 1) == self.num_slots:
                logger.debug(f"Initialized {i + 1}/{self.num_slots} task slots")

        elapsed = time.time() - start_time
        logger.info(f"Successfully pre-allocated {self.num_slots} tasks in {elapsed:.2f}s")
        logger.info("Curriculum server ready - clients can now connect and start working immediately")

    def _monitor_tasks(self):
        """Monitor tasks and refresh those with > 5 completions."""
        while self.running:
            try:
                for slot_idx in range(self.num_slots):
                    if self._acquire_lock(slot_idx):
                        try:
                            slot_data = self._read_slot(slot_idx)

                            # Check if task has more than 5 completions
                            if slot_data["num_completed"] > 1:
                                # Complete the task in the curriculum
                                task = self._active_tasks.get(slot_idx)
                                if task:
                                    task.complete(slot_data["mean_score"])
                                    logger.debug(
                                        f"Completed task {slot_data['task_name']} with mean score"
                                        f"{slot_data['mean_score']}"
                                    )

                                # Get a new task and write it to the slot
                                new_task = self.curriculum.get_task()
                                self._write_slot(slot_idx, new_task)
                                self.tasks_created += 1
                                logger.debug(f"Refreshed slot {slot_idx} with new task {new_task.name()}")
                        finally:
                            self._release_lock(slot_idx)

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")
                time.sleep(1.0)

    def start(self):
        """Start the monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_tasks, daemon=True)
            self.monitor_thread.start()
            logger.info("Curriculum server monitor thread started")

    def stop(self):
        """Stop the server and clean up shared memory."""
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)

        # Clean up shared memory
        self.shm.close()
        self.shm.unlink()
        logger.info("Curriculum server stopped and shared memory cleaned up")

    def is_running(self):
        """Check if the server is running."""
        return self.running and self.monitor_thread is not None and self.monitor_thread.is_alive()

    def is_ready(self) -> bool:
        """Check if the server has finished pre-allocating tasks."""
        return hasattr(self, "_ready") and self._ready

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait until server is ready with pre-allocated tasks."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                return True
            time.sleep(0.1)
        return False

    def stats(self) -> Dict:
        """Get curriculum statistics."""
        stats = self.curriculum.stats()

        # Add shared memory stats
        total_completions = 0
        active_tasks = 0

        for slot_idx in range(self.num_slots):
            if self._acquire_lock(slot_idx, timeout=0.1):
                try:
                    slot_data = self._read_slot(slot_idx)
                    if slot_data["task_id"]:
                        active_tasks += 1
                        total_completions += slot_data["num_completed"]
                finally:
                    self._release_lock(slot_idx)

        stats["active_tasks"] = active_tasks
        stats["total_completions"] = total_completions
        stats["num_slots"] = self.num_slots
        stats["tasks_created"] = self.tasks_created

        return stats


def run_curriculum_server(curriculum: Curriculum, num_slots: int = 100):
    """Convenience function to create and run a curriculum server."""
    server = CurriculumServer(curriculum, num_slots=num_slots)
    # The server auto-starts in __init__, so we just need to keep it running
    # This function blocks to maintain backward compatibility
    try:
        if server.monitor_thread is not None:
            server.monitor_thread.join()
    except KeyboardInterrupt:
        server.stop()

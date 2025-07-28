import logging
import random
import time
from typing import Any, Dict

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_state import CurriculumState

logger = logging.getLogger(__name__)


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
        self._start_time = time.time()  # Track when task was acquired

    def complete(self, score: float):
        """Complete the task by updating shared memory."""
        if self._is_complete:
            self._client._log(f"Task {self._id} is already complete", level="warning")
            return

        # Convert score to Python float in case it's a numpy type
        score = float(score)

        # Calculate task completion time
        completion_time = time.time() - self._start_time

        # Update shared memory
        success = self._client.complete_task(self._slot_idx, score)
        if success:
            self._is_complete = True
            self._client._log(
                f"Task {self._name} completed with score {score:.3f} in {completion_time:.2f}s", level="debug"
            )
            # Update client's completion time tracking
            self._client._track_completion_time(completion_time)
        else:
            self._client._log(f"Failed to complete task {self._name} in shared memory", level="error")

    def short_name(self) -> str:
        return self._name.split("/")[-1]


class CurriculumClient(Curriculum):
    """Client for accessing curriculum tasks from shared memory.

    This client connects to a pre-allocated task pool managed by CurriculumServer.
    Tasks are read directly from shared memory slots without locks, allowing fast
    parallel access. The server continuously refreshes tasks as they are completed.

    The shared memory layout consists of a fixed number of slots, each containing
    a task ID, name, and pickled environment configuration. Clients read tasks
    randomly from available slots, ensuring even distribution of work across
    parallel workers.

    Note: This implementation uses the base class's default implementations for
    methods like completed_tasks() and get_completion_rates() which return empty
    values, as these would require synchronization across processes.

    Example:
        client = CurriculumClient()
        task = client.get_task()  # Returns immediately with pre-allocated task
        # ... run simulation ...
        task.complete(score)  # Updates shared memory directly
    """

    def __init__(self, name: str = "curriculum_server", timeout: float = 30.0):
        """Initialize the curriculum client.

        Args:
            name: Name of the shared memory segment to connect to.
            timeout: Timeout in seconds for connecting to the server.
        """
        # Generate a random client ID
        self.client_id = random.randint(0, 999999)
        self._should_log = (self.client_id % 100) == 0

        # Track timing statistics
        self._task_acquisition_times = []
        self._task_completion_times = []
        self._tasks_acquired = 0
        self._tasks_completed = 0

        self._log("Initializing curriculum client", level="info")

        # Connect to CurriculumState using the new connect method
        start_time = time.time()
        self.state = CurriculumState.connect(name=name, timeout=timeout, wait_ready=True)
        connection_time = time.time() - start_time

        # Get initial number of active slots
        self.num_active_slots = self.state.get_num_active_slots()

        self._log(
            f"Connected to curriculum server with {self.num_active_slots} active slots in {connection_time:.2f}s",
            level="debug",
        )

        # Verify at least some slots have tasks
        self._verify_connection()

    def _log(self, message: str, level: str = "info"):
        """Log a message with client ID prefix, only if this client should log."""
        if not self._should_log:
            return

        prefixed_message = f"[Client {self.client_id}] {message}"

        if level == "debug":
            logger.debug(prefixed_message)
        elif level == "info":
            logger.info(prefixed_message)
        elif level == "warning":
            logger.warning(prefixed_message)
        elif level == "error":
            logger.error(prefixed_message)
        else:
            logger.info(prefixed_message)

    def _track_completion_time(self, completion_time: float):
        """Track task completion time statistics."""
        self._task_completion_times.append(completion_time)
        self._tasks_completed += 1

        # Keep only last 100 completion times to avoid memory growth
        if len(self._task_completion_times) > 100:
            self._task_completion_times.pop(0)

        # Log statistics every 10 completions
        if self._tasks_completed % 10 == 0:
            mean_acquisition = (
                sum(self._task_acquisition_times) / len(self._task_acquisition_times)
                if self._task_acquisition_times
                else 0
            )
            mean_completion = (
                sum(self._task_completion_times) / len(self._task_completion_times)
                if self._task_completion_times
                else 0
            )
            self._log(
                f"Stats - Tasks: {self._tasks_acquired} acquired, {self._tasks_completed} completed | "
                f"Mean times: {mean_acquisition * 1000:.1f}ms to acquire, {mean_completion:.2f}s to complete",
                level="info",
            )

    def _verify_connection(self):
        """Verify that the shared memory contains valid tasks."""
        # Update our knowledge of active slots
        self.num_active_slots = self.state.get_num_active_slots()

        valid_count = 0
        for i in range(min(5, self.num_active_slots)):  # Check first 5 slots
            try:
                slot_data = self.state.read_task(i)
                if slot_data["task_id"]:
                    valid_count += 1
            except Exception as e:
                self._log(f"Error reading slot {i} during verification: {e}", level="debug")

        if valid_count == 0:
            self._log(
                f"No valid tasks found in first {min(5, self.num_active_slots)} "
                f"slots. Server may still be initializing. "
                f"Active slots: {self.num_active_slots}, Max slots: {self.state.max_slots}",
                level="warning",
            )

    def get_task(self) -> Task:
        """Get a task from shared memory by sampling multiple slots and selecting the best one.

        Samples 5 random slots (or all available if fewer) and selects the task with:
        1. The least number of outstanding clients
        2. If tied, the least number of completions (to ensure balanced sampling)

        This approach helps distribute work evenly across tasks and prevents task starvation.
        """
        start_time = time.time()

        # Update our knowledge of active slots
        self.num_active_slots = self.state.get_num_active_slots()

        if self.num_active_slots == 0:
            self._log("No active slots available", level="warning")
            raise RuntimeError("No active slots available in curriculum server")

        # Number of slots to sample
        sample_size = min(5, self.num_active_slots)

        # Sample random slots without replacement
        if self.num_active_slots <= sample_size:
            # If we have 5 or fewer slots, just use all of them
            sampled_slots = list(range(self.num_active_slots))
        else:
            # Sample 5 random slots
            sampled_slots = random.sample(range(self.num_active_slots), sample_size)

        # Collect valid tasks from sampled slots
        valid_tasks = []
        empty_slots = 0
        invalid_slots = 0

        for slot_idx in sampled_slots:
            # Read without acquiring lock using CurriculumState
            slot_data = self.state.read_task(slot_idx)

            # Check if we have valid task data
            if slot_data["task_id"] and slot_data["env_cfg"] is not None:
                valid_tasks.append((slot_idx, slot_data))
            elif not slot_data["task_id"]:
                empty_slots += 1
            else:
                invalid_slots += 1

        # If no valid tasks found, try more attempts with random sampling
        max_attempts = 10
        if not valid_tasks:
            for _ in range(max_attempts):
                slot_idx = random.randint(0, self.num_active_slots - 1)
                slot_data = self.state.read_task(slot_idx)

                if slot_data["task_id"] and slot_data["env_cfg"] is not None:
                    valid_tasks.append((slot_idx, slot_data))
                    break
                elif not slot_data["task_id"]:
                    empty_slots += 1
                else:
                    invalid_slots += 1

        # If still no valid tasks, raise error
        if not valid_tasks:
            acquisition_time = time.time() - start_time
            self._log(
                f"Failed to find valid task after sampling {sample_size} slots and {max_attempts} additional attempts "
                f"in {acquisition_time:.2f}s. Empty slots: {empty_slots}, Invalid slots: {invalid_slots}",
                level="error",
            )
            raise RuntimeError(
                f"Failed to find valid task. Empty slots: {empty_slots}, Invalid slots: {invalid_slots}. "
                f"The curriculum server may not have finished initializing."
            )

        # Select the best task based on:
        # 1. Least outstanding (primary criterion)
        # 2. Least completed (tie breaker)
        best_slot_idx, best_slot_data = min(valid_tasks, key=lambda x: (x[1]["num_outstanding"], x[1]["num_completed"]))

        # Increment outstanding count when task is sampled
        self.state.increment_outstanding(best_slot_idx)

        # Track acquisition time
        acquisition_time = time.time() - start_time
        self._task_acquisition_times.append(acquisition_time)
        self._tasks_acquired += 1

        # Keep only last 100 acquisition times
        if len(self._task_acquisition_times) > 100:
            self._task_acquisition_times.pop(0)

        self._log(
            f"Acquired task '{best_slot_data['task_name']}' from slot {best_slot_idx} "
            f"(outstanding: {best_slot_data['num_outstanding']}, completed: {best_slot_data['num_completed']}) "
            f"after sampling {len(valid_tasks)} valid tasks in {acquisition_time * 1000:.1f}ms",
            level="debug",
        )

        return RemoteTask(
            best_slot_data["task_id"], best_slot_data["task_name"], best_slot_data["env_cfg"], self, best_slot_idx
        )

    def complete_task(self, slot_idx: int, score: float) -> bool:
        """Update task completion in shared memory."""
        # Get task ID and call CurriculumState's complete_task
        slot_data = self.state.read_task(slot_idx)
        return self.state.complete_task(slot_data["task_id"], score)

    def stats(self) -> Dict[str, Any]:
        """Get statistics from shared memory."""
        stats = {"total_completions": 0, "active_tasks": 0, "slot_utilization": []}

        for slot_idx in range(self.num_active_slots):
            # Read without acquiring lock
            slot_data = self.state.read_task(slot_idx)
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

        # Add client-specific stats
        stats["client_stats"] = {
            "client_id": self.client_id,
            "tasks_acquired": self._tasks_acquired,
            "tasks_completed": self._tasks_completed,
            "mean_acquisition_time_ms": sum(self._task_acquisition_times) / len(self._task_acquisition_times) * 1000
            if self._task_acquisition_times
            else 0,
            "mean_completion_time_sec": sum(self._task_completion_times) / len(self._task_completion_times)
            if self._task_completion_times
            else 0,
        }

        return stats

    def __del__(self):
        """Clean up shared memory connection."""
        if hasattr(self, "state"):
            self.state.close()
            if hasattr(self, "_should_log") and self._should_log:
                self._log(
                    f"Closing client - Acquired {self._tasks_acquired} tasks, completed {self._tasks_completed}",
                    level="info",
                )

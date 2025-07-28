import logging
import threading
import time
import uuid
from typing import Dict

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_state import CurriculumState

logger = logging.getLogger(__name__)


class CurriculumServer:
    """Server that manages curriculum tasks using shared memory.

    Pre-allocates a fixed number of task slots in shared memory that clients can
    access immediately without waiting. Tasks are automatically refreshed when completed.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        num_slots: int = 100,
        buffer_size: int = 4096,
        auto_start: bool = True,
        status_interval: float = 30.0,
    ):
        self.curriculum = curriculum
        self.num_slots = num_slots
        self.buffer_size = buffer_size
        self.status_interval = status_interval
        self.tasks_created = 0  # Track total tasks created

        # Create the shared memory state manager
        self.state = CurriculumState(num_slots=num_slots, name="curriculum_server", create=True)

        # Pre-allocate all slots with tasks before starting monitor
        self._initialize_slots()

        # Set ready flag after all tasks are allocated
        self.state.set_ready(True)
        logger.info("Set ready flag - clients can now connect")

        # Mark server as ready
        self._ready = True

        # Background threads
        self.running = False
        self.monitor_thread = None
        self.status_thread = None

        if auto_start:
            self.start()

    def _write_slot(self, slot_idx: int, task: Task):
        """Write a task to a specific slot."""
        # Generate task info
        task_id = str(uuid.uuid4())
        task_name = task.name()
        env_cfg = task.env_cfg()

        # Write to shared memory using the state manager
        self.state.write_task(
            slot_idx=slot_idx, task_id=task_id, task_name=task_name, env_cfg=env_cfg, num_completed=0, mean_score=0.0
        )

        # Store task reference for completion
        self._active_tasks[slot_idx] = task

    def _read_slot(self, slot_idx: int) -> Dict:
        """Read task info from a slot."""
        return self.state.read_task(slot_idx)

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

    def _print_status(self):
        """Print detailed status of the task store."""
        try:
            # Gather statistics
            total_completions = 0
            total_outstanding = 0
            active_tasks = 0
            task_stats = []

            for slot_idx in range(self.num_slots):
                slot_data = self._read_slot(slot_idx)
                if slot_data["task_id"]:
                    active_tasks += 1
                    total_completions += slot_data["num_completed"]
                    total_outstanding += slot_data["num_outstanding"]

                    # Collect task info for detailed reporting
                    task_stats.append(
                        {
                            "slot": slot_idx,
                            "name": slot_data["task_name"],
                            "completed": slot_data["num_completed"],
                            "outstanding": slot_data["num_outstanding"],
                            "mean_score": slot_data["mean_score"],
                        }
                    )

            # Sort by number of completions (descending)
            task_stats.sort(key=lambda x: x["completed"], reverse=True)

            # Print status header
            print("\n" + "=" * 80)
            print(f"CURRICULUM SERVER STATUS - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            # Print summary statistics
            print("\nSUMMARY:")
            print(f"  Total slots:          {self.num_slots}")
            print(f"  Active tasks:         {active_tasks}")
            print(f"  Tasks created:        {self.tasks_created}")
            print(f"  Total completions:    {total_completions}")
            print(f"  Outstanding work:     {total_outstanding}")
            print(
                f"  Avg completions/task: {total_completions / active_tasks:.2f}"
                if active_tasks > 0
                else "  Avg completions/task: 0.00"
            )

            # Print top tasks by completions
            print("\nTOP 10 TASKS BY COMPLETIONS:")
            print(f"{'Slot':<6} {'Task Name':<40} {'Completed':<12} {'Outstanding':<12} {'Mean Score':<10}")
            print("-" * 80)

            for task in task_stats[:10]:
                print(
                    f"{task['slot']:<6} {task['name'][:39]:<40} {task['completed']:<12} "
                    f"{task['outstanding']:<12} {task['mean_score']:<10.2f}"
                )

            # Print tasks with no completions (if any)
            no_completion_tasks = [t for t in task_stats if t["completed"] == 0]
            if no_completion_tasks:
                print(f"\nTASKS WITH NO COMPLETIONS: {len(no_completion_tasks)}")
                for task in no_completion_tasks[:5]:  # Show first 5
                    print(f"  Slot {task['slot']}: {task['name']}")
                if len(no_completion_tasks) > 5:
                    print(f"  ... and {len(no_completion_tasks) - 5} more")

            # Print curriculum-specific stats if available
            try:
                curriculum_stats = self.curriculum.stats()
                if curriculum_stats:
                    print("\nCURRICULUM STATS:")
                    for key, value in curriculum_stats.items():
                        if key not in ["active_tasks", "total_completions", "num_slots", "tasks_created"]:
                            print(f"  {key}: {value}")
            except Exception:
                pass  # Curriculum might not have stats() method

            print("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Error printing status: {e}")

    def _status_monitor(self):
        """Background thread that prints status periodically."""
        while self.running:
            try:
                # Sleep for the status interval
                time.sleep(self.status_interval)

                if self.running:  # Check again after sleep
                    self._print_status()

            except Exception as e:
                logger.error(f"Error in status monitor thread: {e}")

    def _monitor_tasks(self):
        """Monitor tasks and refresh those with > 5 completions and no outstanding work."""
        while self.running:
            try:
                # Get all tasks that can be completed (no outstanding work)
                completable_tasks = self.state.get_completable_tasks()

                for task_data in completable_tasks:
                    # Check if task has more than 5 completions
                    if task_data["num_completed"] > 5:
                        slot_idx = task_data["slot_idx"]

                        # Complete the task in the curriculum
                        task = self._active_tasks.get(slot_idx)
                        if task:
                            task.complete(task_data["mean_score"])
                            logger.debug(
                                f"Completed task {task_data['task_name']} with mean score {task_data['mean_score']:.2f}"
                            )

                        # Get a new task and write it to the slot
                        new_task = self.curriculum.get_task()
                        self._write_slot(slot_idx, new_task)
                        self.tasks_created += 1
                        logger.debug(f"Refreshed slot {slot_idx} with new task {new_task.name()}")

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")
                time.sleep(1.0)

    def start(self):
        """Start the monitoring and status threads."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_tasks, daemon=True)
            self.monitor_thread.start()
            logger.info("Curriculum server monitor thread started")

        if self.status_thread is None or not self.status_thread.is_alive():
            self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
            self.status_thread.start()
            logger.info(f"Status printing thread started (interval: {self.status_interval}s)")

            # Print initial status immediately
            self._print_status()

    def stop(self):
        """Stop the server and clean up shared memory."""
        self.running = False

        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)

        if self.status_thread is not None:
            self.status_thread.join(timeout=2.0)

        # Clean up shared memory
        self.state.cleanup()
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
            slot_data = self._read_slot(slot_idx)
            if slot_data["task_id"]:
                active_tasks += 1
                total_completions += slot_data["num_completed"]

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

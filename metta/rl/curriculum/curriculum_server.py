import logging
import threading
import time
import uuid
from typing import Dict

from metta.mettagrid.curriculum.core import Curriculum
from metta.rl.curriculum.curriculum_state import CurriculumState

logger = logging.getLogger(__name__)


class CurriculumServer:
    """Server that manages curriculum tasks using shared memory.

    Dynamically allocates task slots as needed. Starts with no tasks and adds
    them as clients request work, ensuring all tasks have at least one outstanding
    client before adding new ones.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        max_slots: int = 10000,
        initial_tasks: int = 10,
        auto_start: bool = True,
        status_interval: float = 30.0,
    ):
        """Initialize the curriculum server.

        Args:
            curriculum: The curriculum to serve tasks from
            max_slots: Maximum number of task slots that can be allocated
            initial_tasks: Number of tasks to start with
            auto_start: Whether to automatically start the server threads
            status_interval: Interval in seconds between status prints
        """
        self.curriculum = curriculum
        self.max_slots = max_slots
        self.initial_tasks = initial_tasks
        self.status_interval = status_interval
        self.tasks_created = 0  # Track total tasks created
        self.tasks_completed = 0  # Track total tasks completed (replaced)
        self.task_completion_times = []  # Track time to complete each task
        self.task_start_times = {}  # Track when each task was created

        # Create the shared memory state manager
        self.state = CurriculumState(max_slots=max_slots, name="curriculum_server", create=True)

        # Store active tasks by slot
        self._active_tasks = {}

        # Add initial tasks
        self._add_initial_tasks()

        # Set ready flag after initial tasks are added
        self.state.set_ready(True)
        logger.info("Set ready flag - clients can now connect")

        # Background threads
        self.running = False
        self.monitor_thread = None
        self.status_thread = None

        if auto_start:
            self.start()

    def _add_initial_tasks(self):
        """Add the initial set of tasks."""
        logger.info(f"Adding {self.initial_tasks} initial tasks...")

        for i in range(self.initial_tasks):
            task = self.curriculum.get_task()
            task_id = str(uuid.uuid4())
            task_name = task.name()
            env_cfg = task.env_cfg()

            slot_idx = self.state.add_task(task_id, task_name, env_cfg)
            if slot_idx is not None:
                self._active_tasks[slot_idx] = task
                self.task_start_times[slot_idx] = time.time()
                self.tasks_created += 1
            else:
                logger.warning(f"Failed to add initial task {i}")
                break

        logger.info(f"Successfully added {len(self._active_tasks)} initial tasks")

    def _add_new_task_if_needed(self):
        """Check if a new task should be added and add it if needed.

        A new task is added when all existing tasks have outstanding work.
        """
        # Check if we need to add a new task
        num_active = self.state.get_num_active_slots()

        if num_active >= self.max_slots:
            return  # Already at maximum capacity

        # Check if all tasks have outstanding work
        all_have_outstanding = True
        for slot_idx in range(num_active):
            try:
                task_data = self.state.read_task(slot_idx)
                if task_data.get("task_id") and task_data["num_outstanding"] == 0:
                    all_have_outstanding = False
                    break
            except Exception:
                pass  # Skip invalid slots

        if all_have_outstanding and num_active > 0:
            # All tasks have outstanding work, add a new one
            task = self.curriculum.get_task()
            task_id = str(uuid.uuid4())
            task_name = task.name()
            env_cfg = task.env_cfg()

            slot_idx = self.state.add_task(task_id, task_name, env_cfg)
            if slot_idx is not None:
                self._active_tasks[slot_idx] = task
                self.task_start_times[slot_idx] = time.time()
                self.tasks_created += 1
                logger.info(f"Added new task '{task_name}' to slot {slot_idx} (now {num_active + 1} active tasks)")

    def _print_status(self):
        """Print detailed status of the task store."""
        try:
            # Gather statistics
            total_completions = 0
            total_outstanding = 0
            active_tasks = 0
            task_stats = []
            num_active_slots = self.state.get_num_active_slots()

            for slot_idx in range(num_active_slots):
                try:
                    slot_data = self.state.read_task(slot_idx)
                    if slot_data.get("task_id"):
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
                except Exception:
                    pass  # Skip invalid slots

            # Sort by number of completions (descending)
            task_stats.sort(key=lambda x: x["completed"], reverse=True)

            # Calculate mean completion time
            mean_completion_time = (
                sum(self.task_completion_times) / len(self.task_completion_times) if self.task_completion_times else 0
            )
            mean_completions_per_task = self.tasks_completed / self.tasks_created if self.tasks_created > 0 else 0

            # Print status header
            print("\n" + "=" * 80)
            print(f"CURRICULUM SERVER STATUS - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            # Print summary statistics
            print("\nSUMMARY:")
            print(f"  Active slots:            {num_active_slots}")
            print(f"  Max slots:               {self.max_slots}")
            print(f"  Active tasks:            {active_tasks}")
            print(f"  Tasks created:           {self.tasks_created}")
            print(f"  Tasks completed:         {self.tasks_completed}")
            print(f"  Total completions:       {total_completions}")
            print(f"  Outstanding work:        {total_outstanding}")
            print(
                f"  Avg completions/task:    {total_completions / active_tasks:.2f}"
                if active_tasks > 0
                else "  Avg completions/task:    0.00"
            )
            print(f"  Mean time to complete:   {mean_completion_time:.2f}s")
            print(f"  Mean completions/task:   {mean_completions_per_task:.2f}")

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
                        if key not in [
                            "active_tasks",
                            "total_completions",
                            "active_slots",
                            "max_slots",
                            "tasks_created",
                        ]:
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
        """Monitor tasks and manage the task pool.

        - Refresh tasks that have no outstanding work
        - Add new tasks when all existing tasks have outstanding work
        """
        while self.running:
            try:
                # Check if we need to add a new task
                self._add_new_task_if_needed()

                # Get all tasks that can be completed (no outstanding work)
                completable_tasks = self.state.get_completable_tasks()

                for task_data in completable_tasks:
                    slot_idx = task_data["slot_idx"]

                    # Calculate completion time if this task had any completions
                    if task_data["num_completed"] > 0:
                        if slot_idx in self.task_start_times:
                            completion_time = time.time() - self.task_start_times[slot_idx]
                            self.task_completion_times.append(completion_time)
                            # Keep only last 1000 completion times to avoid memory growth
                            if len(self.task_completion_times) > 1000:
                                self.task_completion_times.pop(0)

                        self.tasks_completed += 1

                        # Complete the task in the curriculum
                        task = self._active_tasks.get(slot_idx)
                        if task:
                            task.complete(task_data["mean_score"])
                            logger.debug(
                                f"Completed task {task_data['task_name']} "
                                f"with mean score {task_data['mean_score']:.2f}, "
                                f"{task_data['num_completed']} completions"
                            )

                    # Get a new task and write it to the slot
                    new_task = self.curriculum.get_task()
                    task_id = str(uuid.uuid4())
                    task_name = new_task.name()
                    env_cfg = new_task.env_cfg()

                    # Write the new task to the same slot
                    self.state.write_task(
                        slot_idx=slot_idx,
                        task_id=task_id,
                        task_name=task_name,
                        env_cfg=env_cfg,
                        num_completed=0,
                        mean_score=0.0,
                    )

                    self._active_tasks[slot_idx] = new_task
                    self.task_start_times[slot_idx] = time.time()
                    self.tasks_created += 1
                    logger.debug(f"Refreshed slot {slot_idx} with new task {task_name}")

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

    def stats(self) -> Dict:
        """Get curriculum statistics."""
        stats = self.curriculum.stats()

        # Add shared memory stats
        total_completions = 0
        active_tasks = 0
        num_active_slots = self.state.get_num_active_slots()

        for slot_idx in range(num_active_slots):
            try:
                slot_data = self.state.read_task(slot_idx)
                if slot_data.get("task_id"):
                    active_tasks += 1
                    total_completions += slot_data["num_completed"]
            except Exception:
                pass  # Skip invalid slots

        # Calculate mean statistics
        mean_completion_time = (
            sum(self.task_completion_times) / len(self.task_completion_times) if self.task_completion_times else 0
        )
        mean_completions_per_task = self.tasks_completed / self.tasks_created if self.tasks_created > 0 else 0

        stats["active_tasks"] = active_tasks
        stats["active_slots"] = num_active_slots
        stats["max_slots"] = self.max_slots
        stats["total_completions"] = total_completions
        stats["tasks_created"] = self.tasks_created
        stats["tasks_completed"] = self.tasks_completed
        stats["mean_completion_time_sec"] = mean_completion_time
        stats["mean_completions_per_task"] = mean_completions_per_task

        return stats

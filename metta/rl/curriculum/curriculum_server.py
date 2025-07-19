import logging
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Union

from flask import Flask, Response, jsonify, request
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from werkzeug.serving import make_server

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)

# Disable werkzeug route logging (only show errors)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class CurriculumServer:
    """HTTP server that serves curriculum tasks to distributed environments.

    Architecture:
    1. Generator thread: Continuously queries curriculum and maintains a task population queue.
       When the queue is full, it automatically evicts oldest tasks (FIFO).
    2. HTTP request handler thread pool: Samples tasks from the population,
       clones them, allocates unique IDs, and tracks them.
    3. Status printing thread: Monitors and reports server metrics.

    Args:
        curriculum: The curriculum to serve tasks from
        port: Port number to listen on (default: 12346)
        queue_size: Maximum size of the task population queue (default: 500)
        generation_batch_size: Number of tasks to generate at once (default: 100)
        generation_interval: Sleep interval in seconds between generation batches (default: 1.0)
        max_workers: Number of threads in the pool for handling requests (default: 16)
    """

    def __init__(
        self,
        curriculum: Curriculum,
        port: int = 12346,
        queue_size: int = 500,
        generation_batch_size: int = 100,
        generation_interval: float = 1.0,
        max_workers: int = 16,
    ):
        self._curriculum = curriculum
        self._host = "0.0.0.0"
        self._port = port
        self._app = Flask(__name__)
        self._assignment_lock = threading.Lock()
        self._server_thread = None
        self._server = None
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()

        # Thread pool for handling concurrent requests
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Task population queue with FIFO eviction
        self._queue_size = queue_size
        self._task_population = deque(maxlen=queue_size)  # Automatically evicts oldest when full
        self._population_lock = threading.Lock()
        self._generation_batch_size = generation_batch_size
        self._generation_interval = generation_interval
        self._generation_thread = None
        self._stop_generation = threading.Event()
        self._last_task = None  # Keep track of last task for cloning

        self._assigned_tasks = {}
        self._task_id_counter = 0

        # Set up routes
        self._app.add_url_rule("/tasks", "get_tasks", self._get_tasks, methods=["GET"])
        self._app.add_url_rule("/complete", "complete", self._complete_task, methods=["POST"])
        self._app.add_url_rule("/health", "health", self._health, methods=["GET"])

        # Metrics
        self._num_tasks_completed = 0
        self._num_tasks_assigned = 0
        self._num_tasks_generated = 0
        self._num_requests = 0
        self._start_time = time.time()

        # Track wait times (keep last 1000 for statistics)
        self._wait_times = deque(maxlen=1000)
        self._last_stats_print = 0

        # Track metrics at last report for rate calculations
        self._last_report_time = time.time()
        self._last_report_tasks_generated = 0
        self._last_report_tasks_assigned = 0
        self._last_report_tasks_completed = 0

    def _generate_tasks_loop(self):
        """Background thread that continuously generates tasks and adds them to the queue."""
        while not self._stop_generation.is_set():
            try:
                # Generate a batch of tasks
                new_tasks = []
                for _ in range(self._generation_batch_size):
                    if self._stop_generation.is_set():
                        break

                    try:
                        task = self._curriculum.get_task()
                        new_tasks.append(task)
                    except Exception as e:
                        logger.error(f"Error generating task: {e}")
                        break

                # Add tasks to population (deque automatically evicts oldest when full)
                if new_tasks:
                    with self._population_lock:
                        self._task_population.extend(new_tasks)
                        self._num_tasks_generated += len(new_tasks)

                    logger.debug(f"Generated {len(new_tasks)} tasks, population size now: {len(self._task_population)}")

                # Sleep between generation batches
                time.sleep(self._generation_interval)
            except Exception as e:
                logger.error(f"Error in task generation loop: {e}")
                time.sleep(1)

    def _monitor_stats(self):
        """Background thread that monitors and prints statistics."""
        while not self._stop_monitoring.is_set():
            time.sleep(1)  # Check every second

            with self._assignment_lock:
                if self._num_requests - self._last_stats_print >= 100:
                    self._print_stats()
                    self._last_stats_print = self._num_requests

    def _print_stats(self):
        """Print server statistics."""
        current_time = time.time()

        # Calculate time since last report
        time_since_last_report = current_time - self._last_report_time

        # Calculate rates since last report (recent rates)
        if time_since_last_report > 0:
            recent_tasks_generated_per_sec = (
                self._num_tasks_generated - self._last_report_tasks_generated
            ) / time_since_last_report
            recent_tasks_assigned_per_sec = (
                self._num_tasks_assigned - self._last_report_tasks_assigned
            ) / time_since_last_report
            recent_tasks_completed_per_sec = (
                self._num_tasks_completed - self._last_report_tasks_completed
            ) / time_since_last_report
        else:
            recent_tasks_generated_per_sec = 0
            recent_tasks_assigned_per_sec = 0
            recent_tasks_completed_per_sec = 0

        # Calculate wait time statistics
        if self._wait_times:
            mean_wait_time = sum(self._wait_times) / len(self._wait_times)
            max_wait_time = max(self._wait_times)
        else:
            mean_wait_time = 0
            max_wait_time = 0

        # Create a rich console and table
        console = Console()
        table = Table(
            title=f"[bold cyan]Curriculum Server Stats - {self._num_requests} requests[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Progress", style="green", justify="right")
        table.add_column("Rate", style="yellow", justify="left")

        # Add rows
        completion_rate = (
            (self._num_tasks_completed / self._num_tasks_assigned * 100) if self._num_tasks_assigned > 0 else 0
        )

        table.add_row(
            "Tasks Generated",
            f"{self._num_tasks_generated:,}",
            f"[dim]{recent_tasks_generated_per_sec:.1f} tasks/sec[/dim]",
        )

        table.add_row(
            "Tasks Assigned",
            f"{self._num_tasks_assigned:,}",
            f"[dim]{recent_tasks_assigned_per_sec:.1f} tasks/sec[/dim]",
        )

        table.add_row(
            "Tasks Completed",
            f"{self._num_tasks_completed:,} ({completion_rate:.1f}%)",
            f"[dim]{recent_tasks_completed_per_sec:.1f} tasks/sec[/dim]",
        )

        table.add_row(
            "Client Wait Time",
            f"Mean: {mean_wait_time * 1000:.1f}ms",
            f"[dim]Max: {max_wait_time * 1000:.1f}ms[/dim]",
        )

        table.add_row(
            "Outstanding Tasks",
            f"{len(self._assigned_tasks):,}",
            "",
        )

        with self._population_lock:
            pool_size = len(self._task_population)

        table.add_row(
            "Task Population Size",
            f"{pool_size:,}",
            f"[dim]Max: {self._queue_size:,}[/dim]",
        )

        # Log the table
        console.print(table)

        # Update last report values for next rate calculation
        self._last_report_time = current_time
        self._last_report_tasks_generated = self._num_tasks_generated
        self._last_report_tasks_assigned = self._num_tasks_assigned
        self._last_report_tasks_completed = self._num_tasks_completed

    def _get_tasks(self) -> Union[Response, Tuple[Response, int]]:
        """Handle HTTP request to get tasks by sampling from the task population."""
        request_start_time = time.time()

        try:
            # Get batch size from query parameter, default to 1
            batch_size = int(request.args.get("batch_size", 1))
            batch_size = max(1, min(batch_size, 1000))  # Clamp between 1 and 1000
            client_id = request.args.get("client_id")
            start_time = time.time()
            logger.debug(f"curriculum server: {client_id} requested {batch_size} tasks")

            tasks = {}
            tasks_response = []
            start_id = None
            with self._assignment_lock:
                start_id = self._task_id_counter
                self._task_id_counter += batch_size

            # Sample tasks from population
            selected_tasks = []
            with self._population_lock:
                if self._task_population:
                    # Randomly sample tasks from population (with replacement)
                    selected_tasks = []
                    for _ in range(batch_size):
                        task = random.choice(self._task_population)
                        selected_tasks.append(task.clone())

            # Allocate unique IDs and create response
            for task in selected_tasks:
                task_id = start_id
                start_id += 1

                # Store in assigned tasks
                tasks[str(task_id)] = task

                # Create response data
                env_cfg = OmegaConf.to_container(task.env_cfg(), resolve=True)
                task_data = {"id": task_id, "env_cfg": env_cfg, "name": task.name()}
                tasks_response.append(task_data)

            # Store assigned tasks and update metrics
            with self._assignment_lock:
                self._assigned_tasks.update(tasks)
                self._num_tasks_assigned += batch_size
                self._num_requests += 1
                wait_time = time.time() - request_start_time
                self._wait_times.append(wait_time)

            logger.debug(
                f"curriculum server: {client_id} sent {batch_size} tasks in {time.time() - start_time} seconds"
            )
            return jsonify({"tasks": tasks_response, "status": "ok"})
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return jsonify({"error": str(e), "status": "error"}), 500

    def _complete_task(self) -> Union[Response, Tuple[Response, int]]:
        """Complete a task."""
        data = request.get_json()
        id = data.get("id")
        score = data.get("score")
        # Convert id to string to match how it's stored in _assigned_tasks
        id_str = str(id)
        with self._assignment_lock:
            if id_str not in self._assigned_tasks:
                logger.error(f"Task {id} not found")
                return jsonify({"error": f"Task {id} not found", "status": "error"}), 404
            task = self._assigned_tasks.pop(id_str)
            self._num_tasks_completed += 1
        task.complete(score)
        return jsonify({"status": "ok"})

    def _health(self) -> Response:
        """Health check endpoint."""
        return jsonify({"status": "healthy"})

    def start(self):
        """Start the curriculum server."""
        # Start generation thread
        self._stop_generation.clear()
        self._generation_thread = threading.Thread(target=self._generate_tasks_loop, daemon=True)
        self._generation_thread.start()

        # Start server
        self._server = make_server(self._host, self._port, self._app, threaded=True)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

        # Start monitoring thread
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_stats, daemon=True)
        self._monitor_thread.start()

        logger.info(f"Curriculum server started at http://{self._host}:{self._port}")

    def stop(self):
        """Stop the curriculum server."""
        # Stop generation thread
        self._stop_generation.set()
        if self._generation_thread:
            self._generation_thread.join(timeout=2)

        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        # Shutdown thread pool
        self._executor.shutdown(wait=True, cancel_futures=True)

        # Shutdown server
        if self._server:
            self._server.shutdown()
            if self._server_thread:
                self._server_thread.join(timeout=5)
            logger.info("Curriculum server stopped")

    # Implement the Curriculum interface
    def get_task(self) -> Task:
        task = self._curriculum.get_task()
        logger.debug(f"Assigning task: {task.name()}")
        return task

    def stats(self) -> Dict[str, float]:
        return {
            **self._curriculum.stats(),
            "tasks_completed": self._num_tasks_completed,
            "tasks_assigned": self._num_tasks_assigned,
        }

    def complete_task(self, id: str, score: float):
        logger.debug(f"Completing task: {id} with score: {score}")
        self._curriculum.complete_task(id, score)
        self._num_tasks_completed += 1

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

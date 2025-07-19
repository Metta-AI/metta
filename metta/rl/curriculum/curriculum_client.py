"""
HTTP client for fetching curriculum tasks from a remote server.

This client implements the Curriculum interface but fetches tasks from
a remote curriculum server using a background thread for fetching.
"""

import logging
import os
import random
import threading
import time
from queue import Empty, Queue
from typing import Optional

import requests
from omegaconf import DictConfig, OmegaConf

from metta.common.util.retry import retry_function
from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class CurriculumClient(Curriculum):
    """Client that fetches curriculum tasks from the server."""

    def __init__(
        self,
        server_url: str,
        batch_size: int = 5,
        timeout: float = 60.0,
        max_retries: int = 20,
        buffer_size: int = 5,
        poll_interval: float = 10.0,
    ):
        """
        Initialize the curriculum client.

        Args:
            server_url: URL of the curriculum server (e.g., "http://localhost:12346")
            batch_size: Number of tasks to fetch in each request
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            buffer_size: Maximum number of tasks to keep in buffer
        """
        self._server_url = server_url.rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout
        self._max_retries = max_retries
        self._poll_interval = poll_interval
        self._client_id = os.environ.get("RANK", "0") + "_" + str(random.randint(0, 1000000))
        self._last_task: Optional[Task] = None
        self._task_queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()

        # Test connection
        self._wait_for_server()

        # Get initial tasks
        self._fetch_tasks()

        # Start background fetch thread
        self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._fetch_thread.start()

    def _wait_for_server(self):
        """Wait for the server to be ready."""
        for _ in range(300):
            try:
                with requests.Session() as session:
                    response = session.get(f"{self._server_url}/health", timeout=5.0)
                    response.raise_for_status()
                return
            except Exception:
                time.sleep(self._poll_interval)
        raise RuntimeError(f"{self._client_id}: Failed to connect to curriculum server: {self._server_url}")

    def _fetch_loop(self):
        """Background thread that continuously fetches tasks."""
        while not self._stop_event.is_set():
            try:
                # Only fetch if queue is not full
                if self._task_queue.qsize() < self._task_queue.maxsize * 0.5:
                    self._fetch_tasks()
                else:
                    time.sleep(self._poll_interval)
            except Exception as e:
                logger.error(f"{self._client_id}: Error in fetch loop: {e}")
                time.sleep(self._poll_interval)

    def _fetch_tasks(self):
        """Fetch a batch of tasks from the server and add to queue."""
        if self._stop_event.is_set():
            return

        def _do_fetch():
            with requests.Session() as session:
                response = session.get(
                    f"{self._server_url}/tasks",
                    params={"batch_size": self._batch_size, "client_id": self._client_id},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()

        try:
            data = retry_function(
                _do_fetch,
                max_retries=self._max_retries,
                error_prefix=f"{self._client_id}: Failed to fetch tasks",
                # logger=logger,
            )

            if data.get("status") != "ok":
                raise RuntimeError(f"Server returned error: {data.get('error', 'Unknown error')}")

            # Add tasks to queue
            tasks_added = 0
            for task_data in data.get("tasks", []):
                if self._stop_event.is_set():
                    return

                env_cfg = OmegaConf.create(task_data["env_cfg"])
                if not isinstance(env_cfg, DictConfig):
                    env_cfg = DictConfig(task_data["env_cfg"])

                task = Task(task_data["id"], self, env_cfg)
                task._name = task_data["name"]

                try:
                    self._task_queue.put(task, timeout=0.1)
                    tasks_added += 1
                except Exception:
                    # Queue is full, stop adding
                    break

            if tasks_added > 0:
                logger.debug(f"{self._client_id}: Added {tasks_added} tasks to buffer {self._task_queue.qsize()}")
            else:
                logger.warning(f"{self._client_id}: Server returned empty task batch")

        except Exception as e:
            logger.error(f"{self._client_id}: Failed to fetch tasks after retries: {e}")

    def get_task(self) -> Task:
        """Get a single task from the queue."""
        try:
            # Try to get from queue with timeout
            task = self._task_queue.get(timeout=10.0)
            self._last_task = task
            return task
        except Empty:
            # Queue is empty, return last task if available
            if self._last_task is not None:
                logger.warning(
                    f"{self._client_id}: No new tasks available, returning last task. "
                    "This may indicate server connectivity issues or task exhaustion."
                )
                return self._last_task

            raise RuntimeError(
                f"{self._client_id}: Failed to get tasks from server and no previous task available"
            ) from None

    def complete_task(self, id: str, score: float):
        """Report task completion to the server."""

        def _do_complete():
            with requests.Session() as session:
                response = session.post(
                    f"{self._server_url}/complete",
                    json={"id": id, "score": float(score), "client_id": self._client_id},
                    timeout=self._timeout,
                )
                response.raise_for_status()

        try:
            retry_function(
                _do_complete,
                max_retries=self._max_retries,
                error_prefix=f"{self._client_id}: Failed to complete task {id}",
                logger=logger,
            )
        except Exception as e:
            logger.error(f"{self._client_id}: Failed to complete task {id} after retries: {e}")

    def stop(self):
        """Clean shutdown - stop background thread."""
        self._stop_event.set()
        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=5.0)

    def stats(self) -> dict:
        """Stats are handled by the server, so return empty dict."""
        return {}

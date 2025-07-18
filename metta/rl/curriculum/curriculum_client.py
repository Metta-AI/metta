"""
HTTP client for fetching curriculum tasks from a remote server.

This client implements the Curriculum interface but fetches tasks from
a remote curriculum server. It includes batching and background prefetching
to reduce network overhead.
"""

import logging
import queue
import threading
import time

import requests
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class CurriculumClient(Curriculum):
    """Client that fetches curriculum tasks from the server with background prefetching."""

    def __init__(
        self,
        server_url: str,
        batch_size: int = 100,
        timeout: float = 30.0,
        retry_delay: float = 1.0,
        max_retries: int = 5,
        prefetch_threshold: float = 0.5,
        queue_size: int = 1000,
    ):
        """
        Initialize the curriculum client.

        Args:
            server_url: URL of the curriculum server (e.g., "http://localhost:5555")
            batch_size: Number of tasks to fetch in each request
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retries for failed requests
            prefetch_threshold: Fraction of batch_size - prefetch when queue drops below this
            queue_size: Maximum size of the task queue
        """
        self.server_url = server_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.prefetch_threshold = prefetch_threshold
        self._task_queue = queue.Queue(maxsize=queue_size)
        self._session = requests.Session()
        self._stop_prefetch = threading.Event()
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None

        self._wait_for_server()
        self._start_prefetch_thread()

    def _wait_for_server(self):
        """Wait for the server to be ready. Timeout after 10 seconds."""
        for _ in range(10):
            try:
                response = self._session.get(f"{self.server_url}/health", timeout=5.0)
                response.raise_for_status()
                logger.info(f"CurriculumClient connected to: {self.server_url}")
                return
            except Exception:
                time.sleep(1)
        logger.error(f"Failed to connect to curriculum server: {self.server_url}")
        raise RuntimeError(f"Failed to connect to curriculum server: {self.server_url}")

    def _start_prefetch_thread(self):
        """Start the background prefetch thread."""
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker that monitors queue and fetches tasks when needed."""
        while not self._stop_prefetch.is_set():
            try:
                # Check if we need to fetch more tasks
                queue_size = self._task_queue.qsize()
                threshold = int(self.batch_size * self.prefetch_threshold)

                if queue_size < threshold:
                    with self._prefetch_lock:
                        # Double-check after acquiring lock and stop flag
                        if self._task_queue.qsize() < threshold and not self._stop_prefetch.is_set():
                            self._fetch_tasks()

                # Small sleep to avoid busy waiting
                time.sleep(0.1)

            except Exception as e:
                # Only log error if we're not stopping
                if not self._stop_prefetch.is_set():
                    logger.error(f"Error in prefetch worker: {e}")
                time.sleep(1)  # Back off on error

    def get_task(self) -> Task:
        """Get a single task from the queue."""
        # If queue is empty, fetch synchronously
        if self._task_queue.empty():
            with self._prefetch_lock:
                if self._task_queue.empty():
                    self._fetch_tasks()

        try:
            # Get task from queue with timeout
            return self._task_queue.get(timeout=5.0)
        except queue.Empty as e:
            raise RuntimeError("Failed to get task from queue - server may be down") from e

    def _fetch_tasks(self) -> None:
        """Fetch a batch of tasks from the server and add to queue."""
        url = f"{self.server_url}/tasks"
        params = {"batch_size": self.batch_size}

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                if data.get("status") != "ok":
                    raise RuntimeError(f"Server returned error: {data.get('error', 'Unknown error')}")

                # Check if server returned any tasks
                if not data["tasks"]:
                    logger.warning("Server returned empty task batch")
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("Server returned empty task batch")
                    continue

                # Add tasks to queue
                tasks_added = 0
                for task_data in data["tasks"]:
                    task = Task(task_data["name"], self, OmegaConf.create(task_data["env_cfg"]))
                    try:
                        self._task_queue.put_nowait(task)
                        tasks_added += 1
                    except queue.Full:
                        # Queue is full, stop adding
                        break

                if tasks_added > 0:
                    logger.debug(f"Added {tasks_added} tasks to queue (fetched {len(data['tasks'])})")
                    return
                else:
                    logger.warning("Could not add any tasks to queue - queue is full")
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("Queue is full and cannot add new tasks")

            except requests.exceptions.RequestException as e:
                # Check if we're stopping before logging/retrying
                if self._stop_prefetch.is_set():
                    return
                logger.warning(f"Failed to fetch tasks (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to fetch tasks after {self.max_retries} attempts") from e

    def get_curriculum_stats(self) -> dict:
        """Return empty stats - all stats are managed by the server."""
        return {}

    def stop(self):
        """Stop the background prefetch thread."""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=2.0)
        if hasattr(self, "_session"):
            self._session.close()

    @staticmethod
    def create(trainer_cfg: DictConfig) -> "CurriculumClient":
        url = f"http://{trainer_cfg.curriculum_server.host}:{trainer_cfg.curriculum_server.port}"
        logger.info(f"CurriculumClient connecting to: {url}")
        return CurriculumClient(
            server_url=url,
            batch_size=trainer_cfg.curriculum_server.batch_size,
        )

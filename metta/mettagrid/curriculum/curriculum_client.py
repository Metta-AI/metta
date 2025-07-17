"""
HTTP client for fetching curriculum tasks from a remote server.

This client implements the Curriculum interface but fetches tasks from
a remote curriculum server. It includes batching to reduce network overhead.
"""

import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import requests
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class RemoteTask(Task):
    """A task fetched from a remote curriculum server."""
    
    def __init__(self, task_id: str, task_name: str, env_cfg: DictConfig, client: "CurriculumClient"):
        # Convert dict to DictConfig if needed
        if isinstance(env_cfg, dict):
            env_cfg = OmegaConf.create(env_cfg)
        
        # Initialize without a curriculum first
        super().__init__(task_name, curriculum=None, env_cfg=env_cfg)  # type: ignore
        
        # Store the remote task ID and client reference
        self._remote_id = task_id
        self._client = client
        # Override the name to match what server sent
        self._name = task_name
    
    def complete(self, score: float):
        """Complete the task by notifying the remote server."""
        if self._is_complete:
            return
            
        try:
            self._client._complete_remote_task(self._remote_id, score)
            self._is_complete = True
        except Exception as e:
            logger.error(f"Failed to complete remote task {self._remote_id}: {e}")
            raise


class CurriculumClient(Curriculum):
    """Client that fetches tasks from a remote curriculum server."""
    
    def __init__(
        self, 
        server_url: str,
        batch_size: int = 100,
        prefetch_threshold: float = 0.5,
        max_retries: int = 3,
        timeout: float = 10.0
    ):
        """
        Initialize curriculum client.
        
        Args:
            server_url: URL of the curriculum server (e.g., "http://localhost:8080")
            batch_size: Number of tasks to fetch in each batch
            prefetch_threshold: Fetch new batch when queue size falls below this fraction
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.batch_size = batch_size
        self.prefetch_threshold = prefetch_threshold
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Task queue for batching
        self._task_queue: Queue[RemoteTask] = Queue()
        self._lock = threading.Lock()
        
        # Cache for stats
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._stats_lock = threading.Lock()
        
        # Ensure server is reachable
        self._check_server_health()
        
        # Prefetch initial batch
        self._fetch_task_batch()
    
    def _check_server_health(self):
        """Check if the server is healthy."""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to curriculum server at {self.server_url}: {e}")
    
    def _fetch_task_batch(self):
        """Fetch a batch of tasks from the server."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/tasks",
                    json={"batch_size": self.batch_size},
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                tasks = data["tasks"]
                
                # Add tasks to queue
                with self._lock:
                    for task_info in tasks:
                        remote_task = RemoteTask(
                            task_id=task_info["task_id"],
                            task_name=task_info["task_name"],
                            env_cfg=task_info["env_cfg"],
                            client=self
                        )
                        self._task_queue.put(remote_task)
                
                logger.debug(f"Fetched {len(tasks)} tasks from server")
                return
                
            except Exception as e:
                logger.warning(f"Failed to fetch tasks (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to fetch tasks after {self.max_retries} attempts: {e}")
    
    def _complete_remote_task(self, task_id: str, score: float):
        """Notify server that a task is complete."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/complete",
                    json={"task_id": task_id, "score": score},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return
                
            except Exception as e:
                logger.warning(f"Failed to complete task (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    # Don't fail training if we can't report completion
                    logger.error(f"Failed to complete task after {self.max_retries} attempts: {e}")
    
    def _fetch_stats(self):
        """Fetch current statistics from server."""
        try:
            response = requests.get(
                f"{self.server_url}/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            with self._stats_lock:
                self._cached_stats = response.json()
                
        except Exception as e:
            logger.warning(f"Failed to fetch stats from server: {e}")
    
    def get_task(self) -> Task:
        """Get a task from the queue, fetching more if needed."""
        with self._lock:
            # Check if we need to prefetch more tasks
            queue_size = self._task_queue.qsize()
            if queue_size < self.batch_size * self.prefetch_threshold:
                # Fetch in background to avoid blocking
                threading.Thread(target=self._fetch_task_batch, daemon=True).start()
            
            # Get task from queue
            try:
                task = self._task_queue.get_nowait()
            except Empty:
                # Queue is empty, fetch synchronously
                logger.warning("Task queue empty, fetching synchronously")
                self._fetch_task_batch()
                task = self._task_queue.get()
        
        return task
    
    def complete_task(self, id: str, score: float):
        """Legacy method - tasks complete themselves in RemoteTask.complete()."""
        # This is called by parent curriculums, but we handle completion in RemoteTask
        pass
    
    def get_env_cfg_by_bucket(self) -> Dict[str, DictConfig]:
        """Get environment configs by bucket from server."""
        self._fetch_stats()
        
        with self._stats_lock:
            if self._cached_stats is None:
                return {}
            
            # Convert dicts back to DictConfigs
            result = {}
            for key, cfg_dict in self._cached_stats.get("env_cfg_by_bucket", {}).items():
                result[key] = OmegaConf.create(cfg_dict)
            return result
    
    def get_task_probs(self) -> Dict[str, float]:
        """Get current task probabilities from server."""
        self._fetch_stats()
        
        with self._stats_lock:
            if self._cached_stats is None:
                return {}
            return self._cached_stats.get("task_probs", {})
    
    def get_completion_rates(self) -> Dict[str, float]:
        """Get task completion rates from server."""
        self._fetch_stats()
        
        with self._stats_lock:
            if self._cached_stats is None:
                return {}
            return self._cached_stats.get("completion_rates", {})
    
    def get_curriculum_stats(self) -> Dict[str, float]:
        """Get curriculum statistics from server."""
        self._fetch_stats()
        
        with self._stats_lock:
            if self._cached_stats is None:
                return {}
            return self._cached_stats.get("curriculum_stats", {})
    
    def completed_tasks(self) -> List[str]:
        """Get list of completed tasks - not supported for remote curriculum."""
        # This would require tracking on server side
        return []
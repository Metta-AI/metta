import logging
import random
import time
from typing import Any, Dict, List

import requests
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class CurriculumTask:
    """A task returned by the curriculum."""
    
    def __init__(self, name: str, env_cfg_dict: Dict[str, Any]):
        self.name = name
        self._env_cfg_dict = env_cfg_dict
        self._env_cfg = None
        self.id = name  # For compatibility
    
    def env_cfg(self) -> DictConfig:
        """Get the environment configuration as a DictConfig."""
        if self._env_cfg is None:
            self._env_cfg = OmegaConf.create(self._env_cfg_dict)
        return self._env_cfg

    def complete(self, score: float):
        """No-op for client - server handles all curriculum logic."""
        pass


class CurriculumClient:
    """Client that fetches curriculum tasks from the server."""
    
    def __init__(
        self, 
        server_url: str, 
        batch_size: int = 100,
        timeout: float = 30.0,
        retry_delay: float = 1.0,
        max_retries: int = 5
    ):
        """
        Initialize the curriculum client.
        
        Args:
            server_url: URL of the curriculum server (e.g., "http://localhost:5555")
            batch_size: Number of tasks to fetch in each request
            timeout: Request timeout in seconds
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.server_url = server_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self._task_batch: List[CurriculumTask] = []
        self._session = requests.Session()
    
    def get_task(self) -> CurriculumTask:
        """Get a single task, fetching more from the server if needed."""
        # If batch is empty, fetch more tasks
        if not self._task_batch:
            self._fetch_tasks()
        
        # Randomly select a task from the batch
        if self._task_batch:
            return random.choice(self._task_batch)
        else:
            raise RuntimeError("Failed to fetch tasks from curriculum server")
    
    def _fetch_tasks(self) -> None:
        """Fetch a batch of tasks from the server."""
        url = f"{self.server_url}/tasks"
        params = {"batch_size": self.batch_size}
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    url, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") != "ok":
                    raise RuntimeError(f"Server returned error: {data.get('error', 'Unknown error')}")
                
                # Replace the batch with new tasks
                self._task_batch = []
                for task_data in data["tasks"]:
                    task = CurriculumTask(
                        name=task_data["name"],
                        env_cfg_dict=task_data["env_cfg"]
                    )
                    self._task_batch.append(task)
                
                if self._task_batch:
                    logger.debug(f"Fetched {len(data['tasks'])} tasks from curriculum server")
                    return
                else:
                    logger.warning("Server returned no tasks")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch tasks (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to fetch tasks after {self.max_retries} attempts") from e
    
    def get_env_cfg_by_bucket(self) -> Dict[str, DictConfig]:
        """
        Get environment configurations by bucket.
        This is a compatibility method for the existing curriculum interface.
        For the client, we return all unique configs from the current batch.
        """
        # Get tasks if we don't have any
        if not self._task_batch:
            self._fetch_tasks()
        
        configs = {}
        for task in self._task_batch:
            configs[task.name] = task.env_cfg()
        
        if not configs:
            # Return empty config if no tasks available
            return {"default": OmegaConf.create()}
        
        return configs
    
    def complete_task(self, id: str, score: float):
        """No-op for client - server handles all curriculum logic."""
        pass
    
    def get_completion_rates(self) -> Dict[str, float]:
        """No-op for client - server handles all stats."""
        return {}
    
    def get_task_probs(self) -> Dict[str, float]:
        """No-op for client - server handles all stats."""
        return {}
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """No-op for client - server handles all stats."""
        return {}
    
    def __del__(self):
        """Close the session when the client is destroyed."""
        if hasattr(self, "_session"):
            self._session.close()
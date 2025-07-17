"""
HTTP server for serving curriculum tasks to distributed environments.

This server wraps a curriculum object and provides HTTP endpoints for:
- Getting tasks (with batch support)
- Completing tasks
- Getting curriculum statistics
"""

import logging
import threading
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    """Request for getting tasks from curriculum."""
    batch_size: int = 1


class TaskInfo(BaseModel):
    """Information about a task."""
    task_id: str
    task_name: str
    env_cfg: Dict[str, Any]


class TaskResponse(BaseModel):
    """Response containing task(s) from curriculum."""
    tasks: List[TaskInfo]


class CompleteTaskRequest(BaseModel):
    """Request to mark a task as complete."""
    task_id: str
    score: float


class StatsResponse(BaseModel):
    """Response containing curriculum statistics."""
    task_probs: Dict[str, float]
    completion_rates: Dict[str, float]
    curriculum_stats: Dict[str, float]
    env_cfg_by_bucket: Dict[str, Dict[str, Any]]


class CurriculumServer:
    """HTTP server that serves tasks from a curriculum."""
    
    def __init__(self, curriculum: Curriculum, host: str = "0.0.0.0", port: int = 8080):
        self.curriculum = curriculum
        self.host = host
        self.port = port
        self.app = FastAPI()
        self._setup_routes()
        self._lock = threading.Lock()
        
        # Track tasks by ID so we can complete them later
        self._tasks: Dict[str, Task] = {}
        
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}
        
        @self.app.post("/tasks", response_model=TaskResponse)
        async def get_tasks(request: TaskRequest):
            """Get one or more tasks from the curriculum."""
            tasks = []
            
            with self._lock:
                for _ in range(request.batch_size):
                    task = self.curriculum.get_task()
                    # Generate a unique ID for this task instance
                    task_uuid = str(uuid4())
                    self._tasks[task_uuid] = task
                    
                    # Convert env_cfg to regular dict for JSON serialization
                    env_cfg_dict = OmegaConf.to_container(task.env_cfg(), resolve=True)
                    if not isinstance(env_cfg_dict, dict):
                        raise ValueError(f"Expected dict for env_cfg, got {type(env_cfg_dict)}")
                    
                    tasks.append(TaskInfo(
                        task_id=task_uuid,
                        task_name=task.name(),
                        env_cfg=env_cfg_dict
                    ))
            
            return TaskResponse(tasks=tasks)
        
        @self.app.post("/complete")
        async def complete_task(request: CompleteTaskRequest):
            """Mark a task as complete with a score."""
            with self._lock:
                if request.task_id not in self._tasks:
                    raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
                
                task = self._tasks[request.task_id]
                if not task.is_complete():
                    task.complete(request.score)
                
                # Clean up completed task after some time to prevent memory growth
                # For now, we'll keep them to avoid issues with duplicate completions
                # In production, you might want to add a TTL or cleanup old tasks
            
            return {"status": "ok"}
        
        @self.app.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """Get curriculum statistics."""
            with self._lock:
                env_cfg_by_bucket = self.curriculum.get_env_cfg_by_bucket()
                # Convert all DictConfigs to regular dicts
                env_cfg_dict = {}
                for key, cfg in env_cfg_by_bucket.items():
                    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                    if isinstance(cfg_dict, dict):
                        env_cfg_dict[key] = cfg_dict
                    else:
                        logger.warning(f"Skipping non-dict config for key {key}")
                
                return StatsResponse(
                    task_probs=self.curriculum.get_task_probs(),
                    completion_rates=self.curriculum.get_completion_rates(), 
                    curriculum_stats=self.curriculum.get_curriculum_stats(),
                    env_cfg_by_bucket=env_cfg_dict
                )
    
    def start(self, run_in_thread: bool = False):
        """Start the server."""
        logger.info(f"Starting curriculum server on {self.host}:{self.port}")
        
        if run_in_thread:
            thread = threading.Thread(
                target=uvicorn.run,
                args=(self.app,),
                kwargs={"host": self.host, "port": self.port, "log_level": "warning"},
                daemon=True
            )
            thread.start()
            return thread
        else:
            uvicorn.run(self.app, host=self.host, port=self.port)


def create_curriculum_server(
    curriculum_config: str,
    env_overrides: Optional[Dict[str, Any]] = None,
    host: str = "0.0.0.0",
    port: int = 8080
) -> CurriculumServer:
    """Create a curriculum server from a config path."""
    from omegaconf import DictConfig
    
    env_overrides_cfg = DictConfig(env_overrides) if env_overrides else DictConfig({})
    curriculum = curriculum_from_config_path(curriculum_config, env_overrides_cfg)
    return CurriculumServer(curriculum, host, port)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python curriculum_server.py <curriculum_config_path> [port]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    server = create_curriculum_server(config_path, port=port)
    server.start()
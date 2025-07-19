import logging
import random
import threading
from typing import Any, Dict

from flask import Flask, jsonify, request
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)

# Disable werkzeug route logging (only show errors)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class CurriculumServer:
    """HTTP server that serves curriculum tasks to distributed environments."""

    def __init__(self, curriculum: Curriculum, host: str = "127.0.0.1", port: int = 12346):
        self._curriculum = curriculum
        self._host = host
        self._port = port
        self._app = Flask(__name__)
        self._lock = threading.Lock()
        self._server_thread = None
        self._server = None

        self._assigned_tasks = {}

        # Set up routes
        self._app.add_url_rule("/tasks", "get_tasks", self._get_tasks, methods=["GET"])
        self._app.add_url_rule("/complete", "complete_task", self._complete_task, methods=["POST"])
        self._app.add_url_rule("/health", "health", self._health, methods=["GET"])

        # Metrics
        self._num_tasks_completed = 0

    def _get_tasks(self) -> Dict[str, Any]:
        """Get one or more tasks from the curriculum."""
        try:
            # Get batch size from query parameter, default to 1
            batch_size = int(request.args.get("batch_size", 1))
            batch_size = max(1, min(batch_size, 1000))  # Clamp between 1 and 1000

            tasks = []
            with self._lock:
                for _ in range(batch_size):
                    try:
                        task = self._curriculum.get_task()
                        # Convert the task's env_cfg to a serializable format
                        env_cfg = OmegaConf.to_container(task.env_cfg(), resolve=True)
                        task_id = random.randint(0, 1000000)
                        tasks.append({"id": task_id, "env_cfg": env_cfg})
                        self._assigned_tasks[task_id] = task
                    except Exception as e:
                        # If curriculum can't provide more tasks, break and return what we have
                        logger.warning(f"Curriculum stopped providing tasks: {e}")
                        break

            return jsonify({"tasks": tasks, "status": "ok"})
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return jsonify({"error": str(e), "status": "error"}), 500

    def _complete_task(self) -> Dict[str, Any]:
        """Complete a task."""
        data = request.get_json()
        id = data.get("id")
        score = data.get("score")
        if id not in self._assigned_tasks:
            logger.error(f"Task {id} not found")
            return jsonify({"error": f"Task {id} not found", "status": "error"}), 404
        task = self._assigned_tasks.pop(id)
        task.complete(score)
        self._num_tasks_completed += 1
        return jsonify({"status": "ok"})

    def _health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return jsonify({"status": "healthy"})

    def start(self, background: bool = True):
        """Start the curriculum server."""
        if background:
            from werkzeug.serving import make_server

            self._server = make_server(self._host, self._port, self._app, threaded=True)
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            logger.debug(f"Curriculum server started at http://{self._host}:{self._port}")
        else:
            # Run in foreground (blocking)
            self._app.run(host=self._host, port=self._port, threaded=True)

    def stop(self):
        """Stop the curriculum server."""
        if self._server:
            self._server.shutdown()
            if self._server_thread:
                self._server_thread.join(timeout=5)
            logger.info("Curriculum server stopped")

    @staticmethod
    def create(trainer_cfg: DictConfig) -> "CurriculumServer":
        # Create curriculum from config
        curriculum_config = trainer_cfg.get("curriculum")
        if not curriculum_config:
            raise ValueError("Curriculum must be set in trainer config")
        env_overrides = DictConfig(trainer_cfg.get("env_overrides", {}))
        curriculum = curriculum_from_config_path(curriculum_config, env_overrides)

        # Get host from config, default to 0.0.0.0 for Docker compatibility
        host = trainer_cfg.curriculum_server.get("host", "0.0.0.0")
        port = trainer_cfg.curriculum_server.port

        server = CurriculumServer(curriculum=curriculum, host=host, port=port)
        server.start(background=True)
        logger.info(f"Started curriculum server on {host}:{port}")
        return server

    # Implement the Curriculum interface
    def get_task(self) -> Task:
        task = self._curriculum.get_task()
        logger.debug(f"Assigning task: {task.name()}")
        return task

    def get_curriculum_stats(self) -> Dict[str, float]:
        return {"tasks_completed": self._num_tasks_completed, **self._curriculum.stats()}

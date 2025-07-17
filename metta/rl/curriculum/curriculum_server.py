import logging
import threading
from typing import Dict, Tuple, Union

from flask import Flask, Response, jsonify, request
from omegaconf import OmegaConf
from werkzeug.serving import make_server

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)

# Disable werkzeug route logging (only show errors)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class CurriculumServer(Curriculum):
    """HTTP server that serves curriculum tasks to distributed environments."""

    def __init__(self, curriculum: Curriculum, port: int = 12346):
        self._curriculum = curriculum
        self._host = "0.0.0.0"
        self._port = port
        self._app = Flask(__name__)
        self._lock = threading.Lock()
        self._server_thread = None
        self._server = None


        self._assigned_tasks = {}
        self._task_id_counter = 0

        # Set up routes
        self._app.add_url_rule("/tasks", "get_tasks", self._get_tasks, methods=["GET"])
        self._app.add_url_rule("/complete", "complete_task", self._complete_task, methods=["POST"])
        self._app.add_url_rule("/health", "health", self._health, methods=["GET"])

        # Metrics
        self._num_tasks_completed = 0

    def _get_tasks(self) -> Union[Response, Tuple[Response, int]]:
        """Get one or more tasks from the curriculum."""
        try:
            # Get batch size from query parameter, default to 1
            batch_size = int(request.args.get("batch_size", 1))
            batch_size = max(1, min(batch_size, 1000))  # Clamp between 1 and 1000

            tasks = []
            with self._lock:
                for _ in range(batch_size):
                    try:
                        task = self.get_task()
                        # Convert the task's env_cfg to a serializable format
                        env_cfg = OmegaConf.to_container(task.env_cfg(), resolve=True)
                        # Generate a numeric task ID
                        self._task_id_counter += 1
                        task_id = str(self._task_id_counter)
                        tasks.append({"id": task_id, "env_cfg": env_cfg})
                        self._assigned_tasks[task_id] = task
                    except Exception as e:
                        # If curriculum can't provide more tasks, break and return what we have
                        logger.warning(f"Curriculum stopped providing tasks: {e}")
                        break

            return jsonify({
                "tasks": tasks,
                "status": "ok"
            })

            return jsonify({"tasks": tasks, "status": "ok"})
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return jsonify({"error": str(e), "status": "error"}), 500

    def _complete_task(self) -> Union[Response, Tuple[Response, int]]:
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

    def _health(self) -> Response:
        """Health check endpoint."""
        return jsonify({"status": "healthy"})

    def start(self, background: bool = True):

    def start(self):
        """Start the curriculum server."""
        self._server = make_server(self._host, self._port, self._app, threaded=True)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info(f"Curriculum server started at http://{self._host}:{self._port}")

    def stop(self):
        """Stop the curriculum server."""
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
        return {**self._curriculum.stats(), "tasks_completed": self._num_tasks_completed}

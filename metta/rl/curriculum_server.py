import logging
import threading
from typing import Any, Dict

from flask import Flask, jsonify, request
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class CurriculumServer:
    """HTTP server that serves curriculum tasks to distributed environments."""
    
    def __init__(self, curriculum: Any, host: str = "0.0.0.0", port: int = 5555):
        self.curriculum = curriculum
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._lock = threading.Lock()
        self._server_thread = None
        self._server = None
        
        # Set up routes
        self.app.add_url_rule("/tasks", "get_tasks", self._get_tasks, methods=["GET"])
        self.app.add_url_rule("/health", "health", self._health, methods=["GET"])
        
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
                        task = self.curriculum.get_task()
                        # Convert the task's env_cfg to a serializable format
                        env_cfg = OmegaConf.to_container(task.env_cfg(), resolve=True)
                        # Get task name - it's a method in the Task class
                        task_name = task.name() if hasattr(task, "name") else "unknown"
                        tasks.append({
                            "name": task_name,
                            "env_cfg": env_cfg
                        })
                    except Exception as e:
                        # If curriculum can't provide more tasks, break and return what we have
                        logger.warning(f"Curriculum stopped providing tasks: {e}")
                        break
            
            return jsonify({
                "tasks": tasks,
                "status": "ok"
            })
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
    def _health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return jsonify({"status": "healthy"})
    
    def start(self, background: bool = True):
        """Start the curriculum server."""
        if background:
            from werkzeug.serving import make_server
            
            self._server = make_server(self.host, self.port, self.app, threaded=True)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True
            )
            self._server_thread.start()
            logger.info(f"Curriculum server started at http://{self.host}:{self.port}")
        else:
            # Run in foreground (blocking)
            self.app.run(host=self.host, port=self.port, threaded=True)
    
    def stop(self):
        """Stop the curriculum server."""
        if self._server:
            self._server.shutdown()
            if self._server_thread:
                self._server_thread.join(timeout=5)
            logger.info("Curriculum server stopped")
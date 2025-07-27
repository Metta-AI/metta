import json
import logging
import socket
import threading
import uuid
from typing import Dict

from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class CurriculumServer:
    """Server that manages curriculum tasks for distributed training using raw TCP sockets."""

    def __init__(self, curriculum: Curriculum, port: int = 5000, host: str = "0.0.0.0", buffer_size: int = 4096):
        self.curriculum = curriculum
        self.port = port
        self.host = host
        self.buffer_size = buffer_size

        # Store active tasks by ID
        self.active_tasks: Dict[str, Task] = {}
        self.active_tasks_lock = threading.Lock()

        # Server socket
        self.server_socket = None
        self.running = False

    def _handle_client(self, client_socket: socket.socket, client_address):
        """Handle a single client connection."""
        try:
            # Receive the request
            data = client_socket.recv(self.buffer_size).decode("utf-8")
            if not data:
                return

            request = json.loads(data)
            command = request.get("command")

            if command == "get_task":
                response = self._handle_get_task()
            elif command == "complete_task":
                response = self._handle_complete_task(request)
            elif command == "stats":
                response = self._handle_stats()
            elif command == "health":
                response = self._handle_health()
            else:
                response = {"success": False, "error": f"Unknown command: {command}"}

            # Send response
            response_data = json.dumps(response).encode("utf-8")
            client_socket.sendall(response_data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client: {e}")
            error_response = json.dumps({"success": False, "error": "Invalid JSON format"}).encode("utf-8")
            client_socket.sendall(error_response)
        except Exception as e:
            logger.error(f"Error handling client request: {e}")
            error_response = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
            try:
                client_socket.sendall(error_response)
            except Exception:
                pass
        finally:
            client_socket.close()

    def _handle_get_task(self) -> dict:
        """Generate a new task and return its information."""
        try:
            # Get a new task from the curriculum
            task = self.curriculum.get_task()

            # Generate a unique ID for tracking
            task_id = str(uuid.uuid4())

            # Store the task
            with self.active_tasks_lock:
                self.active_tasks[task_id] = task

            # Return task information
            return {
                "success": True,
                "task_id": task_id,
                "task_name": task.name(),
                "task_short_name": task.short_name(),
                "env_cfg": OmegaConf.to_container(task.env_cfg()),
            }
        except Exception as e:
            logger.error(f"Error generating task: {e}")
            return {"success": False, "error": str(e)}

    def _handle_complete_task(self, request: dict) -> dict:
        """Mark a task as complete with the given score."""
        try:
            task_id = request.get("task_id")
            score = request.get("score")

            if not task_id:
                return {"success": False, "error": "task_id is required"}

            if score is None:
                return {"success": False, "error": "score is required"}

            # Find the task
            with self.active_tasks_lock:
                task = self.active_tasks.get(task_id)
                if not task:
                    return {"success": False, "error": f"Task {task_id} not found"}

                # Complete the task
                task.complete(float(score))

                # Remove from active tasks
                del self.active_tasks[task_id]

            logger.info(f"Task {task_id} ({task.name()}) completed with score {score}")

            return {"success": True, "message": f"Task {task_id} completed successfully"}

        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return {"success": False, "error": str(e)}

    def _handle_stats(self) -> dict:
        """Get curriculum statistics."""
        try:
            stats = self.curriculum.stats()
            with self.active_tasks_lock:
                stats["active_tasks"] = len(self.active_tasks)

            return {"success": True, "stats": stats}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"success": False, "error": str(e)}

    def _handle_health(self) -> dict:
        """Health check endpoint."""
        with self.active_tasks_lock:
            active_count = len(self.active_tasks)

        return {"success": True, "status": "healthy", "active_tasks": active_count}

    def run(self):
        """Start the curriculum server."""
        logger.info(f"Starting curriculum server on {self.host}:{self.port}")

        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        self.running = True

        logger.info(f"Curriculum server listening on {self.host}:{self.port}")

        try:
            while self.running:
                # Accept client connections
                client_socket, client_address = self.server_socket.accept()

                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_socket, client_address), daemon=True
                )
                client_thread.start()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Curriculum server stopped")


def run_curriculum_server(curriculum: Curriculum, port: int = 5000, host: str = "0.0.0.0"):
    """Convenience function to create and run a curriculum server."""
    server = CurriculumServer(curriculum, port, host)
    server.run()

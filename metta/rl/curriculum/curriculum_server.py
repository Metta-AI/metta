import base64
import json
import logging
import pickle
import socket
import threading
import time
import uuid
from typing import Dict

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class CurriculumServer:
    """Server that manages curriculum tasks for distributed training using raw TCP sockets."""

    def __init__(self, curriculum: Curriculum, port: int = 5555, buffer_size: int = 4096, auto_start: bool = True):
        self.curriculum = curriculum
        self.port = port
        self.host = "0.0.0.0"  # Always bind to all interfaces
        self.buffer_size = buffer_size

        # Store active tasks by ID
        self.active_tasks: Dict[str, Task] = {}
        self.active_tasks_lock = threading.Lock()

        # Server socket
        self.server_socket = None
        self.running = False

        # Background thread for server
        self.server_thread = None

        # Start the server automatically in a background thread if requested
        if auto_start:
            self.start()

    def _send_message(self, client_socket: socket.socket, message: dict):
        """Send a message with length prefix to handle large payloads."""
        # Convert message to JSON bytes
        message_bytes = json.dumps(message).encode("utf-8")

        # Send length prefix (4 bytes, big-endian)
        message_length = len(message_bytes)
        client_socket.sendall(message_length.to_bytes(4, byteorder="big"))

        # Send the actual message
        client_socket.sendall(message_bytes)

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

            # Send response using length-prefixed protocol
            self._send_message(client_socket, response)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client: {e}")
            error_response = {"success": False, "error": "Invalid JSON format"}
            self._send_message(client_socket, error_response)
        except Exception as e:
            logger.error(f"Error handling client request: {e}")
            error_response = {"success": False, "error": str(e)}
            try:
                self._send_message(client_socket, error_response)
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
                "env_cfg": base64.b64encode(pickle.dumps(task.env_cfg())).decode("ascii"),
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

            logger.debug(f"Task {task_id} ({task.name()}) completed with score {score}")

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
        try:
            logger.info(f"Starting curriculum server on {self.host}:{self.port}")

            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            self.running = True

            logger.info(f"Curriculum server listening on {self.host}:{self.port}")

            while self.running:
                # Set a timeout so we can check self.running periodically
                self.server_socket.settimeout(1.0)
                try:
                    # Accept client connections
                    client_socket, client_address = self.server_socket.accept()

                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(client_socket, client_address), daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    # Timeout is expected, continue to check self.running
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server startup error: {e}")
            raise  # Re-raise to allow caller to handle
        finally:
            self.running = False
            if self.server_socket:
                try:
                    self.server_socket.close()
                except Exception:
                    pass

    def start(self):
        """Start the server in a background thread."""
        if self.server_thread is None or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(target=self.run, daemon=True)
            self.server_thread.start()
            # Give the server a moment to start up

            time.sleep(0.1)
            if not self.is_running():
                raise RuntimeError(f"Failed to start CurriculumServer on {self.host}:{self.port}")

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass  # Socket might already be closed
        if self.server_thread is not None:
            self.server_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        logger.info("Curriculum server stopped")

    def is_running(self):
        """Check if the server is running."""
        return self.running and self.server_thread is not None and self.server_thread.is_alive()


def run_curriculum_server(curriculum: Curriculum, port: int = 5555):
    """Convenience function to create and run a curriculum server."""
    server = CurriculumServer(curriculum, port)
    # The server auto-starts in __init__, so we just need to keep it running
    # This function blocks to maintain backward compatibility
    try:
        if server.server_thread is not None:
            server.server_thread.join()
    except KeyboardInterrupt:
        server.stop()

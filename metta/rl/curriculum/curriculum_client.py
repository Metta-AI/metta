import base64
import json
import logging
import pickle
import socket
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum, Task

logger = logging.getLogger(__name__)


class RemoteTask(Task):
    """A task that represents a remote task from the curriculum server."""

    def __init__(self, task_id: str, task_name: str, env_cfg: DictConfig, client: "CurriculumClient"):
        # Initialize without calling parent __init__ since we handle things differently
        self._id = task_id
        self._name = task_name
        self._is_complete = False
        self._env_cfg = env_cfg
        self._client = client
        self._server_task_id = task_id

    def complete(self, score: float):
        """Complete the task by notifying the server."""
        if self._is_complete:
            logger.warning(f"Task {self._id} is already complete")
            return

        # Convert score to Python float in case it's a numpy type
        score = float(score)

        # Notify the server
        success = self._client.complete_task(self._server_task_id, score)
        if success:
            self._is_complete = True
            logger.debug(f"Task {self._name} completed with score {score}")
        else:
            logger.error(f"Failed to complete task {self._name} on server")

    def short_name(self) -> str:
        return self._name.split("/")[-1]


class CurriculumClient(Curriculum):
    """Client that connects to a curriculum server to get tasks using raw TCP sockets."""

    def __init__(self, server_port: int = 5555, timeout: float = 30.0, buffer_size: int = 4096):
        self.server_host = "127.0.0.1"  # Always connect to localhost
        self.server_port = server_port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self._check_connection()

    def _receive_message(self, client_socket: socket.socket) -> Optional[dict]:
        """Receive a length-prefixed message from the server."""
        try:
            # First, receive the 4-byte length prefix
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = client_socket.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Server closed connection while reading length")
                length_bytes += chunk

            # Convert length from bytes
            message_length = int.from_bytes(length_bytes, byteorder="big")

            # Now receive the actual message
            message_bytes = b""
            while len(message_bytes) < message_length:
                chunk = client_socket.recv(min(self.buffer_size, message_length - len(message_bytes)))
                if not chunk:
                    raise ConnectionError("Server closed connection while reading message")
                message_bytes += chunk

            # Decode JSON
            return json.loads(message_bytes.decode("utf-8"))

        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def _send_request(self, request: dict) -> Optional[dict]:
        """Send a request to the server and get the response."""
        client_socket = None
        try:
            # Create socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # Connect to server
            client_socket.connect((self.server_host, self.server_port))

            # Send request
            request_data = json.dumps(request).encode("utf-8")
            client_socket.sendall(request_data)

            # Receive response using length-prefixed protocol
            return self._receive_message(client_socket)

        except socket.timeout:
            logger.error(f"Timeout connecting to curriculum server at {self.server_host}:{self.server_port}")
            return None
        except socket.error as e:
            logger.error(f"Socket error connecting to curriculum server: {e}")
            return None
        except Exception as e:
            logger.error(f"Error communicating with curriculum server: {e}")
            return None
        finally:
            if client_socket:
                client_socket.close()

    def _check_connection(self):
        """Check if the server is reachable."""
        response = self._send_request({"command": "health"})

        if response and response.get("success"):
            logger.info(f"Connected to curriculum server at {self.server_host}:{self.server_port}")
        else:
            raise ConnectionError(f"Failed to connect to curriculum server at {self.server_host}:{self.server_port}")

    def get_task(self) -> Task:
        """Get a new task from the server."""
        response = self._send_request({"command": "get_task"})

        if not response:
            raise RuntimeError("Failed to get task from server - no response")

        if not response.get("success"):
            raise RuntimeError(f"Server error: {response.get('error', 'Unknown error')}")

        # Create a RemoteTask with the server's data
        task_id = response["task_id"]
        task_name = response["task_name"]
        env_cfg_pickled = response["env_cfg"]

        # Decode base64 and unpickle the env_cfg
        env_cfg = pickle.loads(base64.b64decode(env_cfg_pickled))

        return RemoteTask(task_id, task_name, env_cfg, self)

    def complete_task(self, task_id: str, score: float) -> bool:
        """Notify the server that a task has been completed."""
        # Convert score to Python float in case it's a numpy type
        score = float(score)

        response = self._send_request({"command": "complete_task", "task_id": task_id, "score": score})

        if not response:
            logger.error("Failed to complete task on server - no response")
            return False

        if response.get("success"):
            return True
        else:
            logger.error(f"Server error completing task: {response.get('error', 'Unknown error')}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Get statistics from the server."""
        response = self._send_request({"command": "stats"})

        if not response:
            logger.error("Failed to get stats from server - no response")
            return {}

        if response.get("success"):
            return response.get("stats", {})
        else:
            logger.error(f"Server error getting stats: {response.get('error', 'Unknown error')}")
            return {}

    def completed_tasks(self) -> list[str]:
        """Get completed tasks from stats."""
        stats = self.stats()
        return stats.get("completed_tasks", [])

    def get_completion_rates(self) -> Dict[str, float]:
        """Get completion rates from stats."""
        stats = self.stats()
        return stats.get("completion_rates", {})

    def get_task_probs(self) -> Dict[str, float]:
        """Get task probabilities from stats."""
        stats = self.stats()
        return stats.get("task_probs", {})

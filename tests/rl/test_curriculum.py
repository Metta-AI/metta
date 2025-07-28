"""
Unit tests for the curriculum server and client.

This module provides tests for the CurriculumServer and CurriculumClient classes,
which handle distributed curriculum task management.
"""

import json
import socket
import threading
import time

import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.rl.curriculum import CurriculumClient, CurriculumServer


class TestCurriculumServerClient:
    """Test suite for the curriculum server and client."""

    @pytest.fixture
    def simple_curriculum(self):
        """Create a simple curriculum for testing."""
        task_cfg = DictConfig(
            {
                "game": {
                    "num_agents": 1,
                    "width": 10,
                    "height": 10,
                    "max_steps": 100,
                }
            }
        )
        return SingleTaskCurriculum("test_task", task_cfg)

    @pytest.fixture
    def curriculum_server(self, simple_curriculum):
        """Create and start a curriculum server in a background thread."""
        server = CurriculumServer(simple_curriculum, port=5555, auto_start=True)
        time.sleep(0.5)  # Give server time to start
        yield server
        server.stop()

    def test_server_initialization(self, simple_curriculum):
        """Test that a curriculum server initializes correctly."""
        server = CurriculumServer(simple_curriculum, port=5556, auto_start=False)

        assert server.curriculum == simple_curriculum
        assert server.port == 5556
        assert server.host == "0.0.0.0"
        assert server.buffer_size == 4096
        assert len(server.active_tasks) == 0
        assert not server.is_running()  # Server should not be running yet since auto_start=False

        # Now start it and check it's running
        server.start()
        assert server.is_running()

        # Clean up
        server.stop()

    def test_client_initialization_success(self, curriculum_server):
        """Test successful client initialization when server is running."""
        client = CurriculumClient(server_port=5555)
        assert client.server_host == "127.0.0.1"
        assert client.server_port == 5555

    def test_client_initialization_failure(self):
        """Test client initialization failure when no server is running."""
        with pytest.raises(ConnectionError):
            CurriculumClient(server_port=9999)

    def test_get_task(self, curriculum_server):
        """Test getting a task from the server."""
        client = CurriculumClient(server_port=5555)

        task = client.get_task()

        assert task is not None
        assert task.name() == "test_task"
        assert task.short_name() == "test_task"
        assert not task.is_complete()

        # Check that task has proper env config
        env_cfg = task.env_cfg()
        assert env_cfg.game.num_agents == 1
        assert env_cfg.game.width == 10
        assert env_cfg.game.height == 10

    def test_complete_task(self, curriculum_server):
        """Test completing a task."""
        client = CurriculumClient(server_port=5555)

        # Get a task
        task = client.get_task()

        # Complete the task
        score = 0.95
        task.complete(score)

        assert task.is_complete()

    def test_multiple_tasks(self, curriculum_server):
        """Test getting and completing multiple tasks."""
        client = CurriculumClient(server_port=5555)

        tasks = []
        for _ in range(3):
            task = client.get_task()
            tasks.append(task)
            assert task is not None

        # Complete tasks with different scores
        for i, task in enumerate(tasks):
            score = 0.8 + i * 0.05
            task.complete(score)
            assert task.is_complete()

    def test_stats(self, curriculum_server):
        """Test getting statistics from the server."""
        client = CurriculumClient(server_port=5555)

        # Initial stats
        stats = client.stats()
        assert "active_tasks" in stats
        assert stats["active_tasks"] == 0

        # Get a task and check stats
        task = client.get_task()
        stats = client.stats()
        assert stats["active_tasks"] == 1

        # Complete task and check stats
        task.complete(0.9)
        stats = client.stats()
        assert stats["active_tasks"] == 0

    def test_double_complete(self, curriculum_server):
        """Test that completing a task twice doesn't cause errors."""
        client = CurriculumClient(server_port=5555)

        task = client.get_task()

        # First completion should succeed
        task.complete(0.8)
        assert task.is_complete()

        # Second completion should be ignored
        task.complete(0.9)  # Should log a warning but not error

    def test_invalid_command(self, curriculum_server):
        """Test server response to invalid command."""
        # Manually create a socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(("127.0.0.1", 5555))

        # Send invalid command
        request = json.dumps({"command": "invalid_command"})
        client_socket.send(request.encode("utf-8"))

        # Receive response
        response_data = client_socket.recv(4096).decode("utf-8")
        response = json.loads(response_data)

        assert not response["success"]
        assert "Unknown command" in response["error"]

        client_socket.close()

    def test_malformed_json(self, curriculum_server):
        """Test server response to malformed JSON."""
        # Manually create a socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(("127.0.0.1", 5555))

        # Send malformed JSON
        client_socket.send(b"not valid json")

        # Receive response
        response_data = client_socket.recv(4096).decode("utf-8")
        response = json.loads(response_data)

        assert not response["success"]
        assert "Invalid JSON" in response["error"]

        client_socket.close()

    def test_concurrent_clients(self, curriculum_server):
        """Test multiple clients accessing the server concurrently."""
        clients = []
        tasks = []

        # Create multiple clients
        for _ in range(3):
            client = CurriculumClient(server_port=5555)
            clients.append(client)

        # Get tasks concurrently
        def get_task(client, result_list):
            task = client.get_task()
            result_list.append(task)

        threads = []
        for client in clients:
            thread = threading.Thread(target=get_task, args=(client, tasks))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all clients got tasks
        assert len(tasks) == 3
        for task in tasks:
            assert task is not None
            assert task.name() == "test_task"

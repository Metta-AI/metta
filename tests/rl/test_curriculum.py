"""
Unit tests for the curriculum server and client.

This module provides tests for the CurriculumServer and CurriculumClient classes,
which handle distributed curriculum task management via shared memory.
"""

import time

import pytest
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.rl.curriculum import CurriculumClient, CurriculumServer
from metta.rl.curriculum.curriculum_client import RemoteTask


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
        """Create and start a curriculum server."""
        server = CurriculumServer(simple_curriculum, num_slots=10, auto_start=True)
        time.sleep(0.5)  # Give server time to initialize
        yield server
        server.stop()

    def test_server_initialization(self, simple_curriculum):
        """Test that a curriculum server initializes correctly."""
        server = CurriculumServer(simple_curriculum, num_slots=10, auto_start=False)

        assert server.curriculum == simple_curriculum
        assert server.num_slots == 10
        assert not server.is_running()

        server.start()
        time.sleep(0.1)
        assert server.is_running()

        server.stop()
        assert not server.is_running()

    def test_client_initialization(self, curriculum_server):
        """Test that a curriculum client initializes correctly."""
        client = CurriculumClient(num_slots=10)
        assert client.num_slots == 10

    def test_get_task(self, curriculum_server):
        """Test getting a task from the server."""
        client = CurriculumClient(num_slots=10)
        task = client.get_task()

        assert task is not None
        assert hasattr(task, "_name")
        assert hasattr(task, "_env_cfg")
        assert hasattr(task, "complete")

    def test_complete_task(self, curriculum_server):
        """Test completing a task."""
        client = CurriculumClient(num_slots=10)
        task = client.get_task()

        # Complete the task
        score = 0.75
        task.complete(score)

        # Verify the task is marked as complete
        assert task._is_complete

    def test_multiple_completions(self, curriculum_server):
        """Test multiple completions of the same slot."""
        client = CurriculumClient(num_slots=10)

        # Get a task and complete it multiple times
        task = client.get_task()
        assert isinstance(task, RemoteTask)
        slot_idx = task._slot_idx

        scores = [0.5, 0.7, 0.9, 0.3, 0.8]
        for score in scores:
            success = client.complete_task(slot_idx, score)
            assert success

        # Check stats to verify completions
        stats = client.stats()
        assert stats["total_completions"] >= len(scores)

    def test_stats(self, curriculum_server):
        """Test getting statistics."""
        client = CurriculumClient(num_slots=10)

        # Get initial stats
        stats = client.stats()
        assert "total_completions" in stats
        assert "active_tasks" in stats
        assert "slot_utilization" in stats

        # Complete a task and check stats again
        task = client.get_task()
        task.complete(0.8)

        stats_after = client.stats()
        assert stats_after["total_completions"] >= stats["total_completions"]

    def test_multiple_clients(self, curriculum_server):
        """Test multiple clients accessing the server."""
        clients = [CurriculumClient(num_slots=10) for _ in range(3)]

        # Each client gets a task
        tasks = []
        for client in clients:
            task = client.get_task()
            tasks.append(task)
            assert task is not None

        # Each client completes their task
        for i, (_, task) in enumerate(zip(clients, tasks, strict=False)):
            task.complete(0.5 + i * 0.1)

    def test_task_refresh(self, curriculum_server):
        """Test that tasks are refreshed after many completions."""
        client = CurriculumClient(num_slots=10)

        # Get a task
        task = client.get_task()
        assert isinstance(task, RemoteTask)
        slot_idx = task._slot_idx

        # Complete the task more than 5 times
        for i in range(7):
            client.complete_task(slot_idx, 0.5 + i * 0.05)

        # Give the server time to refresh the task
        time.sleep(0.5)

        # The slot should have been refreshed with a new task
        # Get stats to check
        stats = curriculum_server.stats()
        assert stats["tasks_created"] > curriculum_server.num_slots

    def test_concurrent_access(self, curriculum_server):
        """Test concurrent access to the same slot."""
        import threading

        client = CurriculumClient(num_slots=10)
        results = []

        def worker(worker_id):
            try:
                task = client.get_task()
                time.sleep(0.01)  # Simulate work
                task.complete(0.5 + worker_id * 0.1)
                results.append((worker_id, True))
            except Exception as e:
                results.append((worker_id, False, str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check that all workers succeeded
        assert len(results) == 5
        for result in results:
            assert result[1], f"Worker {result[0]} failed"

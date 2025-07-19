#!/usr/bin/env python3
"""Test curriculum server and client functionality."""

import time

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer


class MockCurriculum(Curriculum):
    """Simple test curriculum that returns tasks with different configs."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create(
            {"game": {"width": 10 + self.task_count, "height": 10 + self.task_count, "num_agents": 2}}
        )
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))

    def get_completion_rates(self) -> dict[str, float]:
        return {"task_completions/test": 0.75}

    def get_task_probs(self) -> dict[str, float]:
        return {"test": 1.0}

    def stats(self) -> dict:
        return {"total_tasks": self.task_count, "completed_tasks": len(self.completed_tasks)}


def test_curriculum_server_client(free_port):
    """Test basic server-client functionality."""
    # Create test curriculum
    curriculum = MockCurriculum()

    # Start server
    server = CurriculumServer(curriculum, port=free_port)
    server.start()

    # Give server time to start
    time.sleep(0.5)

    try:
        # Create client
        client = CurriculumClient(server_url=f"http://127.0.0.1:{free_port}", batch_size=5)

        # Test getting tasks
        tasks_received = []
        for _ in range(10):
            task = client.get_task()
            assert task is not None
            # Task names come from the MockCurriculum (e.g., "task_1", "task_2", etc.)
            assert task.name().startswith("task_")
            assert hasattr(task, "env_cfg")
            env_cfg = task.env_cfg()
            assert "game" in env_cfg
            assert "width" in env_cfg.game
            tasks_received.append(task.name())

        # Tasks should come from batches (batch size is 5)
        unique_tasks = set(tasks_received)
        assert len(unique_tasks) == 10  # Should have exactly 10 different tasks

        # Test that complete_task is a no-op on client
        client.complete_task("1", 0.8)  # Should not raise

        # Test stats methods (should return empty dicts)
        assert client.stats() == {}

    finally:
        # Clean up
        client.stop()
        server.stop()


def test_curriculum_server_batch_sizes(free_port):
    """Test different batch sizes."""
    curriculum = MockCurriculum()
    server = CurriculumServer(curriculum, port=free_port)
    server.start()
    time.sleep(0.5)

    try:
        # Test small batch size
        client = CurriculumClient(server_url=f"http://127.0.0.1:{free_port}", batch_size=2)

        # Get tasks - should trigger multiple fetches
        tasks = []
        for _ in range(5):
            task = client.get_task()
            tasks.append(task.name())

        # With batch size 2, we need 3 fetches for 5 tasks
        # Fetch 1: task_1, task_2
        # Fetch 2: task_3, task_4
        # Fetch 3: task_5, task_6
        unique_tasks = set(tasks)
        assert len(unique_tasks) == 5  # Should get exactly 5 different tasks

    finally:
        client.stop()
        server.stop()


def test_curriculum_server_error_handling():
    """Test error handling when server is not available."""
    # Try to connect to non-existent server
    with pytest.raises(RuntimeError) as exc_info:
        CurriculumClient(server_url="http://127.0.0.1:19999", batch_size=5, max_retries=2)

    assert "Failed to connect to curriculum server" in str(exc_info.value)


if __name__ == "__main__":
    import random

    # Use random ports to avoid conflicts
    port1 = random.randint(20000, 30000)
    port2 = random.randint(30001, 40000)

    test_curriculum_server_client(port1)
    test_curriculum_server_batch_sizes(port2)
    test_curriculum_server_error_handling()
    print("All tests passed!")

#!/usr/bin/env python3
"""Test curriculum server and client functionality."""

import time

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum_client import CurriculumClient
from metta.rl.curriculum_server import CurriculumServer


class MockCurriculum(Curriculum):
    """Simple test curriculum that returns tasks with different configs."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create({
            "game": {
                "width": 10 + self.task_count,
                "height": 10 + self.task_count,
                "num_agents": 2
            }
        })
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))

    def get_completion_rates(self) -> dict[str, float]:
        return {"task_completions/test": 0.75}

    def get_task_probs(self) -> dict[str, float]:
        return {"test": 1.0}

    def get_curriculum_stats(self) -> dict:
        return {
            "total_tasks": self.task_count,
            "completed_tasks": len(self.completed_tasks)
        }


def test_curriculum_server_client():
    """Test basic server-client functionality."""
    # Create test curriculum
    curriculum = MockCurriculum()

    # Start server
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15555)
    server.start(background=True)

    # Give server time to start
    time.sleep(0.5)

    try:
        # Create client
        client = CurriculumClient(
            server_url="http://127.0.0.1:15555",
            batch_size=5
        )

        # Test getting tasks
        tasks_received = []
        for _ in range(10):
            task = client.get_task()
            assert task is not None
            assert task.name.startswith("task_")
            assert hasattr(task, "env_cfg")
            env_cfg = task.env_cfg()
            assert "game" in env_cfg
            assert "width" in env_cfg.game
            tasks_received.append(task.name)

        # Should have received tasks from first batch (task_1 through task_5)
        # Since client randomly selects, we can't predict exact order
        unique_tasks = set(tasks_received)
        assert len(unique_tasks) <= 5  # At most 5 unique tasks from first batch

        # Test that complete_task is a no-op on client
        client.complete_task("task_1", 0.8)  # Should not raise

        # Test compatibility methods
        assert client.get_completion_rates() == {}
        assert client.get_task_probs() == {}
        assert client.get_curriculum_stats() == {}

    finally:
        # Stop server
        server.stop()


def test_curriculum_server_batch_sizes():
    """Test different batch sizes."""
    curriculum = MockCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15556)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Test small batch size
        client = CurriculumClient(
            server_url="http://127.0.0.1:15556",
            batch_size=2
        )

        # Get tasks - should trigger multiple fetches
        tasks = []
        for _ in range(5):
            task = client.get_task()
            tasks.append(task.name)

        # With batch size 2, we should see tasks from multiple batches
        # task_1, task_2 from first batch
        # task_3, task_4 from second batch
        # task_5, task_6 from third batch
        unique_tasks = set(tasks)
        assert len(unique_tasks) <= 6  # Could be from up to 3 batches

    finally:
        server.stop()


def test_curriculum_server_error_handling():
    """Test error handling when server is not available."""
    # Try to connect to non-existent server
    client = CurriculumClient(
        server_url="http://127.0.0.1:19999",
        batch_size=5,
        max_retries=2,
        retry_delay=0.1
    )

    # Should raise after retries
    with pytest.raises(RuntimeError) as exc_info:
        client.get_task()

    assert "Failed to fetch tasks" in str(exc_info.value)


if __name__ == "__main__":
    test_curriculum_server_client()
    test_curriculum_server_batch_sizes()
    test_curriculum_server_error_handling()
    print("All tests passed!")

#!/usr/bin/env python3
"""Test curriculum server integration with trainer features."""

import time

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer


class TrackedCurriculum(Curriculum):
    """Curriculum that tracks method calls for testing."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []
        self.completion_rates = {"task_completions/task_1": 0.5, "task_completions/task_2": 0.3}
        self.task_probs = {"task_1": 0.6, "task_2": 0.4}
        self.curriculum_stats = {"test_stat": 42, "another_stat": 3.14}

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{(self.task_count - 1) % 2 + 1}"  # Alternate between task_1 and task_2
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
        # Update completion rates
        if id in self.completion_rates:
            self.completion_rates[f"task_completions/{id}"] = score

    def get_completion_rates(self) -> dict[str, float]:
        return self.completion_rates

    def get_task_probs(self) -> dict[str, float]:
        return self.task_probs

    def get_curriculum_stats(self) -> dict:
        return self.curriculum_stats


def test_curriculum_stats_collection():
    """Test that curriculum stats are properly collected and accessible."""
    # Create curriculum with stats
    curriculum = TrackedCurriculum()

    # Test local curriculum stats
    assert curriculum.get_curriculum_stats()["test_stat"] == 42
    assert curriculum.get_task_probs()["task_1"] == 0.6
    assert curriculum.get_completion_rates()["task_completions/task_1"] == 0.5

    # Start server
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15557)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Create client
        client = CurriculumClient(
            server_url="http://127.0.0.1:15557",
            batch_size=5
        )

        # Client should return empty stats (all handled by server)
        assert client.get_curriculum_stats() == {}
        assert client.get_task_probs() == {}
        assert client.get_completion_rates() == {}

        # Get some tasks
        tasks = []
        for _ in range(10):
            task = client.get_task()
            tasks.append(task)
            # Complete should be no-op
            task.complete(0.9)

        # Client stats should still be empty
        assert client.get_curriculum_stats() == {}

        # Server-side curriculum should have tracked the gets
        # With background prefetching, may fetch multiple batches
        assert curriculum.task_count >= 5  # At least one batch of 5

    finally:
        client.stop()
        server.stop()


def test_multiple_clients():
    """Test multiple clients connecting to the same server."""
    curriculum = TrackedCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15558)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Create multiple clients
        clients = []
        for i in range(3):
            client = CurriculumClient(
                server_url="http://127.0.0.1:15558",
                batch_size=2
            )
            clients.append(client)

        # Each client gets tasks
        all_tasks = []
        for client in clients:
            for _ in range(4):
                task = client.get_task()
                all_tasks.append(task.name)

        # With background prefetching, clients may fetch multiple batches
        # So we expect at least 3 clients * 1 batch each = 3 * 2 = 6 tasks
        assert curriculum.task_count >= 6

        # But we should have gotten 12 task selections (4 per client)
        assert len(all_tasks) == 12

    finally:
        # Clean up clients
        for client in clients:
            client.stop()
        server.stop()


def test_learning_progress_curriculum():
    """Test that complex curricula like LearningProgress work with server/client."""
    # Create a custom curriculum that tracks learning progress
    class LearningProgressMock(Curriculum):
        def __init__(self):
            self.task_names = ["task_1", "task_2", "task_3"]
            self.task_weights = {name: 1.0 for name in self.task_names}
            self.task_count = 0
            self.completed = {}

        def get_task(self) -> Task:
            # Simple weighted random selection
            import random
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values())
            )[0]
            self.task_count += 1
            env_cfg = OmegaConf.create({"game": {"task_id": task_name}})
            return Task(task_name, self, env_cfg)

        def complete_task(self, id: str, score: float):
            self.completed[id] = score
            # Simulate learning progress - reduce weight of completed tasks
            if id in self.task_weights:
                self.task_weights[id] *= 0.8

        def get_curriculum_stats(self) -> dict:
            return {
                "total_tasks": self.task_count,
                "completed_tasks": len(self.completed),
                "avg_score": sum(self.completed.values()) / len(self.completed) if self.completed else 0
            }

    lp_curriculum = LearningProgressMock()

    # Start server
    server = CurriculumServer(lp_curriculum, host="127.0.0.1", port=15560)
    server.start(background=True)
    time.sleep(0.5)

    try:
        client = CurriculumClient(
            server_url="http://127.0.0.1:15560",
            batch_size=10
        )

        # Get multiple tasks
        task_names = []
        for _ in range(20):
            task = client.get_task()
            task_names.append(task.name)

        # Should have gotten tasks from the curriculum
        assert len(task_names) == 20
        # Tasks should be from our defined set
        for name in task_names:
            assert name in ["task_1", "task_2", "task_3"]

        # Complete some tasks on the server side
        lp_curriculum.complete_task("task_1", 0.9)
        lp_curriculum.complete_task("task_2", 0.5)

        # Get stats from server
        stats = lp_curriculum.get_curriculum_stats()
        assert "total_tasks" in stats
        assert stats["completed_tasks"] == 2
        assert stats["avg_score"] == 0.7  # (0.9 + 0.5) / 2

    finally:
        server.stop()


def test_server_shutdown():
    """Test clean server shutdown."""
    curriculum = TrackedCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15561)
    server.start(background=True)
    time.sleep(0.5)

    # Create client and get task
    client = CurriculumClient(
        server_url="http://127.0.0.1:15561",
        batch_size=5
    )
    task = client.get_task()
    assert task is not None

    # Stop server
    server.stop()
    time.sleep(0.5)

    # Clear the client's queue to force a fetch
    while not client._task_queue.empty():
        try:
            client._task_queue.get_nowait()
        except:
            break

    # Now client should fail when trying to get a task
    with pytest.raises(RuntimeError) as exc_info:
        client.get_task()

    assert "Failed to fetch tasks" in str(exc_info.value) or "Failed to get task from queue" in str(exc_info.value)

    # Clean up
    client.stop()


if __name__ == "__main__":
    test_curriculum_stats_collection()
    test_multiple_clients()
    test_learning_progress_curriculum()
    test_server_shutdown()
    print("All integration tests passed!")

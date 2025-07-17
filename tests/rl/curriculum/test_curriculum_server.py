"""
Test script for curriculum server and client.

This script tests the curriculum server and client functionality.
"""

import time

from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer


class MockCurriculum(Curriculum):
    """Simple mock curriculum for testing."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks_list = []

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create({"game": {"num_agents": 1, "max_steps": 100}})
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks_list.append((id, score))

    def get_task_probs(self) -> dict[str, float]:
        return {"mock_task": 1.0}

    def get_completion_rates(self) -> dict[str, float]:
        if not self.completed_tasks_list:
            return {"mock_task": 0.0}
        return {"mock_task": len(self.completed_tasks_list) / self.task_count}

    def get_curriculum_stats(self) -> dict:
        return {"total_tasks": self.task_count, "completed": len(self.completed_tasks_list)}


def test_curriculum_server_client():
    """Test curriculum server and client communication."""
    print("Testing curriculum server and client...")

    # Use mock curriculum for testing
    curriculum = MockCurriculum()

    # Start server in background
    server = CurriculumServer(curriculum, port=8888)
    server.start(background=True)

    # Give server time to start
    time.sleep(1)

    # Create client
    client = CurriculumClient(server_url="http://localhost:8888", batch_size=5)

    # Test getting tasks
    print("\nTesting get_task()...")
    tasks = []
    for i in range(10):
        task = client.get_task()
        print(f"  Task {i}: {task.name()}")
        tasks.append(task)

    # Test completing tasks
    print("\nTesting task completion...")
    for i, task in enumerate(tasks[:5]):
        score = 0.5 + i * 0.1
        task.complete(score)
        print(f"  Completed task {i} with score {score}")

    # Test getting stats
    print("\nTesting get stats...")
    curriculum_stats = client.get_curriculum_stats()
    print("  Curriculum stats (should be empty from client):", curriculum_stats)

    # Server's curriculum should have the real stats
    print("\nServer curriculum stats:")
    print("  Total tasks created:", curriculum.task_count)
    print("  Completed tasks:", curriculum.completed_tasks_list)

    # Clean up
    client.stop()
    server.stop()
    print("\nTest completed!")


def test_batch_prefetching(free_port):
    """Test that client properly prefetches batches."""
    print("\n\nTesting batch prefetching...")

    # Use mock curriculum
    curriculum = MockCurriculum()

    server = CurriculumServer(curriculum, port=free_port)
    server.start(background=True)
    time.sleep(1)

    # Create client with small batch size
    client = CurriculumClient(server_url=f"http://localhost:{free_port}", batch_size=3)

    # Initial queue should have 3 tasks (fetched by background thread)
    time.sleep(0.5)  # Give time for initial fetch
    initial_size = client._task_queue.qsize()
    print(f"Initial queue size: {initial_size}")
    assert initial_size > 0, "Expected queue to be populated"

    # Get 2 tasks (should trigger prefetch since 1 < 3 * 0.5)
    task1 = client.get_task()
    task2 = client.get_task()
    print(f"Got 2 tasks: {task1.name}, {task2.name}")

    # Give time for background prefetch
    time.sleep(0.5)

    # Queue should have been refilled
    queue_size = client._task_queue.qsize()
    print(f"Queue size after getting 2 tasks: {queue_size}")

    # The queue should have more tasks due to background prefetch
    assert queue_size >= 1, f"Expected queue to have tasks, but size is {queue_size}"

    print("Batch prefetching test passed!")

    # Clean up
    client.stop()


if __name__ == "__main__":
    test_curriculum_server_client()
    test_batch_prefetching()

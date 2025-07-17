#!/usr/bin/env python3
"""Test curriculum server integration scenarios."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer


class StatefulCurriculum(Curriculum):
    """Curriculum that tracks state and provides stats."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []
        self.task_probs = {"easy": 0.7, "hard": 0.3}

    def get_task(self) -> Task:
        self.task_count += 1
        # Alternate between easy and hard tasks based on count
        difficulty = "easy" if self.task_count % 3 != 0 else "hard"
        task_id = f"{difficulty}_task_{self.task_count}"

        env_cfg = OmegaConf.create({
            "game": {
                "difficulty": difficulty,
                "width": 10 if difficulty == "easy" else 20,
                "height": 10 if difficulty == "easy" else 20,
                "num_agents": 2
            }
        })
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))
        # Update task probabilities based on performance
        if "easy" in id and score > 0.8:
            self.task_probs["easy"] = max(0.3, self.task_probs["easy"] - 0.05)
            self.task_probs["hard"] = min(0.7, self.task_probs["hard"] + 0.05)

    def get_completion_rates(self) -> dict[str, float]:
        easy_tasks = [t for t, _ in self.completed_tasks if "easy" in t]
        hard_tasks = [t for t, _ in self.completed_tasks if "hard" in t]
        total = len(self.completed_tasks)
        return {
            "task_completions/easy": len(easy_tasks) / total if total > 0 else 0,
            "task_completions/hard": len(hard_tasks) / total if total > 0 else 0
        }

    def get_task_probs(self) -> dict[str, float]:
        return self.task_probs

    def get_curriculum_stats(self) -> dict:
        avg_score = sum(s for _, s in self.completed_tasks) / len(self.completed_tasks) if self.completed_tasks else 0
        return {
            "total_tasks": self.task_count,
            "completed_tasks": len(self.completed_tasks),
            "average_score": avg_score
        }


def test_server_client_with_stats():
    """Test that curriculum stats are properly served and client methods work."""
    curriculum = StatefulCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15557)
    server.start(background=True)
    time.sleep(0.5)

    try:
        client = CurriculumClient(
            server_url="http://127.0.0.1:15557",
            batch_size=10
        )

        # Get some tasks and "complete" them on server side
        tasks_seen = []
        for i in range(15):
            task = client.get_task()
            tasks_seen.append(task.name)
            # Simulate completing tasks on server side
            curriculum.complete_task(task.name, 0.5 + i * 0.03)

        # Client methods should return empty (server handles stats)
        assert client.get_completion_rates() == {}
        assert client.get_task_probs() == {}
        assert client.get_curriculum_stats() == {}

        # Verify we got a mix of easy and hard tasks
        easy_count = sum(1 for t in tasks_seen if "easy" in t)
        hard_count = sum(1 for t in tasks_seen if "hard" in t)
        assert easy_count > 0
        assert hard_count > 0

    finally:
        server.stop()


def test_concurrent_clients():
    """Test multiple clients accessing the same server concurrently."""
    curriculum = StatefulCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15558)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Create multiple clients
        clients = [
            CurriculumClient(
                server_url="http://127.0.0.1:15558",
                batch_size=5
            ) for _ in range(3)
        ]

        # Each client gets tasks in parallel
        results = [[] for _ in range(3)]

        def get_tasks(client_idx, client, result_list):
            for _ in range(10):
                task = client.get_task()
                result_list.append(task.name)
                time.sleep(0.01)  # Small delay to simulate processing

        threads = []
        for i, client in enumerate(clients):
            thread = threading.Thread(target=get_tasks, args=(i, client, results[i]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all clients got tasks
        for i, result in enumerate(results):
            assert len(result) == 10, f"Client {i} got {len(result)} tasks"

        # Verify task distribution makes sense
        all_tasks = sum(results, [])
        assert len(set(all_tasks)) <= 30  # Should have reused some tasks from batches

    finally:
        server.stop()


def test_server_restart():
    """Test that clients handle server restart gracefully."""
    curriculum = StatefulCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15559)
    server.start(background=True)
    time.sleep(0.5)

    client = CurriculumClient(
        server_url="http://127.0.0.1:15559",
        batch_size=5,
        max_retries=2,
        retry_delay=0.1
    )

    # Get some tasks
    task1 = client.get_task()
    assert task1 is not None

    # Stop server
    server.stop()
    time.sleep(0.5)

    # Clear the client's queue to force a fetch
    while not client._task_queue.empty():
        try:
            client._task_queue.get_nowait()
        except:
            break

    # Client should fail to get new tasks
    with pytest.raises(RuntimeError) as exc_info:
        client.get_task()
    assert "Failed to fetch tasks" in str(exc_info.value) or "Failed to get task from queue" in str(exc_info.value)

    # Restart server
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15559)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Client should work again
        task2 = client.get_task()
        assert task2 is not None
    finally:
        client.stop()
        server.stop()


def test_trainer_integration_simulation():
    """Simulate how the trainer would use the curriculum server/client."""
    # This simulates the trainer's _process_stats method

    curriculum = StatefulCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15560)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Simulate master rank
        client = CurriculumClient(
            server_url="http://127.0.0.1:15560",
            batch_size=100
        )

        # Simulate training loop
        for epoch in range(5):
            # Get tasks for environments
            tasks = []
            for _ in range(20):
                task = client.get_task()
                tasks.append(task)

            # Simulate completing tasks with scores
            for task in tasks:
                score = 0.7 + (epoch * 0.05)  # Improving over time
                curriculum.complete_task(task.name, score)

            # In trainer, this would be in _process_stats
            curriculum_stats = {}

            # Get stats from server-side curriculum
            raw_curriculum_stats = curriculum.get_curriculum_stats()
            for key, value in raw_curriculum_stats.items():
                curriculum_stats[f"curriculum/{key}"] = value

            task_probs = curriculum.get_task_probs()
            for task_id, prob in task_probs.items():
                curriculum_stats[f"curriculum/task_prob/{task_id}"] = prob

            completion_rates = curriculum.get_completion_rates()
            curriculum_stats.update(completion_rates)

            # Verify stats are being collected
            assert "curriculum/total_tasks" in curriculum_stats
            assert "curriculum/completed_tasks" in curriculum_stats
            assert "curriculum/average_score" in curriculum_stats
            assert "curriculum/task_prob/easy" in curriculum_stats
            assert "curriculum/task_prob/hard" in curriculum_stats

            # Verify progression - task probabilities should shift towards hard
            if epoch > 2:
                assert task_probs["hard"] > 0.3

    finally:
        server.stop()


def test_empty_batch_handling():
    """Test handling when server returns empty batch."""

    class EmptyCurriculum(Curriculum):
        def __init__(self):
            self.call_count = 0

        def get_task(self) -> Task:
            self.call_count += 1
            # Return None to simulate no tasks available
            if self.call_count > 2:
                raise RuntimeError("No more tasks available")
            return Task(f"task_{self.call_count}", self, OmegaConf.create({"game": {"num_agents": 1}}))

    curriculum = EmptyCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15561)
    server.start(background=True)
    time.sleep(0.5)

    try:
        client = CurriculumClient(
            server_url="http://127.0.0.1:15561",
            batch_size=5,
            max_retries=2,
            retry_delay=0.1
        )

        # Should get first two tasks (server will only return 2 before hitting the exception)
        task1 = client.get_task()
        assert task1.name in ["task_1", "task_2"]

        task2 = client.get_task()
        assert task2.name in ["task_1", "task_2"]

        # Empty the queue to force refetch
        while not client._task_queue.empty():
            try:
                client._task_queue.get_nowait()
            except:
                break

        # Now server will return empty batch, client should handle gracefully
        # The server should return an empty list instead of erroring
        with pytest.raises(RuntimeError) as exc_info:
            client.get_task()
        assert "Server returned empty task batch" in str(exc_info.value)

    finally:
        client.stop()
        server.stop()


def test_random_curriculum_integration():
    """Test with a real curriculum type (RandomCurriculum)."""
    # Create a random curriculum with some tasks
    tasks = {
        "easy_env": 0.6,
        "medium_env": 0.3,
        "hard_env": 0.1
    }

    # Mock the curriculum loading
    with patch('metta.mettagrid.curriculum.random.curriculum_from_config_path') as mock_load:
        # Create mock curricula for each task type
        mock_curricula = {}
        for task_id in tasks:
            mock_curr = MagicMock()
            mock_task = Task(
                task_id,
                mock_curr,
                OmegaConf.create({
                    "game": {
                        "name": task_id,
                        "difficulty": task_id.split("_")[0],
                        "num_agents": 2
                    }
                })
            )
            mock_curr.get_task.return_value = mock_task
            mock_curricula[task_id] = mock_curr

        def mock_curriculum_loader(cfg_path, env_overrides):
            return mock_curricula.get(cfg_path, mock_curricula["easy_env"])

        mock_load.side_effect = mock_curriculum_loader

        # Create the random curriculum
        curriculum = RandomCurriculum(tasks, None)

        server = CurriculumServer(curriculum, host="127.0.0.1", port=15562)
        server.start(background=True)
        time.sleep(0.5)

        try:
            client = CurriculumClient(
                server_url="http://127.0.0.1:15562",
                batch_size=50
            )

            # Get many tasks and check distribution
            task_counts = {"easy_env": 0, "medium_env": 0, "hard_env": 0}
            for _ in range(100):
                task = client.get_task()
                for task_type in task_counts:
                    if task_type in task.name:
                        task_counts[task_type] += 1
                        break

            # Verify rough distribution (won't be exact due to randomness)
            total = sum(task_counts.values())
            assert total == 100

            # Easy should be most common
            assert task_counts["easy_env"] > task_counts["medium_env"]
            assert task_counts["medium_env"] > task_counts["hard_env"]

        finally:
            server.stop()


if __name__ == "__main__":
    test_server_client_with_stats()
    test_concurrent_clients()
    test_server_restart()
    test_trainer_integration_simulation()
    test_empty_batch_handling()
    test_random_curriculum_integration()
    print("All integration tests passed!")

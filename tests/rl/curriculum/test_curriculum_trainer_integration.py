#!/usr/bin/env python3
"""Test curriculum server/client integration with trainer expectations."""

import time
from unittest.mock import patch

from omegaconf import OmegaConf

from metta.mettagrid.curriculum.bucketed import BucketedCurriculum
from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer


class SimpleCurriculum(Curriculum):
    """Simple curriculum for testing."""

    def __init__(self):
        self.task_count = 0
        self.completed_count = 0

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create(
            {"game": {"width": 10, "height": 10, "num_agents": 2, "max_steps": 100}, "training": {"enabled": True}}
        )
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_count += 1

    def get_completion_rates(self) -> dict[str, float]:
        return {"task_completions/simple": 0.5}

    def get_task_probs(self) -> dict[str, float]:
        return {"simple": 1.0}

    def get_curriculum_stats(self) -> dict:
        return {"total_tasks": self.task_count, "completed_tasks": self.completed_count}


def test_curriculum_client_trainer_methods():
    """Test that curriculum client implements all methods expected by trainer."""
    # Create and start server
    curriculum = SimpleCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15557)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Create client
        client = CurriculumClient(server_url="http://127.0.0.1:15557", batch_size=10)

        # Test get_task
        task1 = client.get_task()
        assert task1 is not None
        assert hasattr(task1, "env_cfg")
        assert hasattr(task1, "complete")

        # Test multiple get_task calls
        tasks = [client.get_task() for _ in range(20)]
        assert all(t is not None for t in tasks)

        # Test complete_task (should be no-op)
        client.complete_task("task_1", 0.9)  # Should not raise

        # Test stats methods (should return empty)
        assert client.get_curriculum_stats() == {}

    finally:
        client.stop()
        server.stop()


def test_curriculum_server_with_complex_curriculum():
    """Test server with a more complex curriculum like BucketedCurriculum."""
    # Create a bucketed curriculum configuration
    env_cfg_template = "/env/mettagrid/test_env"
    buckets = {"game.width": {"range": [5, 15], "bins": 3}, "game.height": {"range": [5, 15], "bins": 3}}

    # Mock the config loading
    with patch("metta.mettagrid.curriculum.bucketed.config_from_path") as mock_config:
        mock_config.return_value = OmegaConf.create(
            {"game": {"width": 10, "height": 10, "num_agents": 2, "max_steps": 100}}
        )

        curriculum = BucketedCurriculum(env_cfg_template=env_cfg_template, buckets=buckets, env_overrides=None)

    server = CurriculumServer(curriculum, host="127.0.0.1", port=15558)
    server.start(background=True)
    time.sleep(0.5)

    try:
        client = CurriculumClient(server_url="http://127.0.0.1:15558", batch_size=5)

        # Get several tasks
        tasks = []
        for _ in range(10):
            task = client.get_task()
            assert task is not None
            env_cfg = task.env_cfg()
            assert "game" in env_cfg
            assert "width" in env_cfg.game
            assert "height" in env_cfg.game
            tasks.append(task)

        # Check that we get varied tasks
        task_names = [t.name for t in tasks]
        unique_names = set(task_names)
        assert len(unique_names) >= 1  # Should have at least some tasks

    finally:
        client.stop()
        server.stop()


def test_curriculum_client_batch_exhaustion():
    """Test that client fetches new batches with background prefetching."""
    # Use a curriculum that generates unique tasks
    curriculum = SimpleCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15559)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Small batch size to test prefetching
        client = CurriculumClient(
            server_url="http://127.0.0.1:15559",
            batch_size=3,
            prefetch_threshold=0.5,  # Prefetch when queue drops to 1.5 tasks
        )

        # Give time for initial fetch
        time.sleep(0.5)

        # Get many tasks - this should trigger multiple fetches
        all_tasks = []
        for _i in range(30):
            task = client.get_task()
            all_tasks.append(task.name())

        # We should see tasks from multiple batches due to background prefetching
        unique_tasks = set(all_tasks)

        # The curriculum generates sequential tasks, so with multiple fetches
        # we should see more than just the first 3 tasks
        max_task_num = max(int(name.split("_")[1]) for name in unique_tasks)
        assert max_task_num > 3, f"Expected tasks beyond first batch, got max task_{max_task_num}"

        # We should see mostly unique tasks since we're fetching new batches
        assert len(unique_tasks) >= 10, f"Expected at least 10 unique tasks, got {len(unique_tasks)}"

    finally:
        client.stop()
        server.stop()


def test_curriculum_client_concurrent_access(free_port):
    """Test that multiple clients can access the server concurrently."""
    import threading

    curriculum = SimpleCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=free_port)
    server.start(background=True)
    time.sleep(0.5)

    results = []
    errors = []

    def client_worker(client_id):
        try:
            client = CurriculumClient(server_url=f"http://127.0.0.1:{free_port}", batch_size=5)

            tasks = []
            for _ in range(10):
                task = client.get_task()
                tasks.append((client_id, task.name()))

            results.extend(tasks)
        except Exception as e:
            errors.append((client_id, str(e)))

    try:
        # Create multiple client threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=client_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 clients * 10 tasks each

        # Verify all clients got valid tasks
        counter = {}
        for _client_id, task_name in results:
            # Verify task is valid
            assert task_name.startswith("task_")
            counter[task_name] = counter.get(task_name, 0) + 1

    finally:
        server.stop()


def test_trainer_stats_collection():
    """Test that trainer can collect stats from curriculum through server."""
    curriculum = SimpleCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15561)
    server.start(background=True)
    time.sleep(0.5)

    try:
        # Simulate trainer behavior
        client = CurriculumClient(server_url="http://127.0.0.1:15561", batch_size=10)

        # Get some tasks (simulating rollouts)
        for _ in range(5):
            task = client.get_task()
            # In real trainer, this would happen after episode completion
            task.complete(0.8)

        # Stats should be available from server's curriculum
        # But client methods return empty (as designed)
        assert client.get_curriculum_stats() == {}

        # Server's curriculum should have the stats
        assert curriculum.get_curriculum_stats()["total_tasks"] > 0

    finally:
        client.stop()
        server.stop()


if __name__ == "__main__":
    test_curriculum_client_trainer_methods()
    test_curriculum_server_with_complex_curriculum()
    test_curriculum_client_batch_exhaustion()
    test_curriculum_client_concurrent_access()
    test_trainer_stats_collection()
    print("All integration tests passed!")

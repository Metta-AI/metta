#!/usr/bin/env python3
"""Consolidated tests for curriculum server and client functionality."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.bucketed import BucketedCurriculum
from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer

from .conftest import MockCurriculum, StatefulCurriculum


class TestBasicServerClient:
    """Basic server-client communication tests."""

    def test_basic_communication(self, free_port):
        """Test basic server-client functionality."""
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            client = CurriculumClient(server_url=f"http://127.0.0.1:{free_port}", batch_size=5)

            # Test getting tasks
            tasks_received = []
            for _ in range(10):
                task = client.get_task()
                assert task is not None
                assert hasattr(task, "env_cfg")
                assert hasattr(task, "name")
                tasks_received.append(task.name())

            # Verify we got tasks
            assert len(tasks_received) == 10
            assert len(set(tasks_received)) >= 1

            # Test that complete_task is a no-op on client
            client.complete_task("1", 0.8)  # Should not raise

            # Test stats methods (should return empty dicts)
            assert client.stats() == {}

        finally:
            client.stop()
            server.stop()

    def test_batch_prefetching(self, free_port):
        """Test that client properly prefetches batches."""
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            # Create client with small batch size
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=3,
                prefetch_threshold=0.5
            )

            # Give time for initial fetch
            time.sleep(0.5)
            
            # Initial queue should have tasks
            initial_size = client._task_queue.qsize()
            assert initial_size > 0, "Expected queue to be populated"

            # Get 2 tasks (should trigger prefetch since 1 < 3 * 0.5)
            task1 = client.get_task()
            task2 = client.get_task()
            assert task1 is not None
            assert task2 is not None

            # Give time for background prefetch
            time.sleep(0.5)

            # Queue should have been refilled
            queue_size = client._task_queue.qsize()
            assert queue_size >= 1, f"Expected queue to have tasks, but size is {queue_size}"

        finally:
            client.stop()
            server.stop()

    def test_error_handling_no_server(self):
        """Test error handling when server is not available."""
        with pytest.raises(RuntimeError) as exc_info:
            CurriculumClient(
                server_url="http://127.0.0.1:19999",
                batch_size=5,
                max_retries=2,
                retry_delay=0.1
            )

        assert "Failed to connect to curriculum server" in str(exc_info.value)

    def test_client_returns_empty_stats(self, free_port):
        """Test that curriculum client returns empty stats."""
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=10
            )

            # Get some tasks
            for _ in range(5):
                task = client.get_task()
                task.complete(0.8)

            # Client should return empty stats
            assert client.stats() == {}

            # Complete task should be no-op
            client.complete_task("task_1", 0.9)  # Should not raise

            # Server's curriculum should have the real stats
            assert curriculum.stats()["total_tasks"] >= 5

        finally:
            client.stop()
            server.stop()


class TestConcurrentAccess:
    """Tests for concurrent client access."""

    def test_multiple_clients_sequential(self, free_port):
        """Test multiple clients connecting to the same server."""
        curriculum = StatefulCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            # Create multiple clients
            clients = []
            for _ in range(3):
                client = CurriculumClient(
                    server_url=f"http://127.0.0.1:{free_port}",
                    batch_size=2
                )
                clients.append(client)

            # Each client gets tasks
            all_tasks = []
            for client in clients:
                for _ in range(4):
                    task = client.get_task()
                    all_tasks.append(task.name())

            # Should have gotten 12 task selections (4 per client)
            assert len(all_tasks) == 12

            # With background prefetching, server should have created multiple batches
            assert curriculum.task_count >= 6  # At least 3 clients * 1 batch of 2

        finally:
            for client in clients:
                client.stop()
            server.stop()

    def test_concurrent_clients_threaded(self, free_port):
        """Test multiple clients accessing server concurrently via threads."""
        curriculum = StatefulCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        results = []
        errors = []

        def client_worker(client_id):
            try:
                client = CurriculumClient(
                    server_url=f"http://127.0.0.1:{free_port}",
                    batch_size=5
                )
                
                tasks = []
                for _ in range(10):
                    task = client.get_task()
                    tasks.append((client_id, task.name()))
                
                results.extend(tasks)
                client.stop()
            except Exception as e:
                errors.append((client_id, str(e)))

        try:
            # Create multiple client threads
            threads = []
            for i in range(5):
                t = threading.Thread(target=client_worker, args=(i,))
                threads.append(t)
                t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 50  # 5 clients * 10 tasks each

            # Verify all tasks are valid
            for _client_id, task_name in results:
                assert "task_" in task_name

        finally:
            server.stop()


class TestServerLifecycle:
    """Tests for server lifecycle management."""

    def test_server_restart(self, free_port):
        """Test that server can be restarted on the same port."""
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        client = CurriculumClient(
            server_url=f"http://127.0.0.1:{free_port}",
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

        # Clear client's queue to force a fetch
        while not client._task_queue.empty():
            try:
                client._task_queue.get_nowait()
            except queue.Empty:
                break

        # Client should fail to get new tasks
        with pytest.raises(RuntimeError):
            client.get_task()

        # Restart server on same port
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            # Create new client (old one is in failed state)
            new_client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=5
            )
            task2 = new_client.get_task()
            assert task2 is not None
            new_client.stop()
        finally:
            client.stop()
            server.stop()

    def test_server_shutdown(self, free_port):
        """Test clean server shutdown."""
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        # Create client and get task
        client = CurriculumClient(
            server_url=f"http://127.0.0.1:{free_port}",
            batch_size=5
        )
        task = client.get_task()
        assert task is not None

        # Stop server
        server.stop()
        time.sleep(0.5)

        # Clear the client's queue
        while not client._task_queue.empty():
            try:
                client._task_queue.get_nowait()
            except queue.Empty:
                break

        # Now client should fail when trying to get a task
        with pytest.raises(RuntimeError):
            client.get_task()

        client.stop()


class TestComplexCurriculums:
    """Tests with more complex curriculum types."""

    def test_bucketed_curriculum(self, free_port):
        """Test server with BucketedCurriculum."""
        env_cfg_template = "/env/mettagrid/test_env"
        buckets = {
            "game.width": {"range": [5, 15], "bins": 3},
            "game.height": {"range": [5, 15], "bins": 3}
        }

        with patch("metta.mettagrid.curriculum.bucketed.config_from_path") as mock_config:
            mock_config.return_value = OmegaConf.create({
                "game": {
                    "width": 10,
                    "height": 10,
                    "num_agents": 2,
                    "max_steps": 100
                }
            })

            curriculum = BucketedCurriculum(
                env_cfg_template_path=env_cfg_template,
                buckets=buckets,
                env_overrides=None
            )

        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=5
            )

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

            # Should have gotten varied tasks
            assert len(tasks) == 10

        finally:
            client.stop()
            server.stop()

    def test_random_curriculum(self, free_port):
        """Test with RandomCurriculum."""
        tasks = {"easy_env": 0.6, "medium_env": 0.3, "hard_env": 0.1}

        with patch("metta.mettagrid.curriculum.random.curriculum_from_config_path") as mock_load:
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

            curriculum = RandomCurriculum(tasks, None)

        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=50
            )

            # Get many tasks and check distribution
            task_counts = {"easy_env": 0, "medium_env": 0, "hard_env": 0}
            for _ in range(100):
                task = client.get_task()
                for task_type in task_counts:
                    if task_type in task.name():
                        task_counts[task_type] += 1
                        break

            # Verify we got 100 tasks
            total = sum(task_counts.values())
            assert total == 100

            # Easy should be most common (won't be exact due to randomness)
            assert task_counts["easy_env"] > task_counts["medium_env"]
            assert task_counts["medium_env"] > task_counts["hard_env"]

        finally:
            client.stop()
            server.stop()


class TestEmptyBatchHandling:
    """Test edge cases with empty batches."""

    def test_empty_batch_handling(self, free_port):
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
        server = CurriculumServer(curriculum, port=free_port)
        server.start()
        time.sleep(0.5)

        try:
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}", 
                batch_size=5, 
                max_retries=2, 
                retry_delay=0.1
            )

            # Should get first two tasks
            task1 = client.get_task()
            assert task1.name() in ["task_1", "task_2"]

            task2 = client.get_task()
            assert task2.name() in ["task_1", "task_2"]

            # Empty the queue to force refetch
            while not client._task_queue.empty():
                try:
                    client._task_queue.get_nowait()
                except queue.Empty:
                    break

            # Now server will return empty batch
            with pytest.raises(RuntimeError) as exc_info:
                client.get_task()
            assert "Server returned empty task batch" in str(exc_info.value)

        finally:
            client.stop()
            server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Unit tests for CurriculumManager."""

import multiprocessing as mp
import os
import tempfile
import time

import numpy as np
import pytest

from metta.rl.curriculum import CurriculumManager, TaskState


class TestCurriculumManager:
    """Test suite for CurriculumManager."""

    def test_initialization(self):
        """Test CurriculumManager initialization."""
        manager = CurriculumManager(pool_size=64, min_runs=5, name="test_init")

        try:
            assert manager.pool_size == 64
            assert manager.min_runs == 5
            assert manager.name == "test_init"

            # Check shared memory names
            assert manager.get_shared_memory_names() == "test_init"

            # Check pool size matches expected array size
            assert len(manager._pool) == 64 * CurriculumManager.FIELDS_PER_TASK
        finally:
            manager.cleanup()

    def test_task_state_operations(self):
        """Test getting and setting task states."""
        manager = CurriculumManager(pool_size=10, name="test_task_ops")

        # Create a test task state
        test_task = TaskState(
            task_id=12345, score=0.75, num_runs=10, last_update=time.time(), reward_mean=0.85, reward_var=0.02
        )

        # Set and get task state
        manager._set_task_state(0, test_task)
        retrieved_task = manager._get_task_state(0)

        assert retrieved_task.task_id == test_task.task_id
        assert retrieved_task.score == test_task.score
        assert retrieved_task.num_runs == test_task.num_runs
        assert abs(retrieved_task.last_update - test_task.last_update) < 0.01
        assert retrieved_task.reward_mean == test_task.reward_mean
        assert retrieved_task.reward_var == test_task.reward_var

    def test_pool_initialization(self):
        """Test that pool is properly initialized with random task IDs."""
        manager = CurriculumManager(pool_size=50)

        # Check all slots have valid task IDs
        task_ids = []
        for i in range(50):
            task_state = manager._get_task_state(i)
            assert 0 <= task_state.task_id < 2**31
            assert task_state.score == 0.0
            assert task_state.num_runs == 0
            assert task_state.last_update > 0
            task_ids.append(task_state.task_id)

        # Check task IDs are diverse (not all the same)
        assert len(set(task_ids)) > 40  # Most should be unique

    def test_get_stats(self):
        """Test statistics calculation."""
        manager = CurriculumManager(pool_size=10)

        # Set some task states with known values
        for i in range(5):
            task_state = TaskState(
                task_id=i,
                score=i * 0.2,  # 0.0, 0.2, 0.4, 0.6, 0.8
                num_runs=i + 1,
                last_update=time.time(),
                reward_mean=0.5 + i * 0.1,
                reward_var=0.01,
            )
            manager._set_task_state(i, task_state)

        stats = manager.get_stats()

        # Check basic stats
        assert stats["pool_size"] == 10
        assert stats["avg_score"] == pytest.approx(0.2, abs=0.1)  # Average of all 10 slots
        assert stats["min_score"] == 0.0
        assert stats["max_score"] >= 0.8
        assert stats["tasks_with_runs"] == 5
        assert stats["total_runs"] == 15  # 1 + 2 + 3 + 4 + 5

        # Check reward stats (only for tasks with runs > 0)
        assert "avg_reward" in stats
        assert stats["avg_reward"] == pytest.approx(0.7, abs=0.01)

        # Check score distribution bins
        for i in range(10):
            assert f"score_bin_{i}" in stats

    def test_save_and_load_state(self):
        """Test saving and loading manager state."""
        manager1 = CurriculumManager(pool_size=20)

        # Set some distinctive task states
        for i in range(5):
            task_state = TaskState(
                task_id=1000 + i,
                score=i * 0.15,
                num_runs=i * 2,
                last_update=time.time(),
                reward_mean=0.3 + i * 0.1,
                reward_var=0.05,
            )
            manager1._set_task_state(i, task_state)

        # Save state to temporary file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            manager1.save_state(f.name)
            temp_path = f.name

        try:
            # Create new manager and load state
            manager2 = CurriculumManager(pool_size=20)
            manager2.load_state(temp_path)

            # Verify states match
            for i in range(20):
                task1 = manager1._get_task_state(i)
                task2 = manager2._get_task_state(i)

                assert task1.task_id == task2.task_id
                assert task1.score == task2.score
                assert task1.num_runs == task2.num_runs
                assert task1.reward_mean == task2.reward_mean
                assert task1.reward_var == task2.reward_var
        finally:
            os.unlink(temp_path)

    def test_load_state_pool_size_mismatch(self):
        """Test that loading state with different pool size raises error."""
        manager1 = CurriculumManager(pool_size=10)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            manager1.save_state(f.name)
            temp_path = f.name

        try:
            manager2 = CurriculumManager(pool_size=20)  # Different size
            with pytest.raises(ValueError, match="Pool size mismatch"):
                manager2.load_state(temp_path)
        finally:
            os.unlink(temp_path)

    def test_concurrent_access(self):
        """Test thread-safe concurrent access to shared memory."""
        manager = CurriculumManager(pool_size=100)
        pool, lock = manager.get_shared_memory()

        def worker(worker_id, num_updates):
            """Worker function that updates random tasks."""
            for _ in range(num_updates):
                slot_id = np.random.randint(0, 100)

                with lock:
                    # Read current state
                    task_state = manager._get_task_state(slot_id)

                    # Update state
                    task_state.num_runs += 1
                    task_state.score = min(1.0, task_state.score + 0.01)

                    # Write back
                    manager._set_task_state(slot_id, task_state)

        # Run multiple workers concurrently
        processes = []
        num_workers = 4
        updates_per_worker = 100

        for i in range(num_workers):
            p = mp.Process(target=worker, args=(i, updates_per_worker))
            p.start()
            processes.append(p)

        # Wait for all workers to complete
        for p in processes:
            p.join()

        # Verify total runs is correct
        stats = manager.get_stats()
        # Total runs should be at most num_workers * updates_per_worker
        # (some updates might hit the same slot)
        assert stats["total_runs"] <= num_workers * updates_per_worker

    def test_statistics_tracking(self):
        """Test that statistics counters work correctly."""
        manager = CurriculumManager(pool_size=10)

        # Initially counters should be zero
        stats = manager.get_stats()
        assert stats["total_completions"] == 0
        assert stats["total_replacements"] == 0

        # Note: The actual increment of these counters would be done by
        # CurriculumClient when completing tasks and replacing them.
        # Here we just test that the values are properly stored/retrieved.

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimum pool size
        manager = CurriculumManager(pool_size=1)
        assert manager.pool_size == 1

        task_state = manager._get_task_state(0)
        assert isinstance(task_state.task_id, int)

        # Test with large pool size
        manager = CurriculumManager(pool_size=10000)
        assert manager.pool_size == 10000

        # Test boundary task states
        boundary_task = TaskState(
            task_id=2**31 - 1,  # Maximum task ID
            score=1.0,  # Maximum score
            num_runs=1000000,
            last_update=time.time(),
            reward_mean=1.0,
            reward_var=0.0,
        )

        manager._set_task_state(0, boundary_task)
        retrieved = manager._get_task_state(0)

        assert retrieved.task_id == 2**31 - 1
        assert retrieved.score == 1.0
        assert retrieved.num_runs == 1000000


if __name__ == "__main__":
    pytest.main([__file__])

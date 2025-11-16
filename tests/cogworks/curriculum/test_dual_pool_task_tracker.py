"""Tests for DualPoolTaskTracker class.

This module tests the dual-pool task tracker infrastructure, including:
- Task creation in separate pools
- Task promotion from explore to exploit
- Pool-based task routing
- Shared memory management for dual pools
- State persistence and checkpointing
"""

import pytest

from metta.cogworks.curriculum.task_tracker import DualPoolTaskTracker


class TestDualPoolTaskTracker:
    """Test suite for DualPoolTaskTracker class."""

    @pytest.fixture
    def dual_tracker(self):
        """Create a dual-pool task tracker for testing."""
        tracker = DualPoolTaskTracker(
            num_explore_tasks=10,
            num_exploit_tasks=20,
            ema_alpha=0.1,
            session_id="test_dual",
            use_shared_memory=False,  # Use local memory for tests
            task_struct_size=18,
            default_success_threshold=0.5,
            default_generator_type=1.0,
        )
        yield tracker
        # Cleanup
        tracker.cleanup_shared_memory()

    def test_initialization(self, dual_tracker):
        """Test that dual-pool tracker initializes correctly."""
        assert dual_tracker.num_explore_tasks == 10
        assert dual_tracker.num_exploit_tasks == 20
        assert dual_tracker.explore_tracker is not None
        assert dual_tracker.exploit_tracker is not None
        assert len(dual_tracker._task_pool_map) == 0

    def test_track_task_creation_explore(self, dual_tracker):
        """Test creating a task in the explore pool."""
        dual_tracker.track_task_creation(task_id=1, pool="explore", success_threshold=0.6)

        # Task should be in explore pool
        assert dual_tracker._task_pool_map[1] == "explore"
        assert dual_tracker.get_pool_tracker(1) == dual_tracker.explore_tracker

        # Task should be tracked in explore tracker
        explore_tasks = dual_tracker.get_all_explore_tasks()
        assert 1 in explore_tasks

        # Task should not be in exploit pool
        exploit_tasks = dual_tracker.get_all_exploit_tasks()
        assert 1 not in exploit_tasks

    def test_track_task_creation_exploit(self, dual_tracker):
        """Test creating a task in the exploit pool."""
        dual_tracker.track_task_creation(task_id=2, pool="exploit", success_threshold=0.7)

        # Task should be in exploit pool
        assert dual_tracker._task_pool_map[2] == "exploit"
        assert dual_tracker.get_pool_tracker(2) == dual_tracker.exploit_tracker

        # Task should be tracked in exploit tracker
        exploit_tasks = dual_tracker.get_all_exploit_tasks()
        assert 2 in exploit_tasks

        # Task should not be in explore pool
        explore_tasks = dual_tracker.get_all_explore_tasks()
        assert 2 not in explore_tasks

    def test_track_task_creation_invalid_pool(self, dual_tracker):
        """Test that invalid pool name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pool"):
            dual_tracker.track_task_creation(task_id=3, pool="invalid")

    def test_update_task_performance_explore(self, dual_tracker):
        """Test updating performance for a task in explore pool."""
        dual_tracker.track_task_creation(task_id=1, pool="explore")

        # Update performance
        dual_tracker.update_task_performance(task_id=1, score=0.8)

        # Check stats were updated
        stats = dual_tracker.get_task_stats(1)
        assert stats is not None
        assert stats["completion_count"] == 1

    def test_update_task_performance_exploit(self, dual_tracker):
        """Test updating performance for a task in exploit pool."""
        dual_tracker.track_task_creation(task_id=2, pool="exploit")

        # Update performance
        dual_tracker.update_task_performance(task_id=2, score=0.7)

        # Check stats were updated
        stats = dual_tracker.get_task_stats(2)
        assert stats is not None
        assert stats["completion_count"] == 1

    def test_promote_task_basic(self, dual_tracker):
        """Test basic task promotion from explore to exploit."""
        # Create task in explore pool
        dual_tracker.track_task_creation(task_id=1, pool="explore")

        # Update performance to build history
        for _ in range(5):
            dual_tracker.update_task_performance(task_id=1, score=0.8)

        # Get stats before promotion
        stats_before = dual_tracker.get_task_stats(1)
        assert stats_before is not None

        # Promote task
        success = dual_tracker.promote_task(1)
        assert success is True

        # Task should now be in exploit pool
        assert dual_tracker._task_pool_map[1] == "exploit"
        assert dual_tracker.get_pool_tracker(1) == dual_tracker.exploit_tracker

        # Task should be in exploit pool
        exploit_tasks = dual_tracker.get_all_exploit_tasks()
        assert 1 in exploit_tasks

        # Task should not be in explore pool
        explore_tasks = dual_tracker.get_all_explore_tasks()
        assert 1 not in explore_tasks

        # Stats should be preserved after promotion
        stats_after = dual_tracker.get_task_stats(1)
        assert stats_after is not None
        assert stats_after["completion_count"] == stats_before["completion_count"]

    def test_promote_task_not_in_explore(self, dual_tracker):
        """Test that promoting a task not in explore pool raises ValueError."""
        # Create task in exploit pool
        dual_tracker.track_task_creation(task_id=1, pool="exploit")

        # Attempting to promote should raise ValueError
        with pytest.raises(ValueError, match="not in explore pool"):
            dual_tracker.promote_task(1)

    def test_promote_task_nonexistent(self, dual_tracker):
        """Test promoting a task that doesn't exist."""
        # Map says it's in explore, but tracker doesn't have it
        dual_tracker._task_pool_map[999] = "explore"

        # Should return False (not found in tracker)
        success = dual_tracker.promote_task(999)
        assert success is False

    def test_promote_task_exploit_pool_full(self, dual_tracker):
        """Test promotion fails when exploit pool is full."""
        # Fill exploit pool to capacity
        for i in range(20):  # num_exploit_tasks = 20
            dual_tracker.track_task_creation(task_id=100 + i, pool="exploit")

        # Create task in explore pool
        dual_tracker.track_task_creation(task_id=1, pool="explore")

        # Update performance
        for _ in range(5):
            dual_tracker.update_task_performance(task_id=1, score=0.8)

        # Promotion should fail (exploit pool is full)
        success = dual_tracker.promote_task(1)
        assert success is False

        # Task should still be in explore pool
        assert dual_tracker._task_pool_map[1] == "explore"

    def test_remove_task_from_explore(self, dual_tracker):
        """Test removing a task from explore pool."""
        dual_tracker.track_task_creation(task_id=1, pool="explore")

        # Task exists
        assert 1 in dual_tracker._task_pool_map

        # Remove task
        dual_tracker.remove_task(1)

        # Task should be gone
        assert 1 not in dual_tracker._task_pool_map
        assert dual_tracker.get_pool_tracker(1) is None

    def test_remove_task_from_exploit(self, dual_tracker):
        """Test removing a task from exploit pool."""
        dual_tracker.track_task_creation(task_id=2, pool="exploit")

        # Task exists
        assert 2 in dual_tracker._task_pool_map

        # Remove task
        dual_tracker.remove_task(2)

        # Task should be gone
        assert 2 not in dual_tracker._task_pool_map
        assert dual_tracker.get_pool_tracker(2) is None

    def test_set_and_get_task_label(self, dual_tracker):
        """Test setting and getting task labels."""
        dual_tracker.track_task_creation(task_id=1, pool="explore")

        # Set label
        dual_tracker.set_task_label(1, "test_label")

        # Get label
        label = dual_tracker.get_task_label(1)
        assert label == "test_label"

    def test_get_task_label_nonexistent(self, dual_tracker):
        """Test getting label for nonexistent task returns None."""
        label = dual_tracker.get_task_label(999)
        assert label is None

    def test_get_state_and_load_state(self, dual_tracker):
        """Test state persistence for dual-pool tracker."""
        # Create tasks in both pools
        dual_tracker.track_task_creation(task_id=1, pool="explore")
        dual_tracker.track_task_creation(task_id=2, pool="exploit")

        # Update performance
        dual_tracker.update_task_performance(task_id=1, score=0.8)
        dual_tracker.update_task_performance(task_id=2, score=0.7)

        # Get state
        state = dual_tracker.get_state()
        assert "explore_tracker" in state
        assert "exploit_tracker" in state
        assert "task_pool_map" in state
        assert state["task_pool_map"][1] == "explore"
        assert state["task_pool_map"][2] == "exploit"

        # Create new tracker and load state
        new_tracker = DualPoolTaskTracker(
            num_explore_tasks=10,
            num_exploit_tasks=20,
            ema_alpha=0.1,
            session_id="test_dual_2",
            use_shared_memory=False,
            task_struct_size=18,
            default_success_threshold=0.5,
            default_generator_type=1.0,
        )

        new_tracker.load_state(state)

        # Verify state was loaded correctly
        assert new_tracker._task_pool_map[1] == "explore"
        assert new_tracker._task_pool_map[2] == "exploit"
        assert 1 in new_tracker.get_all_explore_tasks()
        assert 2 in new_tracker.get_all_exploit_tasks()

        # Cleanup
        new_tracker.cleanup_shared_memory()

    def test_pickle_support(self, dual_tracker):
        """Test that dual-pool tracker can be pickled and unpickled."""
        import pickle

        # Create tasks
        dual_tracker.track_task_creation(task_id=1, pool="explore")
        dual_tracker.track_task_creation(task_id=2, pool="exploit")

        # Pickle and unpickle
        pickled = pickle.dumps(dual_tracker)
        restored = pickle.loads(pickled)

        # Verify state
        assert restored.num_explore_tasks == 10
        assert restored.num_exploit_tasks == 20
        assert restored._task_pool_map[1] == "explore"
        assert restored._task_pool_map[2] == "exploit"

        # Cleanup
        restored.cleanup_shared_memory()

    def test_multiple_tasks_in_each_pool(self, dual_tracker):
        """Test managing multiple tasks across both pools."""
        # Create 5 tasks in explore
        for i in range(5):
            dual_tracker.track_task_creation(task_id=i, pool="explore")

        # Create 10 tasks in exploit
        for i in range(10, 20):
            dual_tracker.track_task_creation(task_id=i, pool="exploit")

        # Check pool sizes
        explore_tasks = dual_tracker.get_all_explore_tasks()
        exploit_tasks = dual_tracker.get_all_exploit_tasks()

        assert len(explore_tasks) == 5
        assert len(exploit_tasks) == 10

        # Verify all explore tasks
        for i in range(5):
            assert i in explore_tasks
            assert dual_tracker._task_pool_map[i] == "explore"

        # Verify all exploit tasks
        for i in range(10, 20):
            assert i in exploit_tasks
            assert dual_tracker._task_pool_map[i] == "exploit"

    def test_shared_memory_cleanup(self, dual_tracker):
        """Test that shared memory cleanup doesn't raise errors."""
        dual_tracker.track_task_creation(task_id=1, pool="explore")
        dual_tracker.track_task_creation(task_id=2, pool="exploit")

        # Should not raise any errors
        dual_tracker.cleanup_shared_memory()


class TestDualPoolTaskTrackerSharedMemory:
    """Test suite for DualPoolTaskTracker with shared memory backend."""

    @pytest.fixture
    def dual_tracker_shared(self):
        """Create a dual-pool task tracker with shared memory for testing."""
        tracker = DualPoolTaskTracker(
            num_explore_tasks=10,
            num_exploit_tasks=20,
            ema_alpha=0.1,
            session_id="test_shared_dual",
            use_shared_memory=True,  # Use shared memory
            task_struct_size=18,
            default_success_threshold=0.5,
            default_generator_type=1.0,
        )
        yield tracker
        # Cleanup
        tracker.cleanup_shared_memory()

    def test_shared_memory_initialization(self, dual_tracker_shared):
        """Test that dual-pool tracker initializes with shared memory."""
        assert dual_tracker_shared.explore_tracker._backend.__class__.__name__ == "SharedMemoryBackend"
        assert dual_tracker_shared.exploit_tracker._backend.__class__.__name__ == "SharedMemoryBackend"

        # Check session IDs are suffixed correctly
        assert dual_tracker_shared.explore_tracker._backend.session_id.endswith("_explore")
        assert dual_tracker_shared.exploit_tracker._backend.session_id.endswith("_exploit")

    def test_shared_memory_task_promotion(self, dual_tracker_shared):
        """Test task promotion with shared memory backend."""
        # Create task in explore pool
        dual_tracker_shared.track_task_creation(task_id=1, pool="explore")

        # Update performance
        for _ in range(5):
            dual_tracker_shared.update_task_performance(task_id=1, score=0.8)

        # Promote task
        success = dual_tracker_shared.promote_task(1)
        assert success is True

        # Verify task is in exploit pool
        assert dual_tracker_shared._task_pool_map[1] == "exploit"
        exploit_tasks = dual_tracker_shared.get_all_exploit_tasks()
        assert 1 in exploit_tasks

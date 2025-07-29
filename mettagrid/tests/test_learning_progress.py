"""
Unit tests for learning progress curriculum with shared memory implementation.

Tests the BidirectionalLearningProgress class and LearningProgressCurriculum
to ensure shared memory functionality works correctly across processes.
"""

import multiprocessing as mp

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.curriculum.learning_progress import (
    BidirectionalLearningProgress,
    LearningProgressCurriculum,
)


class MockTask:
    """Mock task for testing curriculum behavior."""

    def __init__(self, task_id: str, cfg: DictConfig):
        self._id = task_id
        self._cfg = cfg
        self._complete = False

    def id(self):
        return self._id

    def short_name(self):
        return self._id.split(":")[0] if ":" in self._id else self._id

    def env_cfg(self):
        return self._cfg

    def is_complete(self):
        return self._complete

    def complete(self, score: float):
        self._complete = True


@pytest.fixture
def mock_env_cfg():
    """Create a mock environment configuration."""
    return OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})


@pytest.fixture
def mock_curriculum_from_config_path(monkeypatch, mock_env_cfg):
    """Mock the curriculum_from_config_path function."""

    def _mock(path, env_overrides=None):
        cfg = OmegaConf.merge(mock_env_cfg, env_overrides or {})
        return SingleTaskCurriculum(path, cfg)

    monkeypatch.setattr("metta.mettagrid.curriculum.random.curriculum_from_config_path", _mock)
    return _mock


class TestBidirectionalLearningProgressSharedMemory:
    """Test the shared memory implementation of BidirectionalLearningProgress."""

    def test_initialization(self):
        """Test that all shared memory structures are properly initialized."""
        lp = BidirectionalLearningProgress(
            search_space=4,
            ema_timescale=0.01,
            progress_smoothing=0.05,
            num_active_tasks=4,
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25,
        )

        # Check that shared memory values are accessible
        assert lp._num_tasks.value == 4
        assert lp._ema_timescale.value == 0.01
        assert lp.progress_smoothing.value == 0.05
        assert lp.num_active_tasks.value == 4
        assert lp._rand_task_rate.value == 0.25
        assert lp._sample_threshold.value == 10
        assert lp._memory.value == 25
        assert bool(lp._stale_dist.value)

        # Check that shared arrays are initialized
        assert len(lp._task_success_rate) == 4
        assert len(lp._update_mask) == 4
        assert len(lp._sample_levels) == 4

        # Check that update mask is all True initially
        update_mask = np.array(lp._update_mask[:])
        assert np.all(update_mask)

        # Check outcomes dict is initialized
        assert len(lp._outcomes) == 4
        for i in range(4):
            assert i in lp._outcomes
            assert len(lp._outcomes[i]) == 0

    def test_collect_data(self):
        """Test that data collection works with shared memory."""
        lp = BidirectionalLearningProgress(search_space=3)

        # Collect some data
        infos = {"tasks/0": [0.5, 0.6], "tasks/1": [0.7], "tasks/2": [0.0, 0.1, 0.2]}
        lp.collect_data(infos)

        # Check that data was stored in shared memory
        assert len(lp._outcomes[0]) == 2
        assert list(lp._outcomes[0]) == [0.5, 0.6]
        assert len(lp._outcomes[1]) == 1
        assert list(lp._outcomes[1]) == [0.7]
        assert len(lp._outcomes[2]) == 3
        assert list(lp._outcomes[2]) == [0.0, 0.1, 0.2]

    def test_update_with_shared_memory(self):
        """Test that update function works with shared memory arrays."""
        lp = BidirectionalLearningProgress(search_space=3, ema_timescale=0.1)

        # Add some outcomes
        lp._outcomes[0].extend([0.0, 0.1, 0.2])
        lp._outcomes[1].extend([0.5, 0.6])
        lp._outcomes[2].extend([1.0])

        # Run update
        task_success_rates = lp._update()

        # Check that shared arrays were created/updated
        assert lp._p_fast_shm is not None
        assert lp._p_slow_shm is not None
        assert lp._p_true_shm is not None
        assert lp._random_baseline_shm is not None

        # Check task success rates
        np.testing.assert_almost_equal(task_success_rates[0], 0.1)  # mean of [0.0, 0.1, 0.2]
        np.testing.assert_almost_equal(task_success_rates[1], 0.55)  # mean of [0.5, 0.6]
        np.testing.assert_almost_equal(task_success_rates[2], 1.0)  # mean of [1.0]

    def test_learning_progress_calculation(self):
        """Test learning progress calculation with shared memory."""
        lp = BidirectionalLearningProgress(search_space=3)

        # Manually set up some shared arrays
        p_fast = np.array([0.2, 0.5, 0.8])
        p_slow = np.array([0.1, 0.4, 0.7])

        lp._p_fast_shm = lp._create_or_update_shared_array("p_fast", p_fast)
        lp._p_slow_shm = lp._create_or_update_shared_array("p_slow", p_slow)

        # Calculate learning progress
        progress = lp._learning_progress(reweight=False)

        # Should be abs(fast - slow)
        expected = np.abs(p_fast - p_slow)
        np.testing.assert_array_almost_equal(progress, expected)

    def test_sample_distribution(self):
        """Test that sample distribution calculation works with shared memory."""
        lp = BidirectionalLearningProgress(search_space=3)

        # Set up some test data
        lp._outcomes[0].extend([0.0, 0.1, 0.2])
        lp._outcomes[1].extend([0.5, 0.6])
        lp._outcomes[2].extend([1.0])

        # Initialize arrays
        lp._update()

        # Calculate distribution
        task_dist = lp._sample_distribution()

        # Check that distribution was created and stored in shared memory
        assert lp._task_dist_shm is not None
        assert task_dist is not None
        assert len(task_dist) == 3
        assert np.sum(task_dist) == pytest.approx(1.0)
        assert not lp._stale_dist.value

    def test_add_stats(self):
        """Test that stats collection works with shared memory."""
        lp = BidirectionalLearningProgress(search_space=3)

        # Set up some test data
        lp._outcomes[0].extend([0.0, 0.1, 0.2])
        lp._outcomes[1].extend([0.5, 0.6])
        lp._outcomes[2].extend([1.0])

        # Run update to initialize arrays
        lp._update()
        lp._sample_distribution()

        # Get stats
        stats = lp.add_stats()

        # Check that stats are properly extracted from shared memory
        assert "lp/num_active_tasks" in stats
        assert "lp/mean_sample_prob" in stats
        assert "lp/num_zeros_lp_dist" in stats
        assert "lp/task_1_success_rate" in stats
        assert "lp/task_success_rate" in stats
        assert "lp/mean_evals_per_task" in stats
        assert "lp/num_nan_tasks" in stats

        # Verify some values
        assert stats["lp/num_active_tasks"] == 3
        assert stats["lp/mean_sample_prob"] > 0
        assert stats["lp/task_success_rate"] == pytest.approx(0.55, rel=1e-2)  # mean of [0.1, 0.55, 1.0]


def shared_memory_worker(task_queue, result_queue, num_steps):
    """Worker process that interacts with shared memory curriculum."""
    # Create curriculum (shares memory with parent)
    tasks = {"task_0": 1.0, "task_1": 1.0, "task_2": 1.0}
    curriculum = LearningProgressCurriculum(tasks=tasks, ema_timescale=0.01, num_active_tasks=3, sample_threshold=5)

    # Run some steps
    for i in range(num_steps):
        # Get task
        task = curriculum.get_task()
        # Extract the base task name (remove curriculum prefix)
        task_name = task.id().split(":")[0] if ":" in task.id() else task.id()

        # Generate score (simple pattern for testing)
        score = min(1.0, i * 0.1) if task_name == "task_0" else 0.0

        # Complete task
        curriculum.complete_task(task_name, score)

        # Send result
        result_queue.put((i, task_name, score))

    # Send final stats
    stats = curriculum.stats()
    result_queue.put(("stats", stats))


class TestLearningProgressCurriculumSharedMemory:
    """Test the LearningProgressCurriculum with shared memory across processes."""

    def test_basic_functionality(self, mock_curriculum_from_config_path):
        """Test basic curriculum functionality with shared memory."""
        tasks = {"task_0": 1.0, "task_1": 1.0, "task_2": 1.0}
        curriculum = LearningProgressCurriculum(tasks=tasks, ema_timescale=0.01, num_active_tasks=3, sample_threshold=5)

        # Run some steps
        for _ in range(20):
            task = curriculum.get_task()
            # Extract the base task name (remove curriculum prefix)
            task_name = task.id().split(":")[0] if ":" in task.id() else task.id()
            score = 0.5 if "task_0" in task_name else 0.1
            curriculum.complete_task(task_name, score)

        # Get stats
        stats = curriculum.stats()
        assert "lp/num_active_tasks" in stats
        assert "lp/task_success_rate" in stats

        # Check task weights were updated
        weights = curriculum.get_task_probs()
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0)

    @pytest.mark.skipif(mp.get_start_method() == "spawn", reason="Shared memory test requires fork or forkserver")
    def test_multiprocess_shared_memory(self, mock_curriculum_from_config_path):
        """Test that shared memory works across multiple processes."""
        # Note: This test may need adjustment based on the multiprocessing start method
        # On some systems (e.g., macOS), 'spawn' is default which doesn't share memory

        # Create queues for communication
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Start worker process
        num_steps = 30
        worker = mp.Process(target=shared_memory_worker, args=(task_queue, result_queue, num_steps))
        worker.start()

        # Collect results
        results = []
        stats = None

        for _ in range(num_steps + 1):  # +1 for stats
            result = result_queue.get(timeout=5)
            if result[0] == "stats":
                stats = result[1]
            else:
                results.append(result)

        worker.join(timeout=5)

        # Verify results
        assert len(results) == num_steps
        assert stats is not None

        # Check that learning happened
        assert "lp/task_success_rate" in stats
        assert stats["lp/task_success_rate"] > 0

        # Check that task_0 (which improves) was preferred
        task_0_count = sum(1 for _, task_id, _ in results if task_id == "task_0")
        assert task_0_count > num_steps // 3  # Should be selected more than uniform

    def test_curriculum_with_varying_success_rates(self, mock_curriculum_from_config_path):
        """Test curriculum behavior with tasks of varying success rates."""
        tasks = {"easy": 1.0, "medium": 1.0, "hard": 1.0, "impossible": 1.0}

        curriculum = LearningProgressCurriculum(
            tasks=tasks,
            ema_timescale=0.05,  # Faster adaptation
            num_active_tasks=4,
            sample_threshold=3,  # Lower threshold for faster updates
            rand_task_rate=0.5,  # Higher exploration rate to ensure all tasks are sampled
            memory=25,  # Ensure we have enough memory for learning
        )

        # Define success rates for each task
        success_rates = {
            "easy": lambda step: 0.9,  # Always high
            "medium": lambda step: min(0.8, step * 0.02),  # Gradual improvement
            "hard": lambda step: min(0.5, step * 0.01),  # Slow improvement
            "impossible": lambda step: 0.0,  # Never succeeds
        }

        task_counts = {task: 0 for task in tasks}

        # First, ensure each task is sampled at least once to initialize the algorithm
        for task_name in tasks:
            for _ in range(3):  # Sample each task 3 times to reach the threshold
                task = curriculum.get_task()
                curriculum.complete_task(task_name, success_rates[task_name](0))

        # Run simulation with more iterations for convergence
        for step in range(200):
            task = curriculum.get_task()
            # Extract the base task name (remove curriculum prefix)
            task_name = task.id().split(":")[0] if ":" in task.id() else task.id()

            # Get score based on task difficulty
            score = success_rates[task_name](step)

            # Complete task
            curriculum.complete_task(task_name, score)
            task_counts[task_name] += 1

        # Analyze results
        print(f"Task counts after 200 steps: {task_counts}")

        # With 50% random exploration, we should see at least some variety
        explored_tasks = sum(1 for count in task_counts.values() if count > 0)
        assert explored_tasks >= 2, (
            f"With 50% random rate, at least 2 tasks should be explored, but only {explored_tasks} were"
        )

        # After convergence, tasks with better performance should get more samples
        # Calculate ratios for clearer comparison
        total_count = sum(task_counts.values())
        task_ratios = {task: count / total_count for task, count in task_counts.items() if count > 0}
        print(f"Task sampling ratios: {task_ratios}")

        # The algorithm should show some preference
        if len(task_ratios) > 1:
            ratio_variance = np.var(list(task_ratios.values()))
            max_min_diff = max(task_ratios.values()) - min(task_ratios.values())
            assert ratio_variance > 0.0001 or max_min_diff > 0.05, (
                f"Algorithm should show preference (variance: {ratio_variance:.6f}, diff: {max_min_diff:.3f})"
            )

        # At least one of the learning tasks (medium/hard) should get reasonable sampling
        assert task_ratios["medium"] > 0.15 or task_ratios["hard"] > 0.15, (
            "At least one learning task should get reasonable sampling"
        )

        # Get final stats
        stats = curriculum.stats()
        assert stats["lp/task_success_rate"] > 0
        assert stats["lp/num_active_tasks"] == 4

    def test_memory_limit(self, mock_curriculum_from_config_path):
        """Test that memory limit for outcomes works correctly."""
        tasks = {"task_0": 1.0}
        memory_limit = 10

        curriculum = LearningProgressCurriculum(
            tasks=tasks,
            memory=memory_limit,
            sample_threshold=5,
            num_active_tasks=1,  # Only 1 active task since we only have 1 task total
        )

        # Add more outcomes than memory limit
        for _ in range(memory_limit * 3):  # Add 30 items
            curriculum.get_task()
            curriculum.complete_task("task_0", 0.5)

        # Force the counter to exceed threshold to trigger an update
        lp_tracker = curriculum._lp_tracker
        lp_tracker._counter[0] = lp_tracker._sample_threshold.value

        # Call calculate_dist to trigger memory trimming
        lp_tracker.calculate_dist()

        # Check that outcomes are limited to memory size
        assert len(lp_tracker._outcomes[0]) <= memory_limit

    def test_edge_cases(self, mock_curriculum_from_config_path):
        """Test edge cases like empty outcomes, NaN values, etc."""
        tasks = {"task_0": 1.0, "task_1": 1.0}
        curriculum = LearningProgressCurriculum(tasks=tasks)

        # Test with no data collected
        stats = curriculum.stats()
        assert stats["lp/mean_sample_prob"] == 0.0  # No distribution yet
        assert stats["lp/num_zeros_lp_dist"] == 0

        # Test with NaN prevention
        task = curriculum.get_task()
        # Extract the base task name (remove curriculum prefix)
        task_name = task.id().split(":")[0] if ":" in task.id() else task.id()
        curriculum.complete_task(task_name, float("inf"))  # Should be capped

        # Test with negative scores (should be clamped to 0)
        task = curriculum.get_task()
        task_name = task.id().split(":")[0] if ":" in task.id() else task.id()
        curriculum.complete_task(task_name, -1.0)

        # Verify curriculum still works
        stats = curriculum.stats()
        assert not np.isnan(stats["lp/task_success_rate"])
        assert 0 <= stats["lp/task_success_rate"] <= 1.0

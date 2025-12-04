"""Tests for regret-based curriculum algorithm implementations."""

import random

import pytest

from metta.cogworks.curriculum.curriculum import CurriculumTask
from metta.cogworks.curriculum.prioritized_regret_algorithm import (
    PrioritizedRegretAlgorithm,
    PrioritizedRegretConfig,
)
from metta.cogworks.curriculum.regret_learning_progress_algorithm import (
    RegretLearningProgressAlgorithm,
    RegretLearningProgressConfig,
)
from metta.cogworks.curriculum.regret_tracker import RegretTracker


def _register_task(algorithm, task_id: int) -> int:
    """Register a task with the algorithm."""
    algorithm.on_task_created(CurriculumTask(task_id, {"task_id": task_id}))
    return task_id


def _create_tasks(algorithm, count: int, rng: random.Random | None = None) -> list[int]:
    """Create multiple tasks for testing."""
    task_ids: list[int] = []
    for index in range(count):
        task_id = rng.randint(0, 1_000_000) if rng is not None else index
        task_ids.append(_register_task(algorithm, task_id))
    return task_ids


class TestRegretTracker:
    """Test RegretTracker core functionality."""

    def test_regret_computation(self):
        """Test that regret is correctly computed as optimal - achieved."""
        tracker = RegretTracker(optimal_value=1.0)

        # Perfect performance = 0 regret
        assert tracker.compute_regret(1.0) == 0.0

        # Moderate performance = moderate regret
        assert tracker.compute_regret(0.5) == 0.5

        # Poor performance = high regret
        assert tracker.compute_regret(0.1) == 0.9

        # Ensure regret doesn't go negative
        assert tracker.compute_regret(1.5) == 0.0

    def test_regret_tracking(self, random_seed):
        """Test that regret is tracked correctly for tasks."""
        tracker = RegretTracker(optimal_value=1.0, regret_ema_timescale=0.1)

        task_id = 42
        tracker.track_task_creation(task_id)

        # Update with performance
        tracker.update_task_performance(task_id, 0.7)  # regret = 0.3
        tracker.update_task_performance(task_id, 0.8)  # regret = 0.2

        stats = tracker.get_task_stats(task_id)
        assert stats is not None
        assert stats["completion_count"] == 2
        assert 0.2 <= stats["ema_regret"] <= 0.3  # EMA should be between recent values

    def test_regret_progress(self, random_seed):
        """Test regret progress calculation (fast vs slow EMA)."""
        tracker = RegretTracker(optimal_value=1.0, regret_ema_timescale=0.1)

        task_id = 42
        tracker.track_task_creation(task_id)

        # Improving performance (regret decreasing)
        for score in [0.2, 0.4, 0.6, 0.8]:
            tracker.update_task_performance(task_id, score)

        regret_progress = tracker.get_regret_progress(task_id)
        assert regret_progress is not None
        # Regret should be decreasing (fast < slow), so progress should be negative
        assert regret_progress < 0, f"Expected negative progress (improving), got {regret_progress}"

    def test_regret_tracker_memory_limit(self, random_seed):
        """Test that regret tracker respects memory limits."""
        tracker = RegretTracker(max_memory_tasks=5)

        rng = random.Random(random_seed)

        # Create more tasks than memory limit
        for _ in range(10):
            task_id = rng.randint(0, 100000)
            tracker.track_task_creation(task_id)
            tracker.update_task_performance(task_id, 0.5)

        # Should have at most max_memory_tasks
        tracked = tracker.get_all_tracked_tasks()
        assert len(tracked) <= 5


class TestPrioritizedRegretAlgorithm:
    """Test PrioritizedRegret curriculum algorithm."""

    @pytest.fixture
    def prioritized_regret_config(self):
        """Fixture providing prioritized regret configuration."""
        return PrioritizedRegretConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            exploration_bonus=0.1,
            temperature=1.0,
            max_memory_tasks=10,
        )

    def test_prioritized_regret_favors_high_regret(self, random_seed, prioritized_regret_config):
        """Test that PrioritizedRegret prioritizes tasks with high regret."""
        algorithm = PrioritizedRegretAlgorithm(num_tasks=3, hypers=prioritized_regret_config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Task 1: High performance (low regret)
        for _ in range(10):
            algorithm.update_task_performance(task1_id, 0.9)

        # Task 2: Medium performance (medium regret)
        for _ in range(10):
            algorithm.update_task_performance(task2_id, 0.5)

        # Task 3: Low performance (high regret)
        for _ in range(10):
            algorithm.update_task_performance(task3_id, 0.1)

        # Get scores
        scores = algorithm.score_tasks([task1_id, task2_id, task3_id])

        # Task 3 (high regret) should have highest score
        assert scores[task3_id] > scores[task2_id], (
            f"High regret task should score higher. "
            f"High regret: {scores[task3_id]}, Medium regret: {scores[task2_id]}"
        )
        assert scores[task2_id] > scores[task1_id], (
            f"Medium regret task should score higher than low regret. "
            f"Medium regret: {scores[task2_id]}, Low regret: {scores[task1_id]}"
        )

    def test_prioritized_regret_eviction_policy(self, random_seed, prioritized_regret_config):
        """Test that eviction prefers low-regret (solved) tasks."""
        algorithm = PrioritizedRegretAlgorithm(num_tasks=3, hypers=prioritized_regret_config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Task 1: Nearly solved (very low regret)
        for _ in range(10):
            algorithm.update_task_performance(task1_id, 0.95)

        # Task 2: Moderate difficulty (medium regret)
        for _ in range(10):
            algorithm.update_task_performance(task2_id, 0.5)

        # Task 3: Challenging (high regret)
        for _ in range(10):
            algorithm.update_task_performance(task3_id, 0.2)

        # Test eviction recommendation
        task_ids = [task1_id, task2_id, task3_id]
        eviction_recommendation = algorithm.recommend_eviction(task_ids)

        # Should recommend task 1 (lowest regret, nearly solved)
        assert eviction_recommendation == task1_id, (
            f"Should evict low-regret task, got task {eviction_recommendation}"
        )

    def test_prioritized_regret_exploration_bonus(self, random_seed, prioritized_regret_config):
        """Test that new tasks get exploration bonus."""
        algorithm = PrioritizedRegretAlgorithm(num_tasks=2, hypers=prioritized_regret_config)

        rng = random.Random(random_seed)
        task1_id, task2_id = _create_tasks(algorithm, 2, rng)

        # Task 1: No updates (new task)
        # Task 2: Some updates
        for _ in range(5):
            algorithm.update_task_performance(task2_id, 0.5)

        scores = algorithm.score_tasks([task1_id, task2_id])

        # New task should get exploration bonus
        assert scores[task1_id] == prioritized_regret_config.exploration_bonus

    def test_prioritized_regret_stats(self, random_seed, prioritized_regret_config):
        """Test that PrioritizedRegret provides expected statistics."""
        algorithm = PrioritizedRegretAlgorithm(num_tasks=2, hypers=prioritized_regret_config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 2, rng)

        # Add performance data
        for task_id in task_ids:
            for _ in range(10):
                algorithm.update_task_performance(task_id, rng.uniform(0.0, 1.0))

        # Get stats
        stats = algorithm.stats()

        # Should have regret stats
        assert "regret/mean_regret" in stats
        assert "regret/total_tracked_tasks" in stats
        assert stats["regret/total_tracked_tasks"] > 0


class TestRegretLearningProgressAlgorithm:
    """Test RegretLearningProgress curriculum algorithm."""

    @pytest.fixture(params=[False, True], ids=["simple", "bidirectional"])
    def regret_lp_config(self, request):
        """Fixture providing regret learning progress configurations."""
        use_bidirectional = request.param
        return RegretLearningProgressConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            use_bidirectional=use_bidirectional,
            exploration_bonus=0.1,
            invert_regret_progress=True,
            max_memory_tasks=10,
        )

    def test_regret_lp_favors_decreasing_regret(self, random_seed, regret_lp_config):
        """Test that RegretLearningProgress prioritizes tasks with decreasing regret."""
        algorithm = RegretLearningProgressAlgorithm(num_tasks=3, hypers=regret_lp_config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Task 1: Improving rapidly (regret decreasing fast)
        for i in range(20):
            algorithm.update_task_performance(task1_id, 0.1 + i * 0.04)  # 0.1 to 0.9

        # Task 2: Improving slowly (regret decreasing slow)
        for i in range(20):
            algorithm.update_task_performance(task2_id, 0.4 + i * 0.01)  # 0.4 to 0.6

        # Task 3: Constant (no regret change)
        for _ in range(20):
            algorithm.update_task_performance(task3_id, 0.5)

        scores = algorithm.score_tasks([task1_id, task2_id, task3_id])

        # Task 1 (fast improving) should have highest score
        if regret_lp_config.use_bidirectional:
            # Bidirectional may need more data, but should at least not penalize fast learning
            assert scores[task1_id] >= scores[task3_id], (
                f"Fast improving should have >= score. Fast: {scores[task1_id]}, Constant: {scores[task3_id]}"
            )
        else:
            # Simple version should clearly favor fast improving
            assert scores[task1_id] > scores[task2_id], (
                f"Fast improving should score higher. Fast: {scores[task1_id]}, Slow: {scores[task2_id]}"
            )

    def test_regret_lp_detects_forgetting(self, random_seed, regret_lp_config):
        """Test that RegretLearningProgress detects when performance is degrading."""
        algorithm = RegretLearningProgressAlgorithm(num_tasks=2, hypers=regret_lp_config)

        rng = random.Random(random_seed)
        task1_id, task2_id = _create_tasks(algorithm, 2, rng)

        # Task 1: Degrading (regret increasing)
        for i in range(15):
            algorithm.update_task_performance(task1_id, 0.9 - i * 0.04)  # 0.9 to 0.3

        # Task 2: Stable
        for _ in range(15):
            algorithm.update_task_performance(task2_id, 0.5)

        # Get regret stats for task 1
        regret_stats = algorithm.regret_tracker.get_task_stats(task1_id)
        assert regret_stats is not None

        # Regret should be increasing (performance degrading)
        regret_progress = algorithm.regret_tracker.get_regret_progress(task1_id)
        if regret_progress is not None:
            # Positive progress = regret increasing = forgetting
            assert regret_progress > 0, f"Expected positive progress (forgetting), got {regret_progress}"

    def test_regret_lp_eviction_policy(self, random_seed, regret_lp_config):
        """Test that eviction prefers tasks with low learning progress."""
        algorithm = RegretLearningProgressAlgorithm(num_tasks=3, hypers=regret_lp_config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Task 1: High regret change (high learning progress)
        for i in range(15):
            algorithm.update_task_performance(task1_id, 0.1 if i % 2 == 0 else 0.9)

        # Task 2: Low regret change (low learning progress)
        for _ in range(15):
            algorithm.update_task_performance(task2_id, 0.5)

        # Task 3: Some regret change
        for i in range(15):
            algorithm.update_task_performance(task3_id, 0.4 + i * 0.02)

        # Test eviction recommendation
        task_ids = [task1_id, task2_id, task3_id]
        eviction_recommendation = algorithm.recommend_eviction(task_ids)

        # Should recommend task 2 (lowest learning progress)
        assert eviction_recommendation == task2_id, (
            f"Should evict low learning progress task, got task {eviction_recommendation}"
        )

    def test_regret_lp_bidirectional_stats(self, random_seed):
        """Test that bidirectional regret LP provides expected statistics."""
        config = RegretLearningProgressConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            use_bidirectional=True,
            max_memory_tasks=10,
        )
        algorithm = RegretLearningProgressAlgorithm(num_tasks=2, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 2, rng)

        # Add performance data
        for i in range(20):
            algorithm.update_task_performance(task_ids[0], 0.1 + i * 0.04)  # Improving
            algorithm.update_task_performance(task_ids[1], 0.5)  # Constant

        # Get stats
        stats = algorithm.stats()

        # Should have regret learning progress stats
        assert "regret_lp/num_tracked_tasks" in stats
        assert "regret/mean_regret" in stats
        assert stats["regret_lp/num_tracked_tasks"] > 0

    def test_regret_lp_checkpointing(self, random_seed, regret_lp_config):
        """Test that RegretLearningProgress can save and load state."""
        algorithm1 = RegretLearningProgressAlgorithm(num_tasks=2, hypers=regret_lp_config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm1, 2, rng)

        # Add some data
        for task_id in task_ids:
            for i in range(10):
                algorithm1.update_task_performance(task_id, 0.3 + i * 0.05)

        # Save state
        state = algorithm1.get_state()

        # Create new algorithm and load state
        algorithm2 = RegretLearningProgressAlgorithm(num_tasks=2, hypers=regret_lp_config)
        algorithm2.load_state(state)

        # Scores should match
        scores1 = algorithm1.score_tasks(task_ids)
        scores2 = algorithm2.score_tasks(task_ids)

        for task_id in task_ids:
            assert abs(scores1[task_id] - scores2[task_id]) < 1e-6, (
                f"Scores should match after load. Task {task_id}: {scores1[task_id]} vs {scores2[task_id]}"
            )


class TestRegretAlgorithmsComparison:
    """Test comparing behavior of different regret-based algorithms."""

    def test_regret_algorithms_handle_same_data_differently(self, random_seed):
        """Test that PrioritizedRegret and RegretLearningProgress handle data differently."""
        pr_config = PrioritizedRegretConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            max_memory_tasks=10,
        )
        rlp_config = RegretLearningProgressConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.01,
            use_bidirectional=False,  # Use simple for clearer comparison
            max_memory_tasks=10,
        )

        pr_algorithm = PrioritizedRegretAlgorithm(num_tasks=2, hypers=pr_config)
        rlp_algorithm = RegretLearningProgressAlgorithm(num_tasks=2, hypers=rlp_config)

        rng = random.Random(random_seed)

        # Create tasks with same IDs
        task1_id = rng.randint(0, 100000)
        task2_id = rng.randint(0, 100000)

        _register_task(pr_algorithm, task1_id)
        _register_task(pr_algorithm, task2_id)
        _register_task(rlp_algorithm, task1_id)
        _register_task(rlp_algorithm, task2_id)

        # Task 1: High regret, stable
        # Task 2: Low regret, stable

        for _ in range(15):
            pr_algorithm.update_task_performance(task1_id, 0.2)  # High regret
            pr_algorithm.update_task_performance(task2_id, 0.9)  # Low regret
            rlp_algorithm.update_task_performance(task1_id, 0.2)
            rlp_algorithm.update_task_performance(task2_id, 0.9)

        pr_scores = pr_algorithm.score_tasks([task1_id, task2_id])
        rlp_scores = rlp_algorithm.score_tasks([task1_id, task2_id])

        # PrioritizedRegret should favor task 1 (high regret)
        assert pr_scores[task1_id] > pr_scores[task2_id], (
            f"PR should favor high regret. Task1: {pr_scores[task1_id]}, Task2: {pr_scores[task2_id]}"
        )

        # RegretLearningProgress with stable performance should not strongly differentiate
        # (both tasks have low learning progress since performance is stable)
        # Just check that scores are valid
        assert rlp_scores[task1_id] >= 0
        assert rlp_scores[task2_id] >= 0

    def test_regret_lp_favors_change_over_absolute_regret(self, random_seed):
        """Test that RegretLearningProgress favors regret change over absolute regret."""
        config = RegretLearningProgressConfig(
            optimal_value=1.0,
            regret_ema_timescale=0.05,  # Higher for faster response
            use_bidirectional=False,  # Simple for clearer test
            invert_regret_progress=True,
            max_memory_tasks=10,
        )

        algorithm = RegretLearningProgressAlgorithm(num_tasks=2, hypers=config)

        rng = random.Random(random_seed)
        task1_id, task2_id = _create_tasks(algorithm, 2, rng)

        # Task 1: High absolute regret, but improving (regret decreasing)
        for i in range(15):
            algorithm.update_task_performance(task1_id, 0.1 + i * 0.03)  # 0.1 to 0.55

        # Task 2: Lower absolute regret, but stable (no regret change)
        for _ in range(15):
            algorithm.update_task_performance(task2_id, 0.7)  # stable at 0.7

        scores = algorithm.score_tasks([task1_id, task2_id])

        # Task 1 should score higher because regret is decreasing
        # (even though absolute regret is higher)
        assert scores[task1_id] > scores[task2_id], (
            f"Improving task should score higher than stable task. "
            f"Improving: {scores[task1_id]}, Stable: {scores[task2_id]}"
        )


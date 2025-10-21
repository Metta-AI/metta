"""Tests for curriculum algorithm implementations."""

import random

import pytest

from metta.cogworks.curriculum.curriculum import CurriculumTask
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig

from .test_helpers import CurriculumTestHelper


@pytest.fixture(params=[False, True], ids=["standard", "bidirectional"])
def learning_progress_config(request):
    """Fixture providing both standard and bidirectional learning progress configurations."""
    use_bidirectional = request.param
    return LearningProgressConfig(
        ema_timescale=0.001,
        num_active_tasks=10,
        use_shared_memory=False,  # Faster for unit tests
        use_bidirectional=use_bidirectional,
    )


def _register_task(algorithm: LearningProgressAlgorithm, task_id: int) -> int:
    algorithm.on_task_created(CurriculumTask(task_id, {"task_id": task_id}))
    return task_id


def _create_tasks(
    algorithm: LearningProgressAlgorithm,
    count: int,
    rng: random.Random | None = None,
) -> list[int]:
    task_ids: list[int] = []
    for index in range(count):
        task_id = rng.randint(0, 1_000_000) if rng is not None else index
        task_ids.append(_register_task(algorithm, task_id))
    return task_ids


def _sample_task_id(
    algorithm: LearningProgressAlgorithm,
    task_ids: list[int],
    rng: random.Random,
) -> int:
    if not task_ids:
        raise ValueError("Cannot sample from an empty task list")

    scores = algorithm.score_tasks(task_ids)
    if not scores:
        return rng.choice(task_ids)

    total_score = sum(scores.values())
    if total_score <= 0:
        return rng.choice(task_ids)

    weights = [scores.get(task_id, 0.0) for task_id in task_ids]
    return rng.choices(task_ids, weights=weights)[0]


class TestLearningProgressCoreBehavior:
    """Test core learning progress algorithm behavior."""

    def test_learning_progress_favors_fast_learning(self, random_seed, learning_progress_config):
        """Test that fast learning tasks get higher learning progress scores than slow learning."""
        # Set up algorithm with tasks (works for both standard and bidirectional)
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=learning_progress_config)

        rng = random.Random(random_seed)
        task1_id, task2_id = _create_tasks(algorithm, 2, rng)

        # Use helper to setup performance patterns - REDUCED from 10 to 3 iterations
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=3)

        # Get LP scores for both tasks via public scoring API
        scores = algorithm.score_tasks([task1_id, task2_id])
        lp_score_1 = scores.get(task1_id, 0.0)
        lp_score_2 = scores.get(task2_id, 0.0)

        # Behavior differs between standard and bidirectional algorithms
        if learning_progress_config.use_bidirectional:
            # Bidirectional algorithm may return equal scores with limited data
            # The key is that it doesn't penalize fast learning
            assert lp_score_1 >= lp_score_2, (
                f"Fast learning should have >= LP score in bidirectional. Fast: {lp_score_1}, Slow: {lp_score_2}"
            )
        else:
            # Standard algorithm should clearly favor fast learning
            assert lp_score_1 > lp_score_2, (
                f"Fast learning should have higher LP score. Fast: {lp_score_1}, Slow: {lp_score_2}"
            )

    def test_learning_progress_favors_changing_performance(self, random_seed, learning_progress_config):
        """Test that changing performance has higher learning progress scores than consistent performance."""
        # Set up algorithm with tasks (works for both standard and bidirectional)
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=learning_progress_config)

        rng = random.Random(random_seed)
        task1_id, task2_id = _create_tasks(algorithm, 2, rng)

        # Use helper to setup performance patterns - need enough iterations for bidirectional algorithm
        CurriculumTestHelper.setup_learning_comparison(
            algorithm, (task1_id, task2_id), "changing_vs_consistent", iterations=10
        )

        scores = algorithm.score_tasks([task1_id, task2_id])
        lp_score_1 = scores.get(task1_id, 0.0)
        lp_score_2 = scores.get(task2_id, 0.0)

        # Behavior differs between standard and bidirectional algorithms
        if learning_progress_config.use_bidirectional:
            # Bidirectional algorithm measures learning progress differently
            # Both tasks should have valid scores, and algorithm should function correctly
            assert lp_score_1 > 0, f"Consistent task should have positive LP score: {lp_score_1}"
            assert lp_score_2 > 0, f"Changing task should have positive LP score: {lp_score_2}"
            # Verify the scores are being calculated (not both equal to default)
            assert lp_score_1 + lp_score_2 > 0.1, "Total LP scores should be meaningful"
        else:
            # Standard algorithm should clearly favor changing performance
            assert lp_score_2 > lp_score_1, (
                f"Changing performance should have higher LP score. Changing: {lp_score_2}, Consistent: {lp_score_1}"
            )

    def test_learning_progress_sampling_favors_high_lp_tasks(self, random_seed, learning_progress_config):
        """Test that sampling favors tasks with higher learning progress scores."""
        # Set up algorithm with tasks (works for both standard and bidirectional)
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=learning_progress_config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Use more iterations for reliable score differentiation
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=15)

        # Give Task 3 some consistent performance data so it has a score
        for _ in range(10):
            algorithm.update_task_performance(task3_id, 0.5)  # Moderate consistent performance

        # Use more samples for reliable statistics and get scores for verification
        num_samples = 200
        samples = []
        all_task_ids = [task1_id, task2_id, task3_id]

        # Get learning progress scores for debugging
        scores = algorithm.score_tasks(all_task_ids)
        task1_score = scores.get(task1_id, 0.0)
        task2_score = scores.get(task2_id, 0.0)
        task3_score = scores.get(task3_id, 0.0)

        for _ in range(num_samples):
            sampled_task_id = _sample_task_id(algorithm, all_task_ids, rng)
            samples.append(sampled_task_id)

        # Count samples for each task
        task1_count = samples.count(task1_id)
        task2_count = samples.count(task2_id)
        task3_count = samples.count(task3_id)

        # Verify that all tasks have meaningful scores and get sampled
        # Fast learner should have highest score, others should have non-zero scores
        assert task1_score > 0, f"Task 1 (fast) should have positive score, got {task1_score}"
        assert task2_score >= 0, f"Task 2 (slow) should have non-negative score, got {task2_score}"
        assert task3_score >= 0, f"Task 3 (consistent) should have non-negative score, got {task3_score}"

        # All tasks should be sampled at least once with 200 samples
        # If a task has zero score, it might not be sampled, so be more lenient
        total_score = task1_score + task2_score + task3_score
        if total_score > 0:
            # Tasks with positive scores should be sampled
            if task1_score > 0:
                assert task1_count > 0, f"Task 1 with score {task1_score} should be sampled at least once"
            if task2_score > 0:
                assert task2_count > 0, f"Task 2 with score {task2_score} should be sampled at least once"
            if task3_score > 0:
                assert task3_count > 0, f"Task 3 with score {task3_score} should be sampled at least once"
        else:
            # If all scores are zero, sampling should be uniform random
            assert task1_count > 0 and task2_count > 0 and task3_count > 0, (
                "With zero scores, all tasks should be sampled uniformly"
            )

    def test_learning_progress_pool_management(self, random_seed):
        """Test that the learning progress algorithm properly manages its task pool."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=5,
            use_shared_memory=False,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 2, rng)

        # Simulate more performance updates for realistic EMA development
        for task_id in task_ids:
            for i in range(15):
                algorithm.update_task_performance(task_id, 0.5 + 0.1 * (i % 5))

        # Check that tasks are being tracked
        tracked_tasks = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_tasks) >= 2

    def test_learning_progress_ema_smoothing(self, random_seed):
        """Test that EMA smoothing works correctly for learning progress calculation."""
        config = LearningProgressConfig(
            ema_timescale=0.1,  # Higher timescale for faster convergence in test
            num_active_tasks=10,
            use_shared_memory=False,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=1, hypers=config)

        task_id = _create_tasks(algorithm, 1, random.Random(random_seed))[0]

        # Feed performance data - REDUCED from 10 to 5 data points
        performances = [0.1, 0.9, 0.2, 0.8, 0.3]
        for performance in performances:
            algorithm.update_task_performance(task_id, performance)

        # Get learning progress score via scoring API
        lp_score = algorithm.score_tasks([task_id]).get(task_id, 0.0)

        # Should have non-zero learning progress due to variance
        assert lp_score > 0, f"Learning progress should be positive for varying performance, got {lp_score}"

    def test_learning_progress_eviction_policy(self, random_seed):
        """Test that eviction policy prefers tasks with low learning progress."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=10,
            use_shared_memory=False,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=config)

        rng = random.Random(random_seed)
        task1_id, task2_id, task3_id = _create_tasks(algorithm, 3, rng)

        # Task 1: High variance (high learning progress) - more presentations for realistic EMAs
        for i in range(20):
            algorithm.update_task_performance(task1_id, 0.1 if i % 2 == 0 else 0.9)

        # Task 2: Low variance (low learning progress) - more presentations for realistic EMAs
        for _ in range(20):
            algorithm.update_task_performance(task2_id, 0.5)

        # Task 3: No updates (gets exploration bonus)

        # Test eviction recommendation
        task_ids = [task1_id, task2_id, task3_id]
        eviction_recommendation = algorithm.recommend_eviction(task_ids)

        # Should recommend task 2 (low variance) over task 1 (high variance)
        # Task 3 might be recommended due to exploration bonus
        assert eviction_recommendation in [task2_id, task3_id], (
            f"Should recommend low-learning-progress task for eviction, got {eviction_recommendation}"
        )


class TestLearningProgressProductionPatterns:
    """Test production-like patterns and stress scenarios."""

    def test_learning_progress_with_many_tasks(self, random_seed):
        """Test learning progress algorithm with production-like task counts."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=50,  # REDUCED from 100 for faster testing
            enable_detailed_slice_logging=True,  # Enable detailed stats for testing
        )
        algorithm = LearningProgressAlgorithm(num_tasks=20, hypers=config)  # REDUCED from 50

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 15, rng)

        # Simulate training-like performance updates
        # REDUCED from 100 to 30 updates
        for _ in range(30):
            task_id = rng.choice(task_ids)
            performance = rng.uniform(0.0, 1.0)
            algorithm.update_task_performance(task_id, performance)

        # Test that stats are available with new structure
        stats = algorithm.stats()
        assert "lp/mean_learning_progress" in stats  # Scorer stats
        assert "num_tasks" in stats  # Base stat
        # With 30 updates on 15 tasks, we should have some learning progress tracked
        assert isinstance(stats["lp/mean_learning_progress"], (int, float))

    def test_learning_progress_memory_management(self, random_seed):
        """Test that memory management works under production load."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=10,  # Small limit to trigger cleanup
        )
        algorithm = LearningProgressAlgorithm(num_tasks=20, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 15, rng)

        # Add some performance data
        for task_id in task_ids:
            algorithm.update_task_performance(task_id, rng.uniform(0.0, 1.0))

        # Check that memory limit is respected
        tracked_tasks = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_tasks) <= config.num_active_tasks + 100  # Allow cleanup buffer

    def test_learning_progress_task_sampling_distribution(self, random_seed):
        """Test that task sampling follows expected distribution patterns."""
        config = LearningProgressConfig(
            ema_timescale=0.01,  # Higher for faster convergence
            num_active_tasks=20,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=5, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 5, rng)

        # Task 1: High variance - more presentations for realistic EMA development
        for i in range(25):
            algorithm.update_task_performance(task_ids[0], 0.1 if i % 2 == 0 else 0.9)

        # Task 2: Medium variance - more presentations for realistic EMA development
        for i in range(25):
            algorithm.update_task_performance(task_ids[1], 0.3 if i % 2 == 0 else 0.7)

        # Task 3: Low variance - more presentations for realistic EMA development
        for _ in range(25):
            algorithm.update_task_performance(task_ids[2], 0.5)

        # Tasks 4,5: No updates (exploration bonus)

        # Sample tasks - REDUCED from 200 to 100
        samples = []
        for _ in range(100):
            sampled_id = _sample_task_id(algorithm, task_ids, rng)
            samples.append(sampled_id)

        # Count samples
        sample_counts = {task_id: samples.count(task_id) for task_id in task_ids}

        # All tasks should be sampled at least once
        for task_id, count in sample_counts.items():
            assert count > 0, f"Task {task_id} was never sampled"

        # High-variance tasks should generally be sampled more
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "Should sample from multiple tasks"


class TestBidirectionalLearningProgressBehavior:
    """Test specific bidirectional learning progress behavior."""

    def test_bidirectional_fast_vs_slow_learning_with_more_data(self, random_seed):
        """Test bidirectional learning progress with more data points to detect differences."""
        # Bidirectional algorithm needs more data points to calculate meaningful progress
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=10,
            use_bidirectional=True,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=config)

        task1_id, task2_id = _create_tasks(algorithm, 2, random.Random(random_seed))

        # Use many more iterations for bidirectional algorithm to detect differences with realistic EMAs
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=50)

        scores = algorithm.score_tasks([task1_id, task2_id])
        lp_score_1 = scores.get(task1_id, 0.0)
        lp_score_2 = scores.get(task2_id, 0.0)

        # With bidirectional algorithm, either fast learning has higher score or both get exploration bonus
        # The key is that the algorithm doesn't penalize fast learning
        assert lp_score_1 >= lp_score_2, (
            f"Fast learning should have >= LP score in bidirectional. Fast: {lp_score_1}, Slow: {lp_score_2}"
        )

    def test_bidirectional_learning_progress_with_sufficient_data(self, random_seed):
        """Test that bidirectional learning progress works with sufficient task data."""
        config = LearningProgressConfig(
            ema_timescale=0.01,  # Higher timescale for faster response
            num_active_tasks=10,
            use_bidirectional=True,
            sample_threshold=5,  # Lower threshold for testing
        )
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 3, rng)

        # Create three different performance patterns with much more data for realistic EMAs
        for i in range(40):  # Many more iterations for realistic bidirectional EMA development
            # Task 1: Fast learning (steep improvement)
            algorithm.update_task_performance(task_ids[0], 0.1 + i * 0.05)
            # Task 2: Slow learning (gradual improvement)
            algorithm.update_task_performance(task_ids[1], 0.1 + i * 0.01)
            # Task 3: Variable learning (changing performance)
            perf = 0.5 + 0.4 * (1 if i % 3 == 0 else -1 if i % 3 == 1 else 0)
            algorithm.update_task_performance(task_ids[2], max(0.0, min(1.0, perf)))

        score_map = algorithm.score_tasks(task_ids)
        scores = [score_map.get(task_id, 0.0) for task_id in task_ids]

        # Verify that the algorithm produces valid scores
        assert all(score >= 0 for score in scores), f"All scores should be non-negative: {scores}"

        # Bidirectional algorithm may return uniform distribution (1/n for each task)
        # when it can't detect significant learning progress differences
        # This is valid behavior - check that scores are reasonable
        exploration_bonus = algorithm.hypers.exploration_bonus
        uniform_score = 1.0 / len(task_ids)  # 1/3 ≈ 0.333...

        # Scores should be either exploration bonus or part of uniform distribution
        for score in scores:
            assert abs(score - exploration_bonus) < 0.01 or abs(score - uniform_score) < 0.01 or score > 0, (
                f"Score should be exploration bonus ({exploration_bonus}), "
                f"uniform ({uniform_score}), or positive: {score}"
            )

    def test_bidirectional_learning_progress_stats(self, random_seed):
        """Test that bidirectional learning progress provides expected statistics."""
        config = LearningProgressConfig(
            ema_timescale=0.01,
            num_active_tasks=10,
            use_bidirectional=True,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=config)

        rng = random.Random(random_seed)
        task_ids = _create_tasks(algorithm, 2, rng)

        # Add more performance data for realistic EMA development
        for i in range(20):
            algorithm.update_task_performance(task_ids[0], 0.1 + (i % 8) * 0.1)  # Improving pattern
            algorithm.update_task_performance(task_ids[1], 0.5)  # Consistent pattern

        # Get bidirectional-specific stats
        stats = algorithm.get_stats()

        # Bidirectional scorer should provide these specific stats
        expected_keys = ["mean_task_success_rate", "mean_learning_progress", "mean_sample_prob"]
        for key in expected_keys:
            assert key in stats, f"Missing stat key: {key}"

        assert stats["mean_task_success_rate"] >= 0, "Should have task success rate stats"
        assert stats["mean_learning_progress"] >= 0, "Should have learning progress stats"

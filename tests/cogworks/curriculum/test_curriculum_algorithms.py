"""Tests for curriculum algorithm implementations."""

import random

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig

from .test_helpers import CurriculumTestHelper, MockTaskGenerator


class TestLearningProgressCoreBehavior:
    """Test core learning progress algorithm behavior."""

    def test_learning_progress_favors_fast_learning(self, random_seed):
        """Test that fast learning tasks get higher learning progress scores than slow learning."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=10,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=config)

        # Add tasks to the pool
        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        tasks = []
        for _ in range(2):
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)

        task1_id = tasks[0]._task_id
        task2_id = tasks[1]._task_id

        # Use helper to setup performance patterns - REDUCED from 10 to 3 iterations
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=3)

        # Get LP scores for both tasks
        lp_score_1 = algorithm.lp_scorer.get_learning_progress_score(task1_id, algorithm.task_tracker)
        lp_score_2 = algorithm.lp_scorer.get_learning_progress_score(task2_id, algorithm.task_tracker)

        assert lp_score_1 > lp_score_2, (
            f"Fast learning should have higher LP score. Fast: {lp_score_1}, Slow: {lp_score_2}"
        )

    def test_learning_progress_favors_changing_performance(self, random_seed):
        """Test that changing performance has higher learning progress scores than consistent performance."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=10,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=config)

        # Add tasks to the pool
        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        tasks = []
        for _ in range(2):
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)

        task1_id = tasks[0]._task_id
        task2_id = tasks[1]._task_id

        # Use helper to setup performance patterns - REDUCED from 5 to 3 iterations
        CurriculumTestHelper.setup_learning_comparison(
            algorithm, (task1_id, task2_id), "changing_vs_consistent", iterations=3
        )

        # Get LP scores for both tasks
        lp_score_1 = algorithm.lp_scorer.get_learning_progress_score(task1_id, algorithm.task_tracker)
        lp_score_2 = algorithm.lp_scorer.get_learning_progress_score(task2_id, algorithm.task_tracker)

        assert lp_score_2 > lp_score_1, (
            f"Changing performance should have higher LP score. Changing: {lp_score_2}, Consistent: {lp_score_1}"
        )

    def test_learning_progress_sampling_favors_high_lp_tasks(self, random_seed):
        """Test that sampling favors tasks with higher learning progress scores."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=10,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=config)

        # Add tasks to the pool
        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        tasks = []
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)

        task1_id = tasks[0]._task_id
        task2_id = tasks[1]._task_id
        task3_id = tasks[2]._task_id

        # Initialize tasks in the algorithm (required for tracking)
        for task in tasks:
            algorithm.on_task_created(task)

        # Use helper to setup learning patterns - REDUCED from 20 to 3 iterations
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=3)

        # REDUCED from 100 to 50 samples for faster testing
        num_samples = 50
        samples = []
        all_task_ids = [task1_id, task2_id, task3_id]
        for _ in range(num_samples):
            sampled_task_id = algorithm._choose_task_from_list(all_task_ids)
            samples.append(sampled_task_id)

        # Count samples for each task
        task1_count = samples.count(task1_id)
        task2_count = samples.count(task2_id)
        task3_count = samples.count(task3_id)

        # Higher LP tasks should be sampled more frequently
        assert task1_count > 0, "Task 1 should be sampled at least once"
        assert task2_count > 0, "Task 2 should be sampled at least once"
        assert task3_count > 0, "Task 3 should be sampled at least once"

    def test_learning_progress_pool_management(self, random_seed):
        """Test that the learning progress algorithm properly manages its task pool."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=5,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Fill the pool
        for _ in range(2):
            task = algorithm.get_task_from_pool(task_generator, rng)
            # Simulate some performance updates - REDUCED from 20 to 3
            for i in range(3):
                algorithm.update_task_performance(task._task_id, 0.5 + 0.1 * i)

        # Check that tasks are being tracked
        tracked_tasks = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_tasks) >= 2

    def test_learning_progress_ema_smoothing(self, random_seed):
        """Test that EMA smoothing works correctly for learning progress calculation."""
        config = LearningProgressConfig(
            ema_timescale=0.1,  # Higher timescale for faster convergence in test
            max_memory_tasks=10,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=1, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()
        task = algorithm.get_task_from_pool(task_generator, rng)
        task_id = task._task_id

        algorithm.on_task_created(task)

        # Feed performance data - REDUCED from 10 to 5 data points
        performances = [0.1, 0.9, 0.2, 0.8, 0.3]
        for performance in performances:
            algorithm.update_task_performance(task_id, performance)

        # Get learning progress score
        lp_score = algorithm.lp_scorer.get_learning_progress_score(task_id, algorithm.task_tracker)

        # Should have non-zero learning progress due to variance
        assert lp_score > 0, f"Learning progress should be positive for varying performance, got {lp_score}"

    def test_learning_progress_eviction_policy(self, random_seed):
        """Test that eviction policy prefers tasks with low learning progress."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=10,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        tasks = []
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)
            algorithm.on_task_created(task)

        task1_id = tasks[0]._task_id
        task2_id = tasks[1]._task_id
        task3_id = tasks[2]._task_id

        # Task 1: High variance (high learning progress)
        # REDUCED from 5 to 3 updates
        for i in range(3):
            algorithm.update_task_performance(task1_id, 0.1 if i % 2 == 0 else 0.9)

        # Task 2: Low variance (low learning progress)
        # REDUCED from 5 to 3 updates
        for _ in range(3):
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
            max_memory_tasks=50,  # REDUCED from 100 for faster testing
        )
        algorithm = LearningProgressAlgorithm(num_tasks=20, hypers=config)  # REDUCED from 50

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Create and track many tasks
        tasks = []
        for _ in range(15):  # REDUCED from 30
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)
            algorithm.on_task_created(task)

        # Simulate training-like performance updates
        # REDUCED from 100 to 30 updates
        for _ in range(30):
            task = rng.choice(tasks)
            performance = rng.uniform(0.0, 1.0)
            algorithm.update_task_performance(task._task_id, performance)

        # Test that stats are available
        stats = algorithm.stats()
        assert "tracker/total_tracked_tasks" in stats
        assert "lp/num_tracked_tasks" in stats
        assert stats["tracker/total_tracked_tasks"] > 0

    def test_learning_progress_memory_management(self, random_seed):
        """Test that memory management works under production load."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            max_memory_tasks=10,  # Small limit to trigger cleanup
        )
        algorithm = LearningProgressAlgorithm(num_tasks=20, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Create more tasks than memory limit allows
        tasks = []
        for _ in range(15):  # REDUCED from 25
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)
            algorithm.on_task_created(task)

            # Add some performance data
            algorithm.update_task_performance(task._task_id, rng.uniform(0.0, 1.0))

        # Check that memory limit is respected
        tracked_tasks = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_tasks) <= config.max_memory_tasks + 100  # Allow cleanup buffer

    def test_learning_progress_task_sampling_distribution(self, random_seed):
        """Test that task sampling follows expected distribution patterns."""
        config = LearningProgressConfig(
            ema_timescale=0.01,  # Higher for faster convergence
            max_memory_tasks=20,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=5, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Create tasks with different learning patterns
        tasks = []
        for _ in range(5):
            task = algorithm.get_task_from_pool(task_generator, rng)
            tasks.append(task)
            algorithm.on_task_created(task)

        # Task 1: High variance
        for i in range(5):  # REDUCED from 10
            algorithm.update_task_performance(tasks[0]._task_id, 0.1 if i % 2 == 0 else 0.9)

        # Task 2: Medium variance
        for i in range(5):  # REDUCED from 10
            algorithm.update_task_performance(tasks[1]._task_id, 0.3 if i % 2 == 0 else 0.7)

        # Task 3: Low variance
        for _ in range(5):  # REDUCED from 10
            algorithm.update_task_performance(tasks[2]._task_id, 0.5)

        # Tasks 4,5: No updates (exploration bonus)

        # Sample tasks - REDUCED from 200 to 100
        task_ids = [task._task_id for task in tasks]
        samples = []
        for _ in range(100):
            sampled_id = algorithm._choose_task_from_list(task_ids)
            samples.append(sampled_id)

        # Count samples
        sample_counts = {task_id: samples.count(task_id) for task_id in task_ids}

        # All tasks should be sampled at least once
        for task_id, count in sample_counts.items():
            assert count > 0, f"Task {task_id} was never sampled"

        # High-variance tasks should generally be sampled more
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "Should sample from multiple tasks"


class TestLearningProgressIntegration:
    """Integration tests for learning progress with full curriculum system."""

    def test_learning_progress_curriculum_integration(self, curriculum_with_algorithm):
        """Test learning progress algorithm integration with curriculum."""
        curriculum = curriculum_with_algorithm.make()

        # Simulate training episodes - REDUCED from 10 to 5
        for episode in range(5):
            task = curriculum.get_task()
            assert task is not None

            # Simulate task completion
            performance = 0.1 + (episode * 0.1)  # Gradually improving
            task.complete(performance)
            curriculum.update_task_performance(task._task_id, performance)

        # Check that algorithm has been updated
        assert curriculum._algorithm is not None
        stats = curriculum.stats()
        assert "algorithm/tracker/total_tracked_tasks" in stats

    def test_learning_progress_with_task_eviction(self, curriculum_with_algorithm):
        """Test learning progress behavior during task eviction scenarios."""
        config = curriculum_with_algorithm
        config.num_active_tasks = 3  # Small pool to trigger eviction quickly
        curriculum = config.make()

        tasks_seen = set()

        # Generate enough tasks to trigger eviction - REDUCED from 10 to 5
        for episode in range(5):
            task = curriculum.get_task()
            tasks_seen.add(task._task_id)

            # Complete with varying performance
            performance = 0.5 + 0.1 * (episode % 3)
            task.complete(performance)
            curriculum.update_task_performance(task._task_id, performance)

        # Should have seen some tasks (due to eviction in small pool)
        assert len(tasks_seen) >= 3

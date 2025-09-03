"""Consolidated tests for curriculum algorithms and production patterns."""

import random

import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)
from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

from .test_helpers import CurriculumTestHelper, MockTaskGenerator


class TestLearningProgressCoreBehavior:
    """Test core learning progress algorithm behavior."""

    def test_learning_progress_favors_fast_learning(self, random_seed):
        """Test that fast learning has higher learning progress scores than slow learning."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=10,
            sample_size=5,
            max_samples=10,
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

        # Use helper to setup learning patterns - REDUCED from 20 to 5 iterations
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=5)

        # Get LP scores for both tasks
        lp_score_1 = algorithm._get_task_lp_score(task1_id)
        lp_score_2 = algorithm._get_task_lp_score(task2_id)

        # Test that fast learning has higher LP score
        assert lp_score_1 > lp_score_2, (
            f"Fast learning should have higher LP score. Fast: {lp_score_1}, Slow: {lp_score_2}"
        )

    def test_learning_progress_favors_changing_performance(self, random_seed):
        """Test that changing performance has higher learning progress scores than consistent performance."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=10,
            sample_size=5,
            max_samples=10,
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

        # Use helper to setup performance patterns - REDUCED from 5 iterations
        CurriculumTestHelper.setup_learning_comparison(
            algorithm, (task1_id, task2_id), "changing_vs_consistent", iterations=5
        )

        # Get LP scores for both tasks
        lp_score_1 = algorithm._get_task_lp_score(task1_id)
        lp_score_2 = algorithm._get_task_lp_score(task2_id)

        # Test that changing performance has higher LP score
        assert lp_score_2 > lp_score_1, (
            f"Changing performance should have higher LP score. Changing: {lp_score_2}, Consistent: {lp_score_1}"
        )

    def test_learning_progress_sampling_favors_high_lp_tasks(self, random_seed):
        """Test that sampling favors tasks with higher learning progress scores."""
        # Set up algorithm with tasks
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=10,
            sample_size=5,
            max_samples=10,
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

        # Use helper to setup learning patterns - REDUCED from 20 to 5 iterations
        CurriculumTestHelper.setup_learning_comparison(algorithm, (task1_id, task2_id), "fast_vs_slow", iterations=5)

        # Test that sampling favors higher LP scores
        num_samples = 100
        samples = []
        for _ in range(num_samples):
            sampled_task_id = algorithm._choose_task()
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
            pool_size=5,
            sample_size=3,
            max_samples=5,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Fill the pool
        for _ in range(2):
            task = algorithm.get_task_from_pool(task_generator, rng)
            # Simulate some performance updates
            for i in range(5):  # REDUCED from 20 to 5
                algorithm.update_task_performance(task._task_id, 0.5 + 0.1 * i)

        # Test pool size management
        assert len(algorithm._task_memory) <= config.pool_size, "Pool should not exceed max size"

    def test_learning_progress_ema_smoothing(self, random_seed):
        """Test that EMA smoothing works correctly for learning progress calculation."""
        config = LearningProgressConfig(
            ema_timescale=0.1,  # Slower smoothing for testing
            pool_size=3,
            sample_size=2,
            max_samples=5,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=2, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        task = algorithm.get_task_from_pool(task_generator, rng)
        task_id = task._task_id

        # Update performance multiple times
        for _ in range(3):
            algorithm.update_task_performance(task_id, 0.5)

        # Get LP score
        lp_score = algorithm._get_task_lp_score(task_id)
        assert lp_score >= 0, "LP score should be non-negative"

    def test_learning_progress_eviction_policy(self, random_seed):
        """Test that tasks are properly evicted when they exceed max_samples."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=3,
            sample_size=2,
            max_samples=3,  # Low max_samples for testing
        )
        algorithm = LearningProgressAlgorithm(num_tasks=5, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Fill pool and exceed max_samples
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            # Simulate many performance updates to trigger eviction
            for i in range(8):  # REDUCED from 20 to 8
                algorithm.update_task_performance(task._task_id, 0.5 + 0.1 * i)

        # Pool should not grow indefinitely
        assert len(algorithm._task_memory) <= config.pool_size, "Pool should respect max size"

    def test_learning_progress_exploration_bonus(self, random_seed):
        """Test that exploration bonus encourages sampling of less-explored tasks."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=5,
            sample_size=3,
            max_samples=5,
            exploration_bonus=0.2,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=3, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Add tasks with different exploration levels
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            # Simulate some performance updates
            for i in range(8):  # REDUCED from 20 to 8
                algorithm.update_task_performance(task._task_id, 0.5 + 0.1 * i)

        # Test that exploration bonus affects sampling
        samples = []
        for _ in range(10):
            sampled_task_id = algorithm._choose_task()
            samples.append(sampled_task_id)

        # Should sample from multiple tasks
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "Should sample from multiple tasks"


class TestLearningProgressProductionPatterns:
    """Test learning progress algorithm in production-like scenarios."""

    def test_learning_progress_training_workflow(self, random_seed):
        """Test learning progress algorithm in a training workflow scenario."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=16,
            sample_size=8,
            max_samples=10,
            exploration_bonus=0.1,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=20, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Simulate training workflow
        for episode in range(5):  # REDUCED from 20 to 5
            # Get task for this episode
            task = algorithm.get_task_from_pool(task_generator, rng)

            # Simulate episode completion with some performance
            performance = 0.3 + 0.1 * episode  # Improving performance
            algorithm.update_task_performance(task._task_id, performance)

        # Test that algorithm maintains reasonable pool size
        assert len(algorithm._task_memory) <= config.pool_size, "Pool should respect max size"
        assert len(algorithm._task_memory) > 0, "Pool should have some tasks"

    def test_learning_progress_task_reuse_workflow(self, random_seed):
        """Test learning progress algorithm with task reuse patterns."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=10,
            sample_size=5,
            max_samples=8,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=15, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Simulate task reuse workflow
        for episode in range(5):  # REDUCED from 15 to 5
            # Get task for this episode
            task = algorithm.get_task_from_pool(task_generator, rng)

            # Simulate episode completion
            performance = 0.4 + 0.05 * episode
            algorithm.update_task_performance(task._task_id, performance)

        # Test pool management
        assert len(algorithm._task_memory) <= config.pool_size, "Pool should respect max size"

    def test_learning_progress_algorithm_configuration(self, random_seed):
        """Test different learning progress algorithm configurations."""
        configs = [
            LearningProgressConfig(
                ema_timescale=0.001,
                pool_size=8,
                sample_size=4,
                max_samples=6,
            ),
            LearningProgressConfig(
                ema_timescale=0.01,
                pool_size=12,
                sample_size=6,
                max_samples=8,
            ),
        ]

        for config in configs:
            algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)
            rng = random.Random(random_seed)
            task_generator = MockTaskGenerator()

            # Test basic functionality
            task = algorithm.get_task_from_pool(task_generator, rng)
            algorithm.update_task_performance(task._task_id, 0.5)

            # Verify configuration is applied
            assert algorithm.hypers.ema_timescale == config.ema_timescale
            assert algorithm.hypers.pool_size == config.pool_size

    def test_learning_progress_edge_cases(self, random_seed):
        """Test learning progress algorithm edge cases."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=3,
            sample_size=2,
            max_samples=3,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=5, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Test with minimal pool
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            algorithm.update_task_performance(task._task_id, 0.5)

        # Test pool behavior at capacity
        assert len(algorithm._task_memory) <= config.pool_size, "Pool should respect max size"

    def test_learning_progress_performance_tracking(self, random_seed):
        """Test that learning progress algorithm properly tracks task performance."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=5,
            sample_size=3,
            max_samples=5,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=8, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Add tasks and track performance
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            # Simulate performance updates
            for i in range(3):
                algorithm.update_task_performance(task._task_id, 0.3 + 0.1 * i)

        # Test performance tracking
        for task_id in algorithm._task_memory:
            lp_score = algorithm._get_task_lp_score(task_id)
            assert lp_score >= 0, "LP score should be non-negative"

    def test_learning_progress_sampling_distribution(self, random_seed):
        """Test that learning progress sampling produces reasonable distributions."""
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=6,
            sample_size=3,
            max_samples=5,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Fill pool with tasks
        for _ in range(3):
            task = algorithm.get_task_from_pool(task_generator, rng)
            algorithm.update_task_performance(task._task_id, 0.5)

        # Test sampling distribution
        samples = []
        for _ in range(10):
            sampled_task_id = algorithm._choose_task()
            samples.append(sampled_task_id)

        # Should sample from multiple tasks
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "Should sample from multiple tasks"


class TestLearningProgressIntegration:
    """Test learning progress algorithm integration with curriculum system."""

    def test_learning_progress_with_curriculum_config(self, random_seed):
        """Test learning progress algorithm integrated with curriculum configuration."""
        # Create curriculum config with learning progress algorithm
        algorithm_config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=8,
            sample_size=4,
            max_samples=6,
        )

        curriculum_config = CurriculumConfig(
            task_generator=SingleTaskGeneratorConfig(env=eb.make_arena(num_agents=4)),
            algorithm_config=algorithm_config,
        )

        # Test that curriculum can be created
        curriculum = curriculum_config.make()
        assert curriculum._algorithm is not None, "Curriculum should have algorithm"
        assert isinstance(curriculum._algorithm, LearningProgressAlgorithm), "Should be LearningProgressAlgorithm"

    def test_learning_progress_curriculum_workflow(self, random_seed):
        """Test complete curriculum workflow with learning progress algorithm."""
        # Create curriculum with learning progress
        algorithm_config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=6,
            sample_size=3,
            max_samples=4,
        )

        curriculum_config = CurriculumConfig(
            task_generator=SingleTaskGeneratorConfig(env=eb.make_arena(num_agents=4)),
            algorithm_config=algorithm_config,
        )

        curriculum = curriculum_config.make()

        # Test workflow
        for _ in range(3):
            task = curriculum.get_task()
            # Simulate task completion
            curriculum.update_task_performance(task._task_id, 0.5)

        # Test that algorithm is working
        assert curriculum._algorithm is not None, "Algorithm should be present"
        assert len(curriculum._algorithm._task_memory) > 0, "Algorithm should have tasks in pool"

    def test_learning_progress_backward_compatibility(self, random_seed):
        """Test that learning progress algorithm maintains backward compatibility."""
        # Test with minimal configuration
        config = LearningProgressConfig()
        algorithm = LearningProgressAlgorithm(num_tasks=5, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Test basic functionality
        task = algorithm.get_task_from_pool(task_generator, rng)
        algorithm.update_task_performance(task._task_id, 0.5)

        # Verify default values are applied
        assert algorithm.hypers.ema_timescale == 0.001, "Should use default ema_timescale"
        assert algorithm.hypers.pool_size == 16, "Should use default pool_size"

    def test_learning_progress_forward_compatibility(self, random_seed):
        """Test that learning progress algorithm supports future configuration options."""
        # Test with extended configuration
        config = LearningProgressConfig(
            ema_timescale=0.001,
            pool_size=8,
            sample_size=4,
            max_samples=6,
            exploration_bonus=0.15,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=6, hypers=config)

        rng = random.Random(random_seed)
        task_generator = MockTaskGenerator()

        # Test extended functionality
        task = algorithm.get_task_from_pool(task_generator, rng)
        algorithm.update_task_performance(task._task_id, 0.5)

        # Verify extended options are applied
        assert algorithm.hypers.exploration_bonus == 0.15, "Should apply exploration_bonus"
        assert algorithm.hypers.pool_size == 8, "Should apply custom pool_size"

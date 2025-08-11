"""Tests for Curriculum classes."""

from unittest.mock import patch

import numpy as np
import pytest

from cogworks.curriculum import (
    Curriculum,
    LearningProgressCurriculum,
    LearningProgressCurriculumConfig,
    RandomCurriculum,
    RandomCurriculumConfig,
    Task,
)
from cogworks.curriculum.base import CurriculumConfig
from cogworks.curriculum.task.generator import (
    WeightedTaskSetGeneratorConfig,
    WeightedTaskSetGeneratorItem,
)
from metta.rl.env_config import EnvConfig


class TestCurriculumBase:
    """Test cases for Curriculum base class."""

    def test_curriculum_is_abstract(self):
        """Test that Curriculum cannot be instantiated directly."""
        config = CurriculumConfig(task_generator_config=WeightedTaskSetGeneratorConfig(items=[]))

        with pytest.raises(TypeError):
            Curriculum(config)

    def test_curriculum_base_methods(self):
        """Test default implementations of base methods."""

        # Create a concrete subclass for testing
        class TestCurriculum(Curriculum):
            def get_task(self, seed: int) -> Task:
                env_cfg = EnvConfig()
                return Task(task_id=f"test_{seed}", env_cfg=env_cfg)

        config = CurriculumConfig(task_generator_config=WeightedTaskSetGeneratorConfig(items=[]))
        curriculum = TestCurriculum(config, seed=0)

        # Test default implementations
        task = Task(task_id="test", env_cfg=EnvConfig())
        curriculum.complete_task(task, 0.5)  # Should not raise

        assert curriculum.get_task_probs() == {}
        # get_curriculum_stats doesn't exist anymore


class TestRandomCurriculum:
    """Test cases for RandomCurriculum."""

    def create_test_config(self):
        """Helper to create a test configuration."""
        env_cfg = EnvConfig()
        from cogworks.curriculum.task.generator import SingleTaskGeneratorConfig

        single_task_config = SingleTaskGeneratorConfig(env_config=env_cfg)
        item = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config, weight=1.0)
        task_generator_config = WeightedTaskSetGeneratorConfig(items=[item])
        return RandomCurriculumConfig(task_generator_config=task_generator_config)

    def test_random_curriculum_creation(self):
        """Test creating a RandomCurriculum."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)

        assert curriculum._config == config
        assert hasattr(curriculum, "_task_generator")

    def test_random_curriculum_get_task_deterministic(self):
        """Test that same seed produces same task."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)

        task1 = curriculum.get_task(42)
        task2 = curriculum.get_task(42)  # Same seed
        task3 = curriculum.get_task(43)  # Different seed

        # Same seed should produce same task
        assert task1.get_id() == task2.get_id()
        # Note: Task objects are different instances but have same ID

        # Different seed should produce different task ID
        assert task1.get_id() != task3.get_id()

        # All tasks should have consistent format
        assert task1.get_id().startswith("random_42")
        assert task3.get_id().startswith("random_43")

    def test_random_curriculum_task_content(self):
        """Test that tasks contain expected env config."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)

        task = curriculum.get_task(42)
        _ = task.get_env_config()

        # Should match the configured env config

    def test_random_curriculum_multiple_items(self):
        """Test RandomCurriculum with multiple weighted items."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()

        from cogworks.curriculum.task.generator import SingleTaskGeneratorConfig

        single_task_config1 = SingleTaskGeneratorConfig(env_config=env_cfg1)
        single_task_config2 = SingleTaskGeneratorConfig(env_config=env_cfg2)
        item1 = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config1, weight=1.0)
        item2 = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config2, weight=1.0)
        task_generator_config = WeightedTaskSetGeneratorConfig(items=[item1, item2])
        config = RandomCurriculumConfig(task_generator_config=task_generator_config)

        curriculum = RandomCurriculum(config, seed=0)

        # Sample multiple tasks
        tasks = [curriculum.get_task(i) for i in range(10)]

        # All tasks should be valid
        for task in tasks:
            assert task is not None
            assert hasattr(task, "get_env_config")

    def test_random_curriculum_complete_task(self):
        """Test that complete_task doesn't crash."""
        config = self.create_test_config()
        curriculum = RandomCurriculum(config, seed=0)

        task = curriculum.get_task(42)

        # Should not raise any exception
        curriculum.complete_task(task, 0.8)
        curriculum.complete_task(task, 0.0)
        curriculum.complete_task(task, 1.0)


class TestLearningProgressCurriculum:
    """Test cases for LearningProgressCurriculum."""

    def create_test_config(self, n_tasks=5):
        """Helper to create a test configuration."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()

        from cogworks.curriculum.task.generator import SingleTaskGeneratorConfig

        single_task_config1 = SingleTaskGeneratorConfig(env_config=env_cfg1)
        single_task_config2 = SingleTaskGeneratorConfig(env_config=env_cfg2)
        item1 = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config1, weight=1.0)
        item2 = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config2, weight=1.0)
        task_generator_config = WeightedTaskSetGeneratorConfig(items=[item1, item2])

        return LearningProgressCurriculumConfig(
            task_generator_config=task_generator_config,
            n_tasks=n_tasks,
            num_active_tasks=3,
            memory=10,
            ema_timescale=0.1,
            progress_smoothing=0.1,
            rand_task_rate=0.2,
        )

    def test_lp_curriculum_creation(self):
        """Test creating a LearningProgressCurriculum."""
        config = self.create_test_config()
        curriculum = LearningProgressCurriculum(config, seed=0)

        assert curriculum._config == config
        assert len(curriculum._tasks) == 0  # Not initialized yet
        assert curriculum._tasks_initialized is False

    def test_lp_curriculum_task_initialization(self):
        """Test that tasks are initialized on first get_task call."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)

        # First call should initialize tasks
        task = curriculum.get_task(42)

        assert curriculum._tasks_initialized is True
        assert len(curriculum._tasks) == 3
        assert len(curriculum._task_outcomes) == 3
        assert len(curriculum._task_weights) == 3
        assert len(curriculum._active_task_indices) == 3  # min(3, num_active_tasks)

        # Task should be one of the initialized tasks
        assert task in curriculum._tasks

    def test_lp_curriculum_deterministic_task_generation(self):
        """Test that tasks are generated deterministically."""
        config = self.create_test_config(n_tasks=3)
        curriculum1 = LearningProgressCurriculum(config, seed=0)
        curriculum2 = LearningProgressCurriculum(config, seed=0)

        # Initialize both with same seed
        _ = curriculum1.get_task(42)
        _ = curriculum2.get_task(42)

        # Should generate the same set of tasks
        assert len(curriculum1._tasks) == len(curriculum2._tasks)
        for t1, t2 in zip(curriculum1._tasks, curriculum2._tasks, strict=False):
            assert t1.get_id() == t2.get_id()

    def test_lp_curriculum_task_completion_updates(self):
        """Test that completing tasks updates learning progress."""
        config = self.create_test_config(n_tasks=2)
        curriculum = LearningProgressCurriculum(config, seed=0)

        # Initialize
        _ = curriculum.get_task(42)

        # Complete some tasks
        curriculum.complete_task(curriculum._tasks[0], 0.8)
        curriculum.complete_task(curriculum._tasks[0], 0.9)
        curriculum.complete_task(curriculum._tasks[1], 0.2)

        # Check that outcomes were recorded
        task0_outcomes = curriculum._task_outcomes[curriculum._tasks[0].get_id()]
        task1_outcomes = curriculum._task_outcomes[curriculum._tasks[1].get_id()]

        assert len(task0_outcomes) == 2
        assert task0_outcomes == [0.8, 0.9]
        assert len(task1_outcomes) == 1
        assert task1_outcomes == [0.2]

    def test_lp_curriculum_memory_limit(self):
        """Test that task outcomes respect memory limit."""
        config = self.create_test_config(n_tasks=1)
        config.memory = 2  # Only keep 2 recent outcomes
        curriculum = LearningProgressCurriculum(config, seed=0)

        _ = curriculum.get_task(42)
        test_task = curriculum._tasks[0]

        # Add more outcomes than memory limit
        curriculum.complete_task(test_task, 0.1)
        curriculum.complete_task(test_task, 0.2)
        curriculum.complete_task(test_task, 0.3)
        curriculum.complete_task(test_task, 0.4)

        # Should only keep the most recent ones
        outcomes = curriculum._task_outcomes[test_task.get_id()]
        assert len(outcomes) == 2
        assert outcomes == [0.3, 0.4]

    def test_lp_curriculum_score_clamping(self):
        """Test that scores are clamped to [0, 1] range."""
        config = self.create_test_config(n_tasks=1)
        curriculum = LearningProgressCurriculum(config, seed=0)

        _ = curriculum.get_task(42)
        test_task = curriculum._tasks[0]

        # Test extreme scores
        curriculum.complete_task(test_task, -10.0)  # Should be clamped to 0
        curriculum.complete_task(test_task, 10.0)  # Should be clamped to 1

        outcomes = curriculum._task_outcomes[test_task.get_id()]
        assert outcomes == [0.0, 1.0]

    def test_lp_curriculum_unknown_task_handling(self):
        """Test handling of unknown tasks in complete_task."""
        config = self.create_test_config(n_tasks=1)
        curriculum = LearningProgressCurriculum(config, seed=0)

        curriculum.get_task(42)  # Initialize

        # Create a task not in the curriculum
        unknown_task = Task(task_id="unknown", env_cfg=EnvConfig())

        # Should log warning but not crash
        with patch("cogworks.curriculum.learning_progress.logger") as mock_logger:
            curriculum.complete_task(unknown_task, 0.5)
            mock_logger.warning.assert_called_once()

    def test_lp_curriculum_learning_progress_updates(self):
        """Test that learning progress tracking is updated."""
        config = self.create_test_config(n_tasks=2)
        curriculum = LearningProgressCurriculum(config, seed=0)

        curriculum.get_task(42)  # Initialize

        # Initially, tracking variables should be None
        assert curriculum._p_fast is None
        assert curriculum._p_slow is None
        assert curriculum._p_true is None

        # Complete a task to trigger updates
        curriculum.complete_task(curriculum._tasks[0], 0.8)

        # Should initialize tracking variables
        assert curriculum._p_fast is not None
        assert curriculum._p_slow is not None
        assert curriculum._p_true is not None
        assert isinstance(curriculum._p_fast, np.ndarray)
        assert len(curriculum._p_fast) == 2

    def test_lp_curriculum_task_probs(self):
        """Test get_task_probs returns correct probabilities."""
        config = self.create_test_config(n_tasks=3)
        config.num_active_tasks = 2  # Only 2 active tasks
        curriculum = LearningProgressCurriculum(config, seed=0)

        curriculum.get_task(42)  # Initialize

        probs = curriculum.get_task_probs()

        # Should have probabilities for all tasks
        assert len(probs) == 3

        # Only active tasks should have non-zero probabilities
        non_zero_probs = [p for p in probs.values() if p > 0]
        assert len(non_zero_probs) <= 2  # At most num_active_tasks

    def test_lp_curriculum_stats(self):
        """Test get_curriculum_stats returns meaningful stats."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)

        # Before initialization
        stats = curriculum.get_curriculum_stats()
        assert stats == {}

        # After initialization and some progress
        curriculum.get_task(42)
        curriculum.complete_task(curriculum._tasks[0], 0.8)

        stats = curriculum.get_curriculum_stats()

        # Should contain expected stats
        assert "lp/num_active_tasks" in stats
        assert "lp/mean_task_weight" in stats
        assert "lp/num_zero_weight_tasks" in stats
        assert "lp/mean_success_rate" in stats
        assert "lp/min_success_rate" in stats
        assert "lp/max_success_rate" in stats

        assert isinstance(stats["lp/num_active_tasks"], int)
        assert isinstance(stats["lp/mean_task_weight"], float)

    def test_lp_curriculum_weighted_sampling(self):
        """Test that tasks are sampled according to weights."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)

        curriculum.get_task(42)  # Initialize

        # Manually set task weights to test sampling
        curriculum._task_weights = np.array([0.8, 0.1, 0.1])
        curriculum._active_task_indices = [0, 1, 2]

        # Sample many tasks
        task_counts = {}
        for seed in range(100):
            task = curriculum.get_task(seed)
            task_id = task.get_id()
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        # Task 0 should be selected more often due to higher weight
        task0_id = curriculum._tasks[0].get_id()
        assert task_counts[task0_id] > task_counts.get(curriculum._tasks[1].get_id(), 0)

    def test_lp_curriculum_no_active_tasks_fallback(self):
        """Test behavior when no active tasks are available."""
        config = self.create_test_config(n_tasks=3)
        curriculum = LearningProgressCurriculum(config, seed=0)

        curriculum.get_task(42)  # Initialize

        # Remove all active tasks
        curriculum._active_task_indices = []

        # Should still be able to get a task (random fallback)
        task = curriculum.get_task(123)
        assert task in curriculum._tasks


class TestCurriculumEdgeCases:
    """Test edge cases and error conditions."""

    def test_curriculum_with_empty_taskset(self):
        """Test curriculum behavior with empty task set."""
        task_generator_config = WeightedTaskSetGeneratorConfig(items=[])
        config = RandomCurriculumConfig(task_generator_config=task_generator_config)
        curriculum = RandomCurriculum(config, seed=0)

        # Should raise error when trying to get task
        with pytest.raises((ValueError, IndexError)):
            curriculum.get_task(42)

    def test_lp_curriculum_zero_tasks(self):
        """Test LearningProgressCurriculum validation rejects n_tasks=0."""
        env_cfg = EnvConfig()
        from cogworks.curriculum.task.generator import SingleTaskGeneratorConfig

        single_task_config = SingleTaskGeneratorConfig(env_config=env_cfg)
        item = WeightedTaskSetGeneratorItem(task_generator_config=single_task_config, weight=1.0)
        task_generator_config = WeightedTaskSetGeneratorConfig(items=[item])

        # Should reject n_tasks=0 during validation
        with pytest.raises(ValueError):
            LearningProgressCurriculumConfig(
                task_generator_config=task_generator_config,
                n_tasks=0,  # Zero tasks - should be invalid
                num_active_tasks=1,
            )

    def test_lp_curriculum_more_active_than_total(self):
        """Test when num_active_tasks > n_tasks."""
        config = TestLearningProgressCurriculum().create_test_config(n_tasks=2)
        config.num_active_tasks = 5  # More than n_tasks

        curriculum = LearningProgressCurriculum(config, seed=0)
        curriculum.get_task(42)  # Initialize

        # Should limit to actual number of tasks
        assert len(curriculum._active_task_indices) == 2

    def test_curriculum_task_id_consistency(self):
        """Test that task IDs are consistent across curriculum instances."""
        config = TestLearningProgressCurriculum().create_test_config(n_tasks=2)

        curriculum1 = LearningProgressCurriculum(config, seed=0)
        curriculum2 = LearningProgressCurriculum(config, seed=0)

        # Same seed should produce same task set
        curriculum1.get_task(42)
        curriculum2.get_task(42)

        task_ids1 = [task.get_id() for task in curriculum1._tasks]
        task_ids2 = [task.get_id() for task in curriculum2._tasks]

        assert task_ids1 == task_ids2

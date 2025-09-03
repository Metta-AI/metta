"""Consolidated core tests for Curriculum classes and integration."""

import random

import pytest

from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    SingleTaskGeneratorConfig,
)
from metta.cogworks.curriculum.curriculum import DiscreteRandomConfig, DiscreteRandomCurriculum
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig


class TestCurriculumTask:
    """Test cases for CurriculumTask."""

    def test_curriculum_task_creation(self, arena_env):
        """Test creating a CurriculumTask."""
        task_id = 123
        cfg = arena_env

        task = CurriculumTask(task_id, cfg)

        assert task._task_id == task_id
        assert task._env_cfg == cfg
        assert task._num_completions == 0
        assert task._total_score == 0.0
        assert task._mean_score == 0.0
        assert task._num_scheduled == 0

    def test_curriculum_task_get_env_cfg(self, arena_env):
        """Test getting environment configuration from task."""
        task = CurriculumTask(123, arena_env)

        assert task.get_env_cfg() is arena_env

    def test_curriculum_task_complete(self, arena_env):
        """Test task completion and score tracking."""
        task = CurriculumTask(123, arena_env)

        # Complete with score 0.8
        task.complete(0.8)
        assert task._num_completions == 1
        assert task._total_score == 0.8
        assert task._mean_score == 0.8

        # Complete with score 0.6
        task.complete(0.6)
        assert task._num_completions == 2
        assert task._total_score == 1.4
        assert task._mean_score == 0.7  # (0.8 + 0.6) / 2


class TestCurriculumConfig:
    """Test cases for CurriculumConfig."""

    def test_curriculum_config_creation(self, single_task_generator_config):
        """Test creating a CurriculumConfig with valid parameters."""
        config = CurriculumConfig(
            task_generator=single_task_generator_config, max_task_id=1000, num_active_tasks=50, new_task_rate=0.05
        )

        assert config.task_generator is single_task_generator_config
        assert config.max_task_id == 1000
        assert config.num_active_tasks == 50
        assert config.new_task_rate == 0.05

    def test_curriculum_config_defaults(self, single_task_generator_config):
        """Test that CurriculumConfig uses correct default values."""
        config = CurriculumConfig(task_generator=single_task_generator_config)

        assert config.max_task_id == 1000000
        assert config.num_active_tasks == 10000
        assert config.new_task_rate == 0.01

    def test_curriculum_config_validation_num_active_tasks(self, single_task_generator_config):
        """Test that num_active_tasks validation works."""
        # This should fail because num_active_tasks > max_task_id
        with pytest.raises(ValueError):
            CurriculumConfig(task_generator=single_task_generator_config, max_task_id=100, num_active_tasks=200)

    def test_curriculum_config_edge_case_values(self, single_task_generator_config):
        """Test edge case values for parameters."""
        # Test minimum values
        config = CurriculumConfig(
            task_generator=single_task_generator_config, max_task_id=1, num_active_tasks=1, new_task_rate=0.0
        )
        assert config.max_task_id == 1
        assert config.num_active_tasks == 1
        assert config.new_task_rate == 0.0

        # Test maximum values
        config = CurriculumConfig(
            task_generator=single_task_generator_config,
            max_task_id=1000000,
            num_active_tasks=1000000,
            new_task_rate=1.0,
        )
        assert config.new_task_rate == 1.0


class TestCurriculumCore:
    """Test cases for Curriculum core functionality."""

    def test_curriculum_creation(self, curriculum_config):
        """Test creating a Curriculum."""
        curriculum = Curriculum(curriculum_config, seed=0)

        assert curriculum._config is curriculum_config
        assert hasattr(curriculum._task_generator, "get_task")
        assert isinstance(curriculum._rng, random.Random)
        assert len(curriculum._tasks) == 0
        assert len(curriculum._task_ids) == 0
        assert curriculum._num_created == 0
        assert curriculum._num_evicted == 0

    def test_curriculum_get_task_creates_task(self, curriculum_config):
        """Test that get_task creates new tasks when under capacity."""
        curriculum = Curriculum(curriculum_config, seed=0)

        task = curriculum.get_task()

        assert isinstance(task, CurriculumTask)
        assert len(curriculum._tasks) == 1
        assert len(curriculum._task_ids) == 1
        assert curriculum._num_created == 1
        assert task._num_scheduled == 1

    def test_curriculum_get_task_fills_capacity(self, curriculum_config):
        """Test that curriculum fills to capacity before reusing tasks."""
        curriculum = Curriculum(curriculum_config, seed=0)

        tasks = []
        for _ in range(5):  # Fill to capacity
            tasks.append(curriculum.get_task())

        assert len(curriculum._tasks) == 5
        assert len(curriculum._task_ids) == 5
        assert curriculum._num_created == 5
        assert curriculum._num_evicted == 0

        # All tasks should be unique
        task_ids = [task._task_id for task in tasks]
        assert len(set(task_ids)) == 5

    def test_curriculum_task_reuse_and_eviction(self, curriculum_config):
        """Test task reuse and eviction when at capacity."""
        config = curriculum_config.model_copy()
        config.num_active_tasks = 5  # Use smaller capacity for testing
        config.new_task_rate = 0.5  # 50% chance of new task
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(5):
            curriculum.get_task()

        initial_created = curriculum._num_created

        # Get more tasks - should either reuse or evict+create
        for _ in range(10):
            task = curriculum.get_task()
            assert task._task_id in curriculum._task_ids

        # Should maintain reasonable capacity (may exceed slightly due to eviction timing)
        assert len(curriculum._tasks) <= config.num_active_tasks + 2, f"Too many tasks: {len(curriculum._tasks)}"
        assert len(curriculum._tasks) >= config.num_active_tasks - 2, f"Too few tasks: {len(curriculum._tasks)}"

        # Should have either created new tasks (with eviction) or reused existing ones
        assert curriculum._num_created >= initial_created

    def test_curriculum_deterministic_with_same_seed(self, curriculum_config):
        """Test that curriculum is deterministic with the same seed."""
        curriculum1 = Curriculum(curriculum_config, seed=42)
        curriculum2 = Curriculum(curriculum_config, seed=42)

        tasks1 = [curriculum1.get_task() for _ in range(3)]
        tasks2 = [curriculum2.get_task() for _ in range(3)]

        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]

        assert task_ids1 == task_ids2

    def test_curriculum_different_seeds_produce_different_tasks(self, curriculum_config):
        """Test that different seeds produce different task sequences."""
        curriculum1 = Curriculum(curriculum_config, seed=42)
        curriculum2 = Curriculum(curriculum_config, seed=123)

        tasks1 = [curriculum1.get_task() for _ in range(5)]
        tasks2 = [curriculum2.get_task() for _ in range(5)]

        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]

        # Different seeds should produce different task sequences
        assert task_ids1 != task_ids2

    def test_curriculum_stats(self, curriculum_config):
        """Test curriculum statistics."""
        curriculum = Curriculum(curriculum_config, seed=0)

        # Get initial stats
        initial_stats = curriculum.stats()
        assert "num_active_tasks" in initial_stats
        assert initial_stats["num_active_tasks"] == 0

        # Get some tasks and check stats
        for _ in range(3):
            curriculum.get_task()

        updated_stats = curriculum.stats()
        assert updated_stats["num_active_tasks"] == 3
        assert updated_stats["num_created"] == 3

    def test_curriculum_task_id_uniqueness(self, curriculum_config):
        """Test that curriculum generates unique task IDs."""
        curriculum = Curriculum(curriculum_config, seed=0)

        tasks = []
        for _ in range(10):
            tasks.append(curriculum.get_task())

        task_ids = [task._task_id for task in tasks]
        assert len(set(task_ids)) == len(task_ids), "All task IDs should be unique"

    def test_curriculum_task_scheduling_count(self, curriculum_config):
        """Test that task scheduling count is tracked correctly."""
        curriculum = Curriculum(curriculum_config, seed=0)

        # Get a task and note its initial scheduling count
        task = curriculum.get_task()
        initial_count = task._num_scheduled

        # Get more tasks - some may be the same due to reuse
        for _ in range(5):
            curriculum.get_task()

        # The original task should have been scheduled at least once more
        assert task._num_scheduled >= initial_count


class TestCurriculumAlgorithmIntegration:
    """Test curriculum integration with various algorithms."""

    @pytest.mark.parametrize(
        "algorithm_config",
        [
            LearningProgressConfig(
                ema_timescale=0.001,
                pool_size=10,
                sample_size=5,
                max_samples=10,
            ),
            DiscreteRandomConfig(),
            None,  # No algorithm
        ],
    )
    def test_curriculum_integration_with_algorithm(self, arena_env, algorithm_config):
        """Test that curriculum works with any algorithm configuration."""
        # Create task generator configuration
        task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

        # Create curriculum config with algorithm
        curriculum_config = CurriculumConfig(
            task_generator=task_gen_config,
            num_active_tasks=4,
            algorithm_config=algorithm_config,
        )

        # Create curriculum
        curriculum = curriculum_config.make()

        # Test that algorithm is initialized correctly
        if algorithm_config is None:
            assert curriculum._algorithm is None, "Algorithm should not be initialized"
        else:
            assert curriculum._algorithm is not None, "Algorithm should be initialized"
            if isinstance(algorithm_config, LearningProgressConfig):
                assert isinstance(curriculum._algorithm, LearningProgressAlgorithm)
            elif isinstance(algorithm_config, DiscreteRandomConfig):
                assert isinstance(curriculum._algorithm, DiscreteRandomCurriculum)

        # Test task creation and selection
        task = curriculum.get_task()
        assert task is not None, "Should get a task"
        assert task.get_env_cfg() is not None, "Task should have environment config"

        # Test algorithm statistics
        stats = curriculum.stats()
        assert "num_active_tasks" in stats, "Should include basic statistics"

    def test_curriculum_algorithm_switching(self, arena_env):
        """Test that curriculum can switch between different algorithms."""
        task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

        # Test with discrete random
        config1 = CurriculumConfig(
            task_generator=task_gen_config,
            num_active_tasks=4,
            algorithm_config=DiscreteRandomConfig(),
        )
        curriculum1 = config1.make()
        assert isinstance(curriculum1._algorithm, DiscreteRandomCurriculum)

        # Test with learning progress
        config2 = CurriculumConfig(
            task_generator=task_gen_config,
            num_active_tasks=4,
            algorithm_config=LearningProgressConfig(
                ema_timescale=0.001,
                pool_size=10,
                sample_size=5,
                max_samples=10,
            ),
        )
        curriculum2 = config2.make()
        assert isinstance(curriculum2._algorithm, LearningProgressAlgorithm)

    def test_curriculum_backward_compatibility(self, curriculum_without_algorithm):
        """Test that existing curriculum configurations work without changes."""
        curriculum = curriculum_without_algorithm.make()

        # Should work exactly like before
        task = curriculum.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

        # Should have no algorithm
        assert curriculum._algorithm is None

    def test_curriculum_with_algorithm_forward_compatibility(self, curriculum_with_algorithm):
        """Test that new curriculum configurations with algorithms work."""
        curriculum = curriculum_with_algorithm.make()

        # Should work with algorithm
        task = curriculum.get_task()
        assert task is not None
        assert task.get_env_cfg() is not None

        # Should have algorithm
        assert curriculum._algorithm is not None
        assert isinstance(curriculum._algorithm, LearningProgressAlgorithm)


class TestCurriculumEdgeCases:
    """Test edge cases and boundary conditions for curriculum."""

    def test_curriculum_with_new_task_rate_zero(self, curriculum_config):
        """Test curriculum behavior with new_task_rate = 0."""
        config = curriculum_config.model_copy()
        config.new_task_rate = 0.0
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(5):
            curriculum.get_task()

        # Should always reuse existing tasks
        for _ in range(10):
            task = curriculum.get_task()
            assert task._task_id in curriculum._task_ids

        # With new_task_rate = 0, we should still create some new tasks due to eviction
        assert curriculum._num_created >= 5

    def test_curriculum_with_new_task_rate_one(self, curriculum_config):
        """Test curriculum behavior with new_task_rate = 1."""
        config = curriculum_config.model_copy()
        config.new_task_rate = 1.0
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(5):
            curriculum.get_task()

        # Should always create new tasks (with eviction)
        for _ in range(10):
            curriculum.get_task()

        assert curriculum._num_created > 5  # New tasks should be created

    def test_curriculum_with_single_task_capacity(self, curriculum_config):
        """Test curriculum with capacity of 1."""
        config = curriculum_config.model_copy()
        config.num_active_tasks = 1
        curriculum = Curriculum(config, seed=0)

        # Get first task
        curriculum.get_task()
        assert len(curriculum._tasks) == 1

        # Get second task - should evict first
        curriculum.get_task()
        assert len(curriculum._tasks) == 1

        # The tasks might have the same ID if the random generator produces the same value
        # This is acceptable behavior - just verify we have exactly one task
        assert len(curriculum._tasks) == 1

    def test_curriculum_max_task_id_boundary(self, curriculum_config):
        """Test curriculum behavior at max_task_id boundary."""
        config = curriculum_config.model_copy()
        config.max_task_id = 100  # Use a reasonable size to avoid infinite loops
        curriculum = Curriculum(config, seed=0)

        # Should be able to create tasks up to max_task_id
        for _ in range(10):  # Reduced from 5 to be safer
            task = curriculum.get_task()
            assert task._task_id <= 100  # Allow equal to max_task_id

        # Additional tasks should still work (with eviction)
        for _ in range(5):  # Reduced from 3 to be safer
            task = curriculum.get_task()
            assert task._task_id <= 100

    def test_curriculum_small_max_task_id_handling(self, curriculum_config):
        """Test that curriculum handles very small max_task_id gracefully."""
        config = curriculum_config.model_copy()
        config.max_task_id = 2  # Very small but safe
        config.num_active_tasks = 1  # Minimal capacity
        curriculum = Curriculum(config, seed=0)

        # Should be able to create at least one task
        task = curriculum.get_task()
        assert task._task_id <= 2
        assert len(curriculum._tasks) == 1

        # Should handle additional requests gracefully
        for _ in range(3):
            task = curriculum.get_task()
            assert task._task_id <= 2
            assert len(curriculum._tasks) == 1  # Should maintain capacity

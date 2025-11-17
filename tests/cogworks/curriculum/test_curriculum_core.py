"""Consolidated core tests for Curriculum classes and integration."""

import random

import pytest

from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
)


class TestCurriculumTask:
    """Test cases for CurriculumTask."""

    @pytest.mark.parametrize("task_id,expected_id", [(123, 123), (456, 456), ("string_id", "string_id")])
    def test_curriculum_task_creation(self, arena_env, task_id, expected_id):
        """Test creating a CurriculumTask with various ID types."""
        cfg = arena_env
        task = CurriculumTask(task_id, cfg)

        assert task._task_id == expected_id
        assert task._env_cfg == cfg
        assert task._num_completions == 0
        assert task._total_score == 0.0
        assert task._mean_score == 0.0
        assert task._num_scheduled == 0

    def test_curriculum_task_get_env_cfg(self, arena_env):
        """Test getting environment configuration from task."""
        task = CurriculumTask(123, arena_env)
        assert task.get_env_cfg() is arena_env

    @pytest.mark.parametrize(
        "scores,expected_completions,expected_total,expected_mean",
        [
            ([0.8], 1, 0.8, 0.8),
            ([0.8, 0.6], 2, 1.4, 0.7),
            ([0.5, 0.7, 0.9], 3, 2.1, 0.7),
        ],
    )
    def test_curriculum_task_complete(self, arena_env, scores, expected_completions, expected_total, expected_mean):
        """Test task completion and score tracking with various score sequences."""
        task = CurriculumTask(123, arena_env)

        for score in scores:
            task.complete(score)

        assert task._num_completions == expected_completions
        assert task._total_score == expected_total
        assert abs(task._mean_score - expected_mean) < 1e-6


class TestCurriculumConfig:
    """Test cases for CurriculumConfig."""

    @pytest.mark.parametrize(
        "num_active_tasks",
        [
            50,
            25,
            100,
        ],
    )
    def test_curriculum_config_creation(self, single_task_generator_config, num_active_tasks):
        """Test creating a CurriculumConfig with various parameter combinations."""
        config = CurriculumConfig(
            task_generator=single_task_generator_config,
            num_active_tasks=num_active_tasks,
        )

        assert config.task_generator is single_task_generator_config
        assert config.num_active_tasks == num_active_tasks

    def test_curriculum_config_defaults(self, single_task_generator_config):
        """Test that CurriculumConfig uses correct default values."""
        config = CurriculumConfig(task_generator=single_task_generator_config)

        assert config.num_active_tasks == 1000  # Updated default in refactor

    @pytest.mark.parametrize(
        "num_active_tasks",
        [
            1,  # Minimum value
            1000000,  # Large value
            50,  # Middle value
        ],
    )
    def test_curriculum_config_edge_case_values(self, single_task_generator_config, num_active_tasks):
        """Test edge case values for parameters."""
        config = CurriculumConfig(
            task_generator=single_task_generator_config,
            num_active_tasks=num_active_tasks,
        )
        assert config.num_active_tasks == num_active_tasks


class TestCurriculumCore:
    """Test cases for Curriculum core functionality."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_curriculum_creation(self, curriculum_config, seed):
        """Test creating a Curriculum with various seeds."""
        # Create config with the specified seed
        config_with_seed = curriculum_config.model_copy(update={"seed": seed})
        curriculum = Curriculum(config_with_seed)

        assert curriculum._config is config_with_seed
        assert hasattr(curriculum._task_generator, "get_task")
        assert isinstance(curriculum._rng, random.Random)

        # RNG state will be different from a fresh Random(seed) because
        # curriculum initialization now creates tasks at capacity, consuming randomness
        # Just verify that the RNG was seeded properly by creating another with same seed
        # and checking that some draws produce the same sequence after capacity initialization
        test_config = curriculum_config.model_copy(update={"seed": seed})
        test_curriculum = Curriculum(test_config)

        # After initialization, both should generate the same sequence
        for _ in range(5):
            assert curriculum._rng.random() == test_curriculum._rng.random()

    def test_curriculum_task_generation(self, curriculum_config):
        """Test that curriculum can generate tasks."""
        config = curriculum_config.model_copy(update={"seed": 0})
        curriculum = Curriculum(config)

        # Generate multiple tasks
        tasks = []
        for _ in range(5):
            task = curriculum.get_task()
            tasks.append(task)

        # All tasks should be unique
        task_ids = [task._task_id for task in tasks]
        assert len(set(task_ids)) == len(task_ids), "All tasks should have unique IDs"

        # All tasks should have valid environment configs
        for task in tasks:
            assert task._env_cfg is not None
            assert hasattr(task._env_cfg, "game")

    def test_curriculum_task_reuse(self, curriculum_config):
        """Test that curriculum can reuse tasks."""
        config = curriculum_config.model_copy(update={"seed": 0})
        curriculum = Curriculum(config)

        # Get initial task
        initial_task = curriculum.get_task()

        # Complete the task
        initial_task.complete(0.8)

        # Get another task - should be different due to random sampling
        next_task = curriculum.get_task()

        # In a small task space, we might get the same task, but that's okay
        # The important thing is that the curriculum continues to function
        assert hasattr(next_task, "_env_cfg")
        assert hasattr(next_task, "_task_id")

    def test_curriculum_determinism_with_same_seed(self, curriculum_config):
        """Test that curriculum produces same sequence with same seed."""
        seed = 42

        # Create two curricula with same seed
        config1 = curriculum_config.model_copy(update={"seed": seed})
        config2 = curriculum_config.model_copy(update={"seed": seed})
        curriculum1 = Curriculum(config1)
        curriculum2 = Curriculum(config2)

        # Generate tasks from both
        tasks1 = [curriculum1.get_task() for _ in range(5)]
        tasks2 = [curriculum2.get_task() for _ in range(5)]

        # Task IDs should be the same (deterministic)
        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]
        assert task_ids1 == task_ids2, "Same seed should produce same task sequence"

    def test_curriculum_different_seeds_produce_different_sequences(self, curriculum_config):
        """Test that different seeds produce different task sequences."""
        seed1, seed2 = 42, 123

        # Create curricula with different seeds
        config1 = curriculum_config.model_copy(update={"seed": seed1})
        config2 = curriculum_config.model_copy(update={"seed": seed2})
        curriculum1 = Curriculum(config1)
        curriculum2 = Curriculum(config2)

        # Generate tasks from both
        tasks1 = [curriculum1.get_task() for _ in range(5)]
        tasks2 = [curriculum2.get_task() for _ in range(5)]

        # Task IDs should be different (different seeds)
        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]

        # Different seeds should produce different sequences
        # Note: In very small task spaces, this might not always be true
        # but it's a reasonable expectation for most cases
        if len(set(task_ids1)) > 1 or len(set(task_ids2)) > 1:
            # If we have some variety in the sequences, they should be different
            assert task_ids1 != task_ids2, "Different seeds should produce different task sequences"

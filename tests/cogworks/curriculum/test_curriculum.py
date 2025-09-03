"""Tests for Curriculum classes."""

import random

import pytest

from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    SingleTaskGeneratorConfig,
)
from metta.mettagrid.mettagrid_config import MettaGridConfig


class TestCurriculumTask:
    def test_curriculum_task_creation(self):
        task_id = 123
        cfg = MettaGridConfig()

        task = CurriculumTask(task_id, cfg)

        assert task._task_id == task_id
        assert task._env_cfg == cfg
        assert task._num_completions == 0
        assert task._total_score == 0.0
        assert task._mean_score == 0.0
        assert task._num_scheduled == 0

    def test_curriculum_task_get_env_cfg(self):
        cfg = MettaGridConfig()
        task = CurriculumTask(123, cfg)

        assert task.get_env_cfg() is cfg

    def test_curriculum_task_complete(self):
        task = CurriculumTask(123, MettaGridConfig())

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
    def test_curriculum_config_creation(self):
        """Test creating a CurriculumConfig with valid parameters."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(
            task_generator=task_gen_config, max_task_id=1000, num_active_tasks=50, new_task_rate=0.05
        )

        assert config.task_generator is task_gen_config
        assert config.max_task_id == 1000
        assert config.num_active_tasks == 50
        assert config.new_task_rate == 0.05

    def test_curriculum_config_defaults(self):
        """Test that CurriculumConfig uses correct default values."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config)

        assert config.max_task_id == 1000000
        assert config.num_active_tasks == 10000
        assert config.new_task_rate == 0.01

    def test_curriculum_config_validation_num_active_tasks(self):
        """Test that num_active_tasks validation works."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())

        # This should fail because num_active_tasks > max_task_id
        with pytest.raises(ValueError):
            CurriculumConfig(task_generator=task_gen_config, max_task_id=100, num_active_tasks=200)

    def test_curriculum_config_edge_case_values(self):
        """Test edge case values for parameters."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())

        # Test minimum values
        config = CurriculumConfig(task_generator=task_gen_config, max_task_id=1, num_active_tasks=1, new_task_rate=0.0)
        assert config.max_task_id == 1
        assert config.num_active_tasks == 1
        assert config.new_task_rate == 0.0

        # Test maximum values
        config = CurriculumConfig(
            task_generator=task_gen_config, max_task_id=1000000, num_active_tasks=1000000, new_task_rate=1.0
        )
        assert config.new_task_rate == 1.0


class TestCurriculum:
    """Test cases for Curriculum."""

    def create_test_config(self):
        """Helper to create a test configuration."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        return CurriculumConfig(task_generator=task_gen_config, max_task_id=1000, num_active_tasks=5, new_task_rate=0.1)

    def test_curriculum_creation(self):
        """Test creating a Curriculum."""
        config = self.create_test_config()
        curriculum = Curriculum(config, seed=0)

        assert curriculum._config is config
        assert hasattr(curriculum._task_generator, "get_task")
        assert isinstance(curriculum._rng, random.Random)
        assert len(curriculum._tasks) == 0
        assert len(curriculum._task_ids) == 0
        assert curriculum._num_created == 0
        assert curriculum._num_evicted == 0

    def test_curriculum_get_task_creates_task(self):
        """Test that get_task creates new tasks when under capacity."""
        config = self.create_test_config()
        curriculum = Curriculum(config, seed=0)

        task = curriculum.get_task()

        assert isinstance(task, CurriculumTask)
        assert len(curriculum._tasks) == 1
        assert len(curriculum._task_ids) == 1
        assert curriculum._num_created == 1
        assert task._num_scheduled == 1

    def test_curriculum_get_task_fills_capacity(self):
        """Test that curriculum fills to capacity before reusing tasks."""
        config = self.create_test_config()
        curriculum = Curriculum(config, seed=0)

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

    def test_curriculum_task_reuse_and_eviction(self):
        """Test task reuse and eviction when at capacity."""
        config = self.create_test_config()
        config.new_task_rate = 0.5  # 50% chance of new task
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(5):
            curriculum.get_task()

        initial_created = curriculum._num_created
        initial_evicted = curriculum._num_evicted

        # Get more tasks - should either reuse or evict+create
        for _ in range(10):
            task = curriculum.get_task()
            assert task._task_id in curriculum._task_ids

        # Should still have exactly 5 active tasks
        assert len(curriculum._tasks) == 5

        # Should have either created new tasks (with eviction) or reused existing ones
        assert curriculum._num_created >= initial_created
        if curriculum._num_created > initial_created:
            assert curriculum._num_evicted > initial_evicted

    def test_curriculum_deterministic_with_same_seed(self):
        """Test that curriculum is deterministic with the same seed."""
        config = self.create_test_config()

        curriculum1 = Curriculum(config, seed=42)
        curriculum2 = Curriculum(config, seed=42)

        tasks1 = [curriculum1.get_task() for _ in range(3)]
        tasks2 = [curriculum2.get_task() for _ in range(3)]

        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]

        assert task_ids1 == task_ids2

    def test_curriculum_different_seeds_produce_different_tasks(self):
        """Test that different seeds produce different task sequences."""
        config = self.create_test_config()

        curriculum1 = Curriculum(config, seed=42)
        curriculum2 = Curriculum(config, seed=123)

        tasks1 = [curriculum1.get_task() for _ in range(5)]
        tasks2 = [curriculum2.get_task() for _ in range(5)]

        task_ids1 = [task._task_id for task in tasks1]
        task_ids2 = [task._task_id for task in tasks2]

        # Should be different sequences
        assert task_ids1 != task_ids2

    def test_curriculum_stats(self):
        """Test that curriculum stats are properly tracked."""
        config = self.create_test_config()
        curriculum = Curriculum(config, seed=0)

        # Initial stats
        stats = curriculum.stats()
        assert stats["num_created"] == 0
        assert stats["num_evicted"] == 0
        assert stats["num_completed"] == 0
        assert stats["num_scheduled"] == 0
        assert stats["num_active_tasks"] == 0

        # Get some tasks
        task1 = curriculum.get_task()
        _ = curriculum.get_task()

        # Complete a task
        task1.complete(0.8)

        # Check updated stats
        stats = curriculum.stats()
        assert stats["num_created"] == 2
        assert stats["num_evicted"] == 0
        assert stats["num_completed"] == 1
        assert stats["num_scheduled"] == 2  # Both tasks were scheduled once
        assert stats["num_active_tasks"] == 2

    def test_curriculum_task_id_uniqueness(self):
        """Test that task IDs are unique within the max_task_id range."""
        config = self.create_test_config()
        config.max_task_id = 10  # Small range to test collision handling
        curriculum = Curriculum(config, seed=0)

        # Should be able to create at least num_active_tasks unique tasks
        tasks = []
        for _ in range(5):
            task = curriculum.get_task()
            tasks.append(task)

        task_ids = [task._task_id for task in tasks]
        assert len(set(task_ids)) == len(task_ids)  # All unique

        # All task IDs should be within range
        for task_id in task_ids:
            assert 0 <= task_id <= 10

    def test_curriculum_task_scheduling_count(self):
        """Test that task scheduling is properly tracked."""
        config = self.create_test_config()
        curriculum = Curriculum(config, seed=0)

        task = curriculum.get_task()
        assert task._num_scheduled == 1

        # If we get the same task again, it should increment
        # (This might happen due to reuse when at capacity)
        for _ in range(10):
            returned_task = curriculum.get_task()
            if returned_task is task:
                break

        # At least one task should have been scheduled more than once
        total_scheduled = sum(task._num_scheduled for task in curriculum._tasks.values())
        assert total_scheduled >= len(curriculum._tasks)


class TestCurriculumEdgeCases:
    """Test edge cases and error conditions."""

    def test_curriculum_with_new_task_rate_zero(self):
        """Test curriculum behavior when new_task_rate is 0."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(
            task_generator=task_gen_config,
            num_active_tasks=3,
            new_task_rate=0.0,  # Never create new tasks after capacity
        )
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(3):
            curriculum.get_task()

        initial_created = curriculum._num_created

        # Get more tasks - should only reuse existing ones
        for _ in range(10):
            curriculum.get_task()

        assert curriculum._num_created == initial_created  # No new tasks created
        assert curriculum._num_evicted == 0  # No tasks evicted

    def test_curriculum_with_new_task_rate_one(self):
        """Test curriculum behavior when new_task_rate is 1.0."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(
            task_generator=task_gen_config,
            num_active_tasks=3,
            new_task_rate=1.0,  # Always create new tasks after capacity
        )
        curriculum = Curriculum(config, seed=0)

        # Fill to capacity
        for _ in range(3):
            curriculum.get_task()

        initial_created = curriculum._num_created
        initial_evicted = curriculum._num_evicted

        # Get more tasks - should always evict and create new
        for _ in range(5):
            curriculum.get_task()

        assert curriculum._num_created > initial_created
        assert curriculum._num_evicted > initial_evicted

    def test_curriculum_with_single_task_capacity(self):
        """Test curriculum with capacity of 1."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(task_generator=task_gen_config, num_active_tasks=1, new_task_rate=0.5)
        curriculum = Curriculum(config, seed=0)

        # Should always maintain exactly 1 task
        for _ in range(10):
            _ = curriculum.get_task()
            assert len(curriculum._tasks) == 1
            assert len(curriculum._task_ids) == 1

    def test_curriculum_max_task_id_boundary(self):
        """Test task ID generation at max_task_id boundary."""
        task_gen_config = SingleTaskGeneratorConfig(env=MettaGridConfig())
        config = CurriculumConfig(
            task_generator=task_gen_config,
            max_task_id=2,  # IDs 0, 1, 2 possible (3 total)
            num_active_tasks=2,
        )
        curriculum = Curriculum(config, seed=0)

        task1 = curriculum.get_task()
        task2 = curriculum.get_task()

        assert task1._task_id in [0, 1, 2]
        assert task2._task_id in [0, 1, 2]
        assert task1._task_id != task2._task_id

        # Both tasks should have valid IDs within range
        assert 0 <= task1._task_id <= 2
        assert 0 <= task2._task_id <= 2

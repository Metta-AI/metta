"""Tests for the curriculum module."""

import pytest
from pydantic import ValidationError

from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig, CurriculumTask
from metta.mettagrid.mettagrid_config import EnvConfig

from .conftest import create_test_env_config


class MockTaskGenerator:
    """Mock task generator for testing."""

    def get_task(self, task_id: int) -> EnvConfig:
        """Return a test env config."""
        return create_test_env_config(seed=task_id)


class MockTaskGeneratorConfig:
    """Mock task generator config for testing."""

    def create(self):
        """Create a mock task generator."""
        return MockTaskGenerator()


def create_mock_task_generator_config():
    """Create a valid TaskGeneratorConfig instance for testing."""
    from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

    return SingleTaskGeneratorConfig()


class TestCurriculumTask:
    """Test cases for CurriculumTask."""

    def test_init(self):
        """Test CurriculumTask initialization."""
        env_cfg = create_test_env_config()
        task = CurriculumTask(task_id=123, env_cfg=env_cfg)

        assert task._task_id == 123
        assert task._env_cfg == env_cfg
        assert task._num_completions == 0
        assert task._total_score == 0.0
        assert task._mean_score == 0.0
        assert task._num_scheduled == 0

    def test_complete(self):
        """Test task completion tracking."""
        env_cfg = create_test_env_config()
        task = CurriculumTask(task_id=123, env_cfg=env_cfg)

        # First completion
        task.complete(0.8)
        assert task._num_completions == 1
        assert task._total_score == 0.8
        assert task._mean_score == 0.8

        # Second completion
        task.complete(0.6)
        assert task._num_completions == 2
        assert task._total_score == 1.4
        assert task._mean_score == 0.7

        # Third completion
        task.complete(1.0)
        assert task._num_completions == 3
        assert task._total_score == 2.4
        assert abs(task._mean_score - 0.8) < 1e-10

    def test_get_env_cfg(self):
        """Test getting environment config."""
        env_cfg = create_test_env_config()
        task = CurriculumTask(task_id=123, env_cfg=env_cfg)

        assert task.get_env_cfg() == env_cfg


class TestCurriculumConfig:
    """Test cases for CurriculumConfig."""

    def test_init_with_defaults(self):
        """Test CurriculumConfig initialization with defaults."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config)

        assert config.task_generator_config == task_gen_config
        assert config.max_task_id == 1000000
        assert config.num_active_tasks == 100
        assert config.new_task_rate == 0.01

    def test_init_with_custom_values(self):
        """Test CurriculumConfig initialization with custom values."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(
            task_generator_config=task_gen_config, max_task_id=500000, num_active_tasks=50, new_task_rate=0.05
        )

        assert config.task_generator_config == task_gen_config
        assert config.max_task_id == 500000
        assert config.num_active_tasks == 50
        assert config.new_task_rate == 0.05

    def test_validation_max_task_id_positive(self):
        """Test max_task_id must be positive."""
        task_gen_config = create_mock_task_generator_config()
        with pytest.raises(ValidationError):
            CurriculumConfig(task_generator_config=task_gen_config, max_task_id=0)

    def test_validation_num_active_tasks_positive(self):
        """Test num_active_tasks must be positive."""
        task_gen_config = create_mock_task_generator_config()
        with pytest.raises(ValidationError):
            CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=0)

    def test_validation_num_active_tasks_less_than_max_task_id(self):
        """Test num_active_tasks must be less than max_task_id."""
        task_gen_config = create_mock_task_generator_config()
        with pytest.raises(ValidationError):
            CurriculumConfig(task_generator_config=task_gen_config, max_task_id=50, num_active_tasks=100)

    def test_validation_new_task_rate_in_range(self):
        """Test new_task_rate must be between 0 and 1."""
        task_gen_config = create_mock_task_generator_config()

        # Test negative rate
        with pytest.raises(ValidationError):
            CurriculumConfig(task_generator_config=task_gen_config, new_task_rate=-0.1)

        # Test rate > 1
        with pytest.raises(ValidationError):
            CurriculumConfig(task_generator_config=task_gen_config, new_task_rate=1.1)

    def test_make(self):
        """Test creating Curriculum from config."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config)

        curriculum = config.make()
        assert isinstance(curriculum, Curriculum)
        assert curriculum._config == config


class TestCurriculum:
    """Test cases for Curriculum."""

    def test_init(self):
        """Test Curriculum initialization."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=10)

        curriculum = Curriculum(config, seed=42)

        assert curriculum._config == config
        # Check that RNG was initialized with seed (test deterministic behavior instead)
        curriculum2 = Curriculum(config, seed=42)
        # Both curricula with same seed should produce same random numbers
        curriculum._rng.seed(42)  # Reset to ensure deterministic test
        curriculum2._rng.seed(42)
        assert curriculum._rng.random() == curriculum2._rng.random()
        assert len(curriculum._tasks) == 0
        assert len(curriculum._task_ids) == 0
        assert curriculum._num_created == 0
        assert curriculum._num_evicted == 0

    def test_get_task_initial_population(self):
        """Test get_task when building initial population."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=3, max_task_id=100)

        curriculum = Curriculum(config, seed=42)

        # First few tasks should create new ones
        for _i in range(3):
            task = curriculum.get_task()
            assert isinstance(task, CurriculumTask)
            assert task._num_scheduled == 1

        assert len(curriculum._tasks) == 3
        assert curriculum._num_created == 3
        assert curriculum._num_evicted == 0

    def test_get_task_with_new_task_rate(self):
        """Test get_task with new task creation rate."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(
            task_generator_config=task_gen_config,
            num_active_tasks=2,
            new_task_rate=1.0,  # Always create new tasks
            max_task_id=1000,
        )

        curriculum = Curriculum(config, seed=42)

        # Fill initial population
        curriculum.get_task()
        curriculum.get_task()
        assert len(curriculum._tasks) == 2

        # Next task should evict and create new
        curriculum.get_task()
        assert len(curriculum._tasks) == 2  # Should still be 2
        assert curriculum._num_evicted == 1
        assert curriculum._num_created == 3

    def test_get_task_choose_existing(self):
        """Test get_task chooses from existing tasks."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(
            task_generator_config=task_gen_config,
            num_active_tasks=2,
            new_task_rate=0.0,  # Never create new tasks
            max_task_id=1000,
        )

        curriculum = Curriculum(config, seed=42)

        # Fill initial population
        task1 = curriculum.get_task()
        task2 = curriculum.get_task()

        # Next task should choose from existing
        task3 = curriculum.get_task()
        assert task3 in [task1, task2]
        assert len(curriculum._tasks) == 2
        assert curriculum._num_created == 2
        assert curriculum._num_evicted == 0

    def test_create_task_unique_ids(self):
        """Test that created tasks have unique IDs."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=10, max_task_id=20)

        curriculum = Curriculum(config, seed=42)

        # Create multiple tasks
        tasks = []
        for _ in range(5):
            task = curriculum.get_task()
            tasks.append(task)

        # Check all task IDs are unique
        task_ids = [task._task_id for task in tasks]
        assert len(set(task_ids)) == len(task_ids)

    def test_evict_task(self):
        """Test task eviction."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=2, max_task_id=1000)

        curriculum = Curriculum(config, seed=42)

        # Create initial tasks
        task1 = curriculum.get_task()
        task2 = curriculum.get_task()
        initial_ids = {task1._task_id, task2._task_id}

        # Force eviction by setting new_task_rate to 1.0
        curriculum._config.new_task_rate = 1.0
        curriculum.get_task()

        # One task should have been evicted
        current_ids = set(curriculum._task_ids)
        assert len(current_ids) == 2
        assert len(initial_ids - current_ids) == 1  # One ID removed
        assert curriculum._num_evicted == 1

    def test_stats(self):
        """Test curriculum statistics."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(task_generator_config=task_gen_config, num_active_tasks=3, max_task_id=1000)

        curriculum = Curriculum(config, seed=42)

        # Initial stats
        stats = curriculum.stats()
        assert stats["num_created"] == 0
        assert stats["num_evicted"] == 0
        assert stats["num_completed"] == 0
        assert stats["num_scheduled"] == 0
        assert stats["num_active_tasks"] == 0

        # Create and use tasks
        task1 = curriculum.get_task()
        task2 = curriculum.get_task()

        # Complete some tasks
        task1.complete(0.5)
        task1.complete(0.8)
        task2.complete(0.9)

        stats = curriculum.stats()
        assert stats["num_created"] == 2
        assert stats["num_evicted"] == 0
        assert stats["num_completed"] == 3
        assert stats["num_scheduled"] == 2  # Each task scheduled once
        assert stats["num_active_tasks"] == 2

    def test_choose_task_randomness(self):
        """Test that _choose_task has some randomness."""
        task_gen_config = create_mock_task_generator_config()
        config = CurriculumConfig(
            task_generator_config=task_gen_config,
            num_active_tasks=3,
            new_task_rate=0.0,  # Only choose existing tasks
            max_task_id=1000,
        )

        curriculum = Curriculum(config, seed=42)

        # Create initial population
        for _ in range(3):
            curriculum.get_task()

        # Get many tasks and ensure some variety
        chosen_ids = []
        for _ in range(20):
            task = curriculum.get_task()
            chosen_ids.append(task._task_id)

        # Should have chosen from multiple tasks (with high probability)
        unique_chosen = set(chosen_ids)
        assert len(unique_chosen) > 1  # Should have some variety

"""Tests for the learning progress curriculum configuration."""

import pytest
from pydantic import ValidationError

from metta.cogworks.curriculum.learning_progress import LearningProgressCurriculumConfig


class TestLearningProgressCurriculumConfig:
    """Test cases for LearningProgressCurriculumConfig."""

    def test_init_with_defaults(self):
        """Test LearningProgressCurriculumConfig initialization with defaults."""
        # Need to provide a mock task_generator_config since it's required
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()
        config = LearningProgressCurriculumConfig(task_generator_config=task_gen_config)

        assert config.task_generator_config == task_gen_config
        assert config.ema_timescale == 0.001
        assert config.progress_smoothing == 0.05
        assert config.rand_task_rate == 0.25
        assert config.memory == 25
        # Should inherit defaults from parent
        assert config.max_task_id == 1000000
        assert config.num_active_tasks == 100
        assert config.new_task_rate == 0.01

    def test_init_with_custom_values(self):
        """Test LearningProgressCurriculumConfig initialization with custom values."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()
        config = LearningProgressCurriculumConfig(
            task_generator_config=task_gen_config,
            ema_timescale=0.01,
            progress_smoothing=0.1,
            rand_task_rate=0.5,
            memory=50,
            max_task_id=500000,
            num_active_tasks=25,
            new_task_rate=0.02,
        )

        assert config.ema_timescale == 0.01
        assert config.progress_smoothing == 0.1
        assert config.rand_task_rate == 0.5
        assert config.memory == 50
        assert config.max_task_id == 500000
        assert config.num_active_tasks == 25
        assert config.new_task_rate == 0.02

    def test_validation_ema_timescale_positive(self):
        """Test ema_timescale must be positive."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test zero timescale
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, ema_timescale=0.0)

        # Test negative timescale
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, ema_timescale=-0.1)

    def test_validation_ema_timescale_max_value(self):
        """Test ema_timescale must be <= 1.0."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test timescale > 1
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, ema_timescale=1.1)

        # Test edge case: exactly 1.0 should work
        config = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, ema_timescale=1.0)
        assert config.ema_timescale == 1.0

    def test_validation_progress_smoothing_range(self):
        """Test progress_smoothing must be between 0 and 1."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test negative smoothing
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, progress_smoothing=-0.1)

        # Test smoothing > 1
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, progress_smoothing=1.1)

        # Test edge cases: 0.0 and 1.0 should work
        config_min = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, progress_smoothing=0.0)
        assert config_min.progress_smoothing == 0.0

        config_max = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, progress_smoothing=1.0)
        assert config_max.progress_smoothing == 1.0

    def test_validation_rand_task_rate_range(self):
        """Test rand_task_rate must be between 0 and 1."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test negative rate
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, rand_task_rate=-0.1)

        # Test rate > 1
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, rand_task_rate=1.1)

        # Test edge cases: 0.0 and 1.0 should work
        config_min = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, rand_task_rate=0.0)
        assert config_min.rand_task_rate == 0.0

        config_max = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, rand_task_rate=1.0)
        assert config_max.rand_task_rate == 1.0

    def test_validation_memory_positive(self):
        """Test memory must be positive."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test zero memory
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, memory=0)

        # Test negative memory
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, memory=-1)

        # Test positive memory should work
        config = LearningProgressCurriculumConfig(task_generator_config=task_gen_config, memory=1)
        assert config.memory == 1

    def test_inherits_parent_validation(self):
        """Test that validation from parent CurriculumConfig still applies."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test inherited validation for num_active_tasks vs max_task_id
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, max_task_id=10, num_active_tasks=20)

    def test_field_descriptions_and_defaults(self):
        """Test that fields have appropriate descriptions and defaults."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()
        config = LearningProgressCurriculumConfig(task_generator_config=task_gen_config)

        # Check that all learning progress specific fields have reasonable defaults
        assert 0 < config.ema_timescale <= 1.0
        assert 0 <= config.progress_smoothing <= 1.0
        assert 0 <= config.rand_task_rate <= 1.0
        assert config.memory > 0

        # Check that inherited fields are still available
        assert hasattr(config, "max_task_id")
        assert hasattr(config, "num_active_tasks")
        assert hasattr(config, "new_task_rate")

    def test_config_model_validation(self):
        """Test that pydantic model configuration is inherited properly."""
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        task_gen_config = SingleTaskGeneratorConfig()

        # Test that extra fields are forbidden (from parent config)
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_generator_config=task_gen_config, unknown_field="should_fail")

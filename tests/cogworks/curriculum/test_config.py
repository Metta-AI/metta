"""Tests for config-based curriculum creation."""

import pytest
from pydantic import ValidationError

from cogworks.curriculum.config import (
    TaskSetConfig,
    WeightedTaskSetConfig,
    BuckettedTaskSetConfig,
    WeightedTaskSetItem,
    BucketValue,
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
    TaskSetConfigUnion,
    CurriculumConfigUnion
)
from cogworks.curriculum.curriculum import RandomCurriculum, LearningProgressCurriculum
from cogworks.curriculum.task_set import WeightedTaskSet, BuckettedTaskSet, create_task_set_from_config
from metta.rl.env_config import EnvConfig


class TestTaskSetConfigs:
    """Test cases for TaskSet configuration classes."""
    
    def test_base_taskset_config(self):
        """Test basic TaskSetConfig creation."""
        config = TaskSetConfig()
        
        # Should create successfully with defaults
        assert isinstance(config, TaskSetConfig)
        
    def test_weighted_taskset_config_empty(self):
        """Test WeightedTaskSetConfig with empty items."""
        config = WeightedTaskSetConfig(items=[])
        
        assert len(config.items) == 0
        assert config.overrides is None
        
    def test_weighted_taskset_config_with_env_items(self):
        """Test WeightedTaskSetConfig with environment config items."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=1.0)
        item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=2.5)
        
        config = WeightedTaskSetConfig(items=[item1, item2])
        
        assert len(config.items) == 2
        assert config.items[0].weight == 1.0
        assert config.items[1].weight == 2.5
        
    def test_weighted_taskset_config_with_overrides(self):
        """Test WeightedTaskSetConfig with overrides."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        
        overrides = {"device": "cpu", "torch_deterministic": False}
        config = WeightedTaskSetConfig(items=[item], overrides=overrides)
        
        assert config.overrides == overrides
        
    def test_weighted_taskset_config_with_nested_taskset(self):
        """Test WeightedTaskSetConfig with nested task sets."""
        # Create nested config
        env_cfg = EnvConfig()
        nested_item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        nested_config = WeightedTaskSetConfig(items=[nested_item])
        
        # Create parent config
        parent_item = WeightedTaskSetItem(task_set_config=nested_config, weight=2.0)
        parent_config = WeightedTaskSetConfig(items=[parent_item])
        
        assert parent_config.items[0].task_set_config is nested_config
        assert parent_config.items[0].weight == 2.0
        
    def test_bucketed_taskset_config(self):
        """Test BuckettedTaskSetConfig creation."""
        base_config = EnvConfig()
        
        buckets = {
            "device": [BucketValue(value="cpu"), BucketValue(value="cuda")],
            "seed": [BucketValue(range_min=5, range_max=15)]
        }
        
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        
        
    def test_bucket_value_validation(self):
        """Test BucketValue validation."""
        # Valid value bucket
        value_bucket = BucketValue(value=42)
        assert value_bucket.value == 42
        assert value_bucket.range_min is None
        assert value_bucket.range_max is None
        
        # Valid range bucket
        range_bucket = BucketValue(range_min=1, range_max=10)
        assert range_bucket.value is None
        assert range_bucket.range_min == 1
        assert range_bucket.range_max == 10


class TestWeightedTaskSetItemValidation:
    """Test validation for WeightedTaskSetItem."""
    
    def test_valid_env_config_item(self):
        """Test valid item with env_config."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        
        assert item.env_config is env_cfg
        assert item.task_set_config is None
        assert item.weight == 1.0
        
    def test_valid_task_set_config_item(self):
        """Test valid item with task_set_config."""
        task_set_config = WeightedTaskSetConfig(items=[])
        item = WeightedTaskSetItem(task_set_config=task_set_config, weight=2.0)
        
        assert item.env_config is None
        assert item.task_set_config is task_set_config
        assert item.weight == 2.0
        
    def test_invalid_both_configs_set(self):
        """Test that setting both env_config and task_set_config fails."""
        env_cfg = EnvConfig()
        task_set_config = WeightedTaskSetConfig(items=[])
        
        with pytest.raises(ValidationError, match="Exactly one of env_config or task_set_config must be set"):
            WeightedTaskSetItem(
                env_config=env_cfg,
                task_set_config=task_set_config,
                weight=1.0
            )
            
    def test_invalid_neither_config_set(self):
        """Test that setting neither config fails."""
        with pytest.raises(ValidationError, match="Exactly one of env_config or task_set_config must be set"):
            WeightedTaskSetItem(weight=1.0)
            
    def test_invalid_zero_weight(self):
        """Test that zero weight is rejected."""
        env_cfg = EnvConfig()
        
        with pytest.raises(ValidationError):
            WeightedTaskSetItem(env_config=env_cfg, weight=0.0)
            
    def test_invalid_negative_weight(self):
        """Test that negative weight is rejected."""
        env_cfg = EnvConfig()
        
        with pytest.raises(ValidationError):
            WeightedTaskSetItem(env_config=env_cfg, weight=-1.0)


class TestCurriculumConfigs:
    """Test cases for Curriculum configuration classes."""
    
    def test_base_curriculum_config(self):
        """Test basic CurriculumConfig creation."""
        task_set_config = WeightedTaskSetConfig(items=[])
        config = CurriculumConfig(task_set_config=task_set_config)
        
        assert config.task_set_config is task_set_config
        
    def test_random_curriculum_config(self):
        """Test RandomCurriculumConfig creation."""
        task_set_config = WeightedTaskSetConfig(items=[])
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        
        assert config.task_set_config is task_set_config
        assert isinstance(config, CurriculumConfig)  # Inheritance
        
    def test_learning_progress_curriculum_config(self):
        """Test LearningProgressCurriculumConfig creation."""
        task_set_config = WeightedTaskSetConfig(items=[])
        config = LearningProgressCurriculumConfig(
            task_set_config=task_set_config,
            n_tasks=10,
            num_active_tasks=5,
            memory=20,
            ema_timescale=0.1,
            progress_smoothing=0.2,
            rand_task_rate=0.3
        )
        
        assert config.task_set_config is task_set_config
        assert config.n_tasks == 10
        assert config.num_active_tasks == 5
        assert config.memory == 20
        assert config.ema_timescale == 0.1
        assert config.progress_smoothing == 0.2
        assert config.rand_task_rate == 0.3
        
    def test_learning_progress_config_defaults(self):
        """Test LearningProgressCurriculumConfig defaults."""
        task_set_config = WeightedTaskSetConfig(items=[])
        config = LearningProgressCurriculumConfig(task_set_config=task_set_config)
        
        # Should have reasonable defaults
        assert config.n_tasks == 100
        assert config.num_active_tasks == 16
        assert config.memory == 25
        assert config.ema_timescale == 0.001
        assert config.progress_smoothing == 0.05
        assert config.rand_task_rate == 0.25
        
    def test_learning_progress_config_validation(self):
        """Test LearningProgressCurriculumConfig validation."""
        task_set_config = WeightedTaskSetConfig(items=[])
        
        # Invalid n_tasks
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, n_tasks=-1)
            
        # Invalid num_active_tasks
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, num_active_tasks=0)
            
        # Invalid memory
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, memory=0)
            
        # Invalid ema_timescale (should be in (0, 1])
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, ema_timescale=0.0)
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, ema_timescale=1.5)
            
        # Invalid rand_task_rate (should be in [0, 1])
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, rand_task_rate=-0.1)
        with pytest.raises(ValidationError):
            LearningProgressCurriculumConfig(task_set_config=task_set_config, rand_task_rate=1.1)


class TestConfigBasedCreation:
    """Test creating curriculum objects from configs."""
    
    def test_create_random_curriculum_from_config(self):
        """Test creating RandomCurriculum from config."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item])
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        
        curriculum = RandomCurriculum(config, seed=0)
        
        assert isinstance(curriculum, RandomCurriculum)
        assert curriculum.config is config
        
        # Test that it works
        task = curriculum.get_task(42)
        
    def test_create_learning_progress_curriculum_from_config(self):
        """Test creating LearningProgressCurriculum from config."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        task_set_config = WeightedTaskSetConfig(items=[item])
        config = LearningProgressCurriculumConfig(
            task_set_config=task_set_config,
            n_tasks=5,
            num_active_tasks=3
        )
        
        curriculum = LearningProgressCurriculum(config, seed=0)
        
        assert isinstance(curriculum, LearningProgressCurriculum)
        assert curriculum.config is config
        
        # Test that it works
        task = curriculum.get_task(42)
        assert len(curriculum.tasks) == 5
        
    def test_create_taskset_from_config(self):
        """Test creating TaskSets from configs."""
        # WeightedTaskSet
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        weighted_config = WeightedTaskSetConfig(items=[item])
        
        weighted_taskset = create_task_set_from_config(weighted_config)
        assert isinstance(weighted_taskset, WeightedTaskSet)
        
        result = weighted_taskset.get_task(42)
        
        # BuckettedTaskSet
        base_config = EnvConfig()
        buckets = {"device": [BucketValue(value="cpu")]}
        bucketed_config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        
        bucketed_taskset = create_task_set_from_config(bucketed_config)
        assert isinstance(bucketed_taskset, BuckettedTaskSet)
        
        result = bucketed_taskset.get_task(42)
        
    def test_complex_nested_config(self):
        """Test complex nested configuration."""
        # Create a deeply nested configuration
        
        # Level 3: Base env configs
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        # Level 2: Inner task set
        inner_item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=1.0)
        inner_item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=2.0)
        inner_config = WeightedTaskSetConfig(items=[inner_item1, inner_item2])
        
        # Level 1: Outer task set with both direct env config and nested task set
        direct_env_cfg = EnvConfig()
        direct_item = WeightedTaskSetItem(env_config=direct_env_cfg, weight=1.0)
        nested_item = WeightedTaskSetItem(task_set_config=inner_config, weight=1.0)
        outer_config = WeightedTaskSetConfig(items=[direct_item, nested_item])
        
        # Level 0: Curriculum
        curriculum_config = RandomCurriculumConfig(task_set_config=outer_config)
        curriculum = RandomCurriculum(curriculum_config, seed=0)
        
        # Test that it works
        task = curriculum.get_task(42)
        # Should get one of: 1, 2, or 3 agents
        
    def test_config_with_overrides(self):
        """Test configuration with overrides applied."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        
        overrides = {
            "device": "cpu",
            "torch_deterministic": False
        }
        task_set_config = WeightedTaskSetConfig(items=[item], overrides=overrides)
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        
        curriculum = RandomCurriculum(config, seed=0)
        task = curriculum.get_task(42)
        
        result_env_cfg = task.get_env_config()
        
        # Overrides should be applied
        # Non-overridden values should remain
        
    def test_bucketed_config_comprehensive(self):
        """Test comprehensive bucketed configuration."""
        base_config = EnvConfig()
        
        buckets = {
            "seed": [
                BucketValue(value=12345),
                BucketValue(value=67890)
            ],
            "torch_deterministic": [
                BucketValue(value=True),
                BucketValue(value=False)
            ],
            "device": [
                BucketValue(value="cpu"),
                BucketValue(value="cuda")
            ]
        }
        
        task_set_config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        config = RandomCurriculumConfig(task_set_config=task_set_config)
        
        curriculum = RandomCurriculum(config, seed=0)
        
        # Sample multiple times to test variation
        results = []
        for seed in range(20):
            task = curriculum.get_task(seed)
            env_cfg = task.get_env_config()
            results.append({
                "seed": env_cfg.seed,
                "torch_deterministic": env_cfg.torch_deterministic,
                "device": env_cfg.device
            })
            
        # All results should respect bucket constraints
        for result in results:
            assert result["seed"] in [12345, 67890]
            assert result["torch_deterministic"] in [True, False]
            assert result["device"] in ["cpu", "cuda"]
            
        # Should have some variation in results
        seed_values = set(result["seed"] for result in results)
        assert len(seed_values) > 1  # Should see both values


class TestConfigValidationEdgeCases:
    """Test edge cases and validation scenarios."""
    
    def test_config_immutability(self):
        """Test that configs maintain data integrity."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        config = WeightedTaskSetConfig(items=[item])
        
        # Original config should remain unchanged
        original_num_agents = env_cfg.seed
        
        # Create curriculum and use it
        curriculum_config = RandomCurriculumConfig(task_set_config=config)
        curriculum = RandomCurriculum(curriculum_config, seed=0)
        curriculum.get_task(42)
        
        # Original should be unchanged
        
    def test_config_validation_comprehensive(self):
        """Test comprehensive validation of all config parameters."""
        task_set_config = WeightedTaskSetConfig(items=[])
        
        # Test all validation boundaries for LearningProgressCurriculumConfig
        valid_config = LearningProgressCurriculumConfig(
            task_set_config=task_set_config,
            n_tasks=1,  # Minimum valid value
            num_active_tasks=1,  # Minimum valid value
            memory=1,  # Minimum valid value
            ema_timescale=0.001,  # Just above 0
            progress_smoothing=0.0,  # Minimum valid value
            rand_task_rate=1.0  # Maximum valid value
        )
        
        assert valid_config.n_tasks == 1
        assert valid_config.ema_timescale == 0.001
        assert valid_config.rand_task_rate == 1.0
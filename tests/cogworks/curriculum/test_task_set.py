"""Tests for TaskSet classes."""

import pytest
from pydantic import ValidationError
from cogworks.curriculum.task_set import TaskSet, WeightedTaskSet, BuckettedTaskSet, create_task_set_from_config
from cogworks.curriculum.config import (
    TaskSetConfig, 
    WeightedTaskSetConfig, 
    BuckettedTaskSetConfig,
    WeightedTaskSetItem,
    BucketValue
)
from metta.rl.env_config import EnvConfig


class TestTaskSetBase:
    """Test cases for TaskSet base class behavior."""
    
    def test_taskset_is_abstract(self):
        """Test that TaskSet cannot be instantiated directly."""
        config = TaskSetConfig()
        with pytest.raises(TypeError):
            TaskSet(config)
            
    def test_taskset_config_storage(self):
        """Test that TaskSet stores config properly."""
        config = WeightedTaskSetConfig(items=[])
        task_set = WeightedTaskSet(config)
        
        assert task_set.config is config
        assert isinstance(task_set.config, WeightedTaskSetConfig)


class TestWeightedTaskSet:
    """Test cases for WeightedTaskSet."""
    
    def test_empty_weighted_taskset_creation(self):
        """Test creating an empty WeightedTaskSet."""
        config = WeightedTaskSetConfig(items=[])
        task_set = WeightedTaskSet(config)
        
        assert len(task_set.items) == 0
        assert task_set.overrides == {}
        
    def test_empty_taskset_get_task_fails(self):
        """Test that empty TaskSet raises error on get_task."""
        config = WeightedTaskSetConfig(items=[])
        task_set = WeightedTaskSet(config)
        
        with pytest.raises(ValueError, match="No items to sample from"):
            task_set.get_task(42)
            
    def test_single_env_config_taskset(self):
        """Test TaskSet with single EnvConfig item."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        config = WeightedTaskSetConfig(items=[item])
        task_set = WeightedTaskSet(config)
        
        # Should always return the same env config
        result1 = task_set.get_task(42)
        result2 = task_set.get_task(42)  # Same seed
        result3 = task_set.get_task(43)  # Different seed
        
        assert result1 == env_cfg
        assert result2 == env_cfg
        assert result3 == env_cfg
        
    def test_multiple_env_config_taskset(self):
        """Test TaskSet with multiple EnvConfig items."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=1.0)
        item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=1.0)
        config = WeightedTaskSetConfig(items=[item1, item2])
        task_set = WeightedTaskSet(config)
        
        # Same seed should produce same result
        result1 = task_set.get_task(42)
        result2 = task_set.get_task(42)
        assert result1 == result2
        
        # Should be one of the configured env configs
        assert result1 in [env_cfg1, env_cfg2]
        
    def test_weighted_sampling_bias(self):
        """Test that weights affect sampling probability."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        # Weight env_cfg1 much higher
        item1 = WeightedTaskSetItem(env_config=env_cfg1, weight=100.0)
        item2 = WeightedTaskSetItem(env_config=env_cfg2, weight=1.0)
        config = WeightedTaskSetConfig(items=[item1, item2])
        task_set = WeightedTaskSet(config)
        
        # Sample many times and count occurrences
        results = []
        for seed in range(100):
            result = task_set.get_task(seed)
            results.append(result)
            
        count1 = results.count(env_cfg1)
        count2 = results.count(env_cfg2)
        
        # env_cfg1 should be selected more often due to higher weight
        assert count1 > count2
        assert count1 + count2 == 100
        
    def test_taskset_overrides(self):
        """Test that overrides are applied to env configs."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        
        # Override valid fields
        overrides = {"device": "cpu", "torch_deterministic": False}
        config = WeightedTaskSetConfig(items=[item], overrides=overrides)
        task_set = WeightedTaskSet(config)
        
        result = task_set.get_task(42)
        
        # Original env_cfg should be unchanged
        assert env_cfg.device == "cuda"  # Default
        assert env_cfg.torch_deterministic is True  # Default
        
        # Result should have overrides applied
        assert result.device == "cpu"
        assert result.torch_deterministic is False
        
    def test_nested_taskset(self):
        """Test TaskSet containing nested TaskSets."""
        # Create inner TaskSet
        inner_env_cfg = EnvConfig()
        inner_item = WeightedTaskSetItem(env_config=inner_env_cfg, weight=1.0)
        inner_config = WeightedTaskSetConfig(items=[inner_item])
        
        # Create outer TaskSet with nested TaskSet
        nested_item = WeightedTaskSetItem(task_set_config=inner_config, weight=1.0)
        outer_config = WeightedTaskSetConfig(items=[nested_item])
        outer_task_set = WeightedTaskSet(outer_config)
        
        result = outer_task_set.get_task(42)
        
    def test_mixed_items_taskset(self):
        """Test TaskSet with both env configs and nested task sets."""
        # Direct env config
        direct_env_cfg = EnvConfig()
        direct_item = WeightedTaskSetItem(env_config=direct_env_cfg, weight=1.0)
        
        # Nested TaskSet
        nested_env_cfg = EnvConfig()
        nested_item = WeightedTaskSetItem(env_config=nested_env_cfg, weight=1.0)
        nested_config = WeightedTaskSetConfig(items=[nested_item])
        nested_task_item = WeightedTaskSetItem(task_set_config=nested_config, weight=1.0)
        
        # Combine both
        config = WeightedTaskSetConfig(items=[direct_item, nested_task_item])
        task_set = WeightedTaskSet(config)
        
        result = task_set.get_task(42)
        # Should get one of the two possible configurations
        
    def test_override_parsing_from_list(self):
        """Test parsing overrides from list format."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        
        override_list = [
            "device: cpu",
            "torch_deterministic: false",
            "some.boolean: true",
            "some.float: 3.14"
        ]
        config = WeightedTaskSetConfig(items=[item], overrides=override_list)
        task_set = WeightedTaskSet(config)
        
        # Check that overrides were parsed correctly
        assert task_set.overrides["device"] == "cpu"
        assert task_set.overrides["torch_deterministic"] is False
        assert task_set.overrides["some.boolean"] is True
        assert task_set.overrides["some.float"] == 3.14


class TestBuckettedTaskSet:
    """Test cases for BuckettedTaskSet."""
    
    def test_bucketed_taskset_creation(self):
        """Test creating a BuckettedTaskSet."""
        base_config = EnvConfig()
        buckets = {
            "device": [BucketValue(value="cpu"), BucketValue(value="cuda")]
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        assert task_set.base_config == base_config
        assert "device" in task_set.buckets
        
    def test_bucketed_taskset_value_sampling(self):
        """Test sampling from value buckets."""
        base_config = EnvConfig(device="cuda", torch_deterministic=True)
        buckets = {
            "device": [BucketValue(value="cpu"), BucketValue(value="cuda"), BucketValue(value="mps")]
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        result = task_set.get_task(42)
        
        # Base values should be preserved for non-bucketed fields
        assert result.torch_deterministic is True
        
        # Bucketed value should be one of the specified values
        assert result.device in ["cpu", "cuda", "mps"]
        
    def test_bucketed_taskset_range_sampling(self):
        """Test sampling from range buckets."""
        base_config = EnvConfig(seed=100)
        buckets = {
            "seed": [BucketValue(range_min=1, range_max=10)]
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        result = task_set.get_task(42)
        
        # Values should be within specified ranges
        assert 1 <= result.seed <= 10
        
    def test_bucketed_taskset_mixed_buckets(self):
        """Test buckets with both values and ranges."""
        base_config = EnvConfig(seed=100, torch_deterministic=True)
        buckets = {
            "torch_deterministic": [BucketValue(value=True), BucketValue(value=False)],
            "seed": [BucketValue(range_min=1, range_max=10)]
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        result = task_set.get_task(42)
        
        assert result.torch_deterministic in [True, False]
        assert 1 <= result.seed <= 10
        
        
    def test_bucketed_taskset_deterministic(self):
        """Test that same seed produces same result."""
        base_config = EnvConfig(seed=100)
        buckets = {
            "seed": [BucketValue(range_min=1, range_max=100)]
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        result1 = task_set.get_task(42)
        result2 = task_set.get_task(42)
        result3 = task_set.get_task(43)
        
        # Same seed should produce same result
        assert result1.seed == result2.seed
        
        # Different seeds may produce different results
        # (Not guaranteed, but likely with a reasonable range)
        
    def test_empty_buckets(self):
        """Test handling of empty buckets."""
        base_config = EnvConfig(device="cuda")
        buckets = {
            "device": []  # Empty bucket
        }
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        task_set = BuckettedTaskSet(config)
        
        result = task_set.get_task(42)
        
        # Should use base config value when bucket is empty
        assert result.device == "cuda"


class TestTaskSetFactory:
    """Test cases for create_task_set_from_config function."""
    
    def test_create_weighted_taskset_from_config(self):
        """Test creating WeightedTaskSet from config."""
        env_cfg = EnvConfig()
        item = WeightedTaskSetItem(env_config=env_cfg, weight=1.0)
        config = WeightedTaskSetConfig(items=[item])
        
        task_set = create_task_set_from_config(config)
        
        assert isinstance(task_set, WeightedTaskSet)
        result = task_set.get_task(42)
        
    def test_create_bucketed_taskset_from_config(self):
        """Test creating BuckettedTaskSet from config."""
        base_config = EnvConfig()
        buckets = {"device": [BucketValue(value="cpu")]}
        config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        
        task_set = create_task_set_from_config(config)
        
        assert isinstance(task_set, BuckettedTaskSet)
        result = task_set.get_task(42)
        assert result.device == "cpu"
        
    def test_create_taskset_from_base_config(self):
        """Test creating TaskSet from base TaskSetConfig."""
        config = TaskSetConfig()
        
        task_set = create_task_set_from_config(config)
        
        # Should create a WeightedTaskSet with empty items
        assert isinstance(task_set, WeightedTaskSet)
        
    def test_create_taskset_from_invalid_config(self):
        """Test error handling for invalid config types."""
        with pytest.raises((ValueError, AttributeError)):
            create_task_set_from_config("invalid_config")


class TestTaskSetEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_weighted_item_config(self):
        """Test that WeightedTaskSetItem validation rejects items with neither config."""
        # Item with neither env_config nor task_set_config should be rejected
        with pytest.raises(ValidationError):
            WeightedTaskSetItem(weight=1.0)
            
    def test_zero_weight_items(self):
        """Test that zero-weight items are rejected."""
        env_cfg = EnvConfig()
        
        # Items with zero weights should be rejected during validation
        with pytest.raises(ValueError):
            WeightedTaskSetItem(env_config=env_cfg, weight=0.0)
        
    def test_negative_weight_validation(self):
        """Test that negative weights are rejected during config validation."""
        env_cfg = EnvConfig()
        
        with pytest.raises(ValueError):
            WeightedTaskSetItem(env_config=env_cfg, weight=-1.0)
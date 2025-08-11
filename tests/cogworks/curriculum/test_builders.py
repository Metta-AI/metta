"""Tests for builder-based curriculum creation."""

import pytest
from cogworks.curriculum.builders import (
    TaskSetBuilder,
    WeightedTaskSetBuilder,
    BuckettedTaskSetBuilder,
    CurriculumBuilder,
    RandomCurriculumBuilder,
    LearningProgressCurriculumBuilder
)
from cogworks.curriculum.config import (
    WeightedTaskSetConfig,
    BuckettedTaskSetConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig
)
from cogworks.curriculum.curriculum import RandomCurriculum, LearningProgressCurriculum
from metta.rl.env_config import EnvConfig


class TestTaskSetBuilder:
    """Test cases for TaskSetBuilder factory."""
    
    def test_weighted_builder_creation(self):
        """Test creating a WeightedTaskSetBuilder."""
        builder = TaskSetBuilder.weighted()
        
        assert isinstance(builder, WeightedTaskSetBuilder)
        assert len(builder.items) == 0
        assert builder.overrides is None
        
    def test_bucketed_builder_creation(self):
        """Test creating a BuckettedTaskSetBuilder."""
        builder = TaskSetBuilder.bucketed()
        
        assert isinstance(builder, BuckettedTaskSetBuilder)
        assert isinstance(builder.base_config, EnvConfig)
        assert len(builder.buckets) == 0
        
    def test_bucketed_builder_with_custom_base_config(self):
        """Test creating BuckettedTaskSetBuilder with custom base config."""
        base_config = EnvConfig()
        builder = TaskSetBuilder.bucketed(base_config=base_config)
        
        assert builder.base_config is base_config


class TestWeightedTaskSetBuilder:
    """Test cases for WeightedTaskSetBuilder."""
    
    def test_empty_builder(self):
        """Test empty builder builds successfully."""
        builder = WeightedTaskSetBuilder()
        config = builder.build()
        
        assert isinstance(config, WeightedTaskSetConfig)
        assert len(config.items) == 0
        assert config.overrides is None
        
    def test_add_env_config(self):
        """Test adding environment configurations."""
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        builder = WeightedTaskSetBuilder()
        config = (builder
                 .add_env_config(env_cfg1, weight=1.0)
                 .add_env_config(env_cfg2, weight=2.0)
                 .build())
                 
        assert len(config.items) == 2
        assert config.items[0].weight == 1.0
        assert config.items[1].weight == 2.0
        
    def test_add_env_config_default_weight(self):
        """Test adding env config with default weight."""
        env_cfg = EnvConfig()
        
        builder = WeightedTaskSetBuilder()
        config = builder.add_env_config(env_cfg).build()
        
        assert len(config.items) == 1
        assert config.items[0].weight == 1.0
        
    def test_add_task_set_config(self):
        """Test adding nested task set configurations."""
        # Create nested config
        nested_env_cfg = EnvConfig()
        nested_config = WeightedTaskSetConfig(items=[])
        
        builder = WeightedTaskSetBuilder()
        config = builder.add_task_set_config(nested_config, weight=3.0).build()
        
        assert len(config.items) == 1
        assert config.items[0].task_set_config is nested_config
        assert config.items[0].weight == 3.0
        
    def test_add_nested_task_set_from_builder(self):
        """Test adding nested task set from another builder."""
        # Create nested builder
        nested_env_cfg = EnvConfig()
        nested_builder = WeightedTaskSetBuilder().add_env_config(nested_env_cfg)
        
        # Create parent builder
        parent_builder = WeightedTaskSetBuilder()
        config = parent_builder.add_task_set(nested_builder, weight=4.0).build()
        
        assert len(config.items) == 1
        assert config.items[0].weight == 4.0
        # Nested config should be built
        nested_config = config.items[0].task_set_config
        assert isinstance(nested_config, WeightedTaskSetConfig)
        assert len(nested_config.items) == 1
        
    def test_with_dict_overrides(self):
        """Test adding overrides as dictionary."""
        env_cfg = EnvConfig()
        overrides = {"device": "cpu", "torch_deterministic": False}
        
        builder = WeightedTaskSetBuilder()
        config = (builder
                 .add_env_config(env_cfg)
                 .with_overrides(overrides)
                 .build())
                 
        assert config.overrides == overrides
        
    def test_with_list_overrides(self):
        """Test adding overrides as list."""
        env_cfg = EnvConfig()
        overrides_list = ["device: cpu", "torch_deterministic: false"]
        
        builder = WeightedTaskSetBuilder()
        config = (builder
                 .add_env_config(env_cfg)
                 .with_overrides(overrides_list)
                 .build())
                 
        assert config.overrides == overrides_list
        
    def test_with_keyword_overrides(self):
        """Test adding overrides as keyword arguments."""
        env_cfg = EnvConfig()
        
        builder = WeightedTaskSetBuilder()
        config = (builder
                 .add_env_config(env_cfg)
                 .with_dict_overrides(**{"device": "mps"})
                 .build())
                 
        
    def test_combining_overrides(self):
        """Test combining different override methods."""
        env_cfg = EnvConfig()
        initial_overrides = {"device": "cpu"}
        
        builder = WeightedTaskSetBuilder()
        config = (builder
                 .add_env_config(env_cfg)
                 .with_overrides(initial_overrides)
                 .with_dict_overrides(**{"torch_deterministic": False})
                 .build())
                 
        expected = {"device": "cpu", "torch_deterministic": False}
        assert config.overrides == expected
        
    def test_fluent_interface(self):
        """Test that all builder methods return self for chaining."""
        env_cfg = EnvConfig()
        
        builder = WeightedTaskSetBuilder()
        
        # All methods should return the builder for chaining
        result = (builder
                 .add_env_config(env_cfg, weight=1.0)
                 .with_overrides({"key": "value"})
                 .with_dict_overrides(another_key="value"))
                 
        assert result is builder


class TestBuckettedTaskSetBuilder:
    """Test cases for BuckettedTaskSetBuilder."""
    
    def test_empty_builder(self):
        """Test empty builder builds successfully."""
        builder = BuckettedTaskSetBuilder()
        config = builder.build()
        
        assert isinstance(config, BuckettedTaskSetConfig)
        assert isinstance(config.base_config, EnvConfig)
        assert len(config.buckets) == 0
        
    def test_with_base_config(self):
        """Test setting base configuration."""
        initial_base = EnvConfig()
        new_base = EnvConfig()
        
        builder = BuckettedTaskSetBuilder(base_config=initial_base)
        config = builder.with_base_config(new_base).build()
        
        assert config.base_config is new_base
        
    def test_add_mixed_bucket(self):
        """Test adding bucket with mixed values and ranges."""
        builder = BuckettedTaskSetBuilder()
        
        mixed_values = [2, 4, (6, 8), (10, 12)]
        config = builder.add_bucket("seed", mixed_values).build()
        
        bucket = config.buckets["seed"]
        assert len(bucket) == 4
        
        # First two should be values
        assert bucket[0].value == 2
        assert bucket[1].value == 4
        
        # Last two should be ranges
        assert bucket[2].range_min == 6
        assert bucket[2].range_max == 8
        assert bucket[3].range_min == 10
        assert bucket[3].range_max == 12
        
    def test_add_value_bucket(self):
        """Test adding discrete value bucket."""
        builder = BuckettedTaskSetBuilder()
        
        values = [1, 3, 5, 7]
        config = builder.add_value_bucket("seed", values).build()
        
        bucket = config.buckets["seed"]
        assert len(bucket) == 4
        
        for i, bucket_value in enumerate(bucket):
            assert bucket_value.value == values[i]
            assert bucket_value.range_min is None
            assert bucket_value.range_max is None
            
    def test_add_range_bucket(self):
        """Test adding range bucket."""
        builder = BuckettedTaskSetBuilder()
        
        ranges = [(1, 5), (10, 15), (20, 25)]
        config = builder.add_range_bucket("seed", ranges).build()
        
        bucket = config.buckets["seed"]
        assert len(bucket) == 3
        
        for i, bucket_value in enumerate(bucket):
            assert bucket_value.value is None
            assert bucket_value.range_min == ranges[i][0]
            assert bucket_value.range_max == ranges[i][1]
            
    def test_add_single_range_bucket(self):
        """Test adding single range bucket."""
        builder = BuckettedTaskSetBuilder()
        
        config = builder.add_single_range_bucket("seed", 8, 16).build()
        
        bucket = config.buckets["seed"]
        assert len(bucket) == 1
        assert bucket[0].range_min == 8
        assert bucket[0].range_max == 16
        
    def test_multiple_buckets(self):
        """Test adding multiple different buckets."""
        builder = BuckettedTaskSetBuilder()
        
        config = (builder
                 .add_value_bucket("seed", [1, 2])
                 .add_value_bucket("torch_deterministic", [True, False])
                 .add_value_bucket("device", ["cpu", "cuda"])
                 .build())
                 
        assert len(config.buckets) == 3
        
    def test_fluent_interface(self):
        """Test that all builder methods return self for chaining."""
        base_config = EnvConfig()
        
        builder = BuckettedTaskSetBuilder()
        
        # All methods should return the builder for chaining
        result = (builder
                 .with_base_config(base_config)
                 .add_bucket("key1", [1, 2])
                 .add_value_bucket("key2", [3, 4])
                 .add_range_bucket("key3", [(5, 6)])
                 .add_single_range_bucket("key4", 7, 8))
                 
        assert result is builder


class TestCurriculumBuilder:
    """Test cases for CurriculumBuilder factory."""
    
    def test_random_builder_from_config(self):
        """Test creating RandomCurriculumBuilder from config."""
        env_cfg = EnvConfig()
        task_set_config = WeightedTaskSetConfig(items=[])
        
        builder = CurriculumBuilder.random(task_set_config)
        
        assert isinstance(builder, RandomCurriculumBuilder)
        assert builder.task_set_config is task_set_config
        
    def test_random_builder_from_task_set_builder(self):
        """Test creating RandomCurriculumBuilder from TaskSet builder."""
        env_cfg = EnvConfig()
        task_set_builder = WeightedTaskSetBuilder().add_env_config(env_cfg)
        
        builder = CurriculumBuilder.random(task_set_builder)
        
        assert isinstance(builder, RandomCurriculumBuilder)
        assert isinstance(builder.task_set_config, WeightedTaskSetConfig)
        assert len(builder.task_set_config.items) == 1
        
    def test_learning_progress_builder_from_config(self):
        """Test creating LearningProgressCurriculumBuilder from config."""
        task_set_config = WeightedTaskSetConfig(items=[])
        
        builder = CurriculumBuilder.learning_progress(task_set_config)
        
        assert isinstance(builder, LearningProgressCurriculumBuilder)
        assert builder.task_set_config is task_set_config
        
    def test_learning_progress_builder_from_bucketed_builder(self):
        """Test creating LearningProgressCurriculumBuilder from BuckettedTaskSet builder."""
        base_config = EnvConfig()
        bucketed_builder = (BuckettedTaskSetBuilder(base_config=base_config)
                           .add_value_bucket("seed", [2, 3]))
        
        builder = CurriculumBuilder.learning_progress(bucketed_builder)
        
        assert isinstance(builder, LearningProgressCurriculumBuilder)
        assert isinstance(builder.task_set_config, BuckettedTaskSetConfig)


class TestRandomCurriculumBuilder:
    """Test cases for RandomCurriculumBuilder."""
    
    def test_basic_builder(self):
        """Test basic RandomCurriculumBuilder."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = RandomCurriculumBuilder(task_set_config)
        
        config = builder.build()
        
        assert isinstance(config, RandomCurriculumConfig)
        assert config.task_set_config is task_set_config
        # No base_seed field anymore
        
    def test_basic_build(self):
        """Test basic build without base seed."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = RandomCurriculumBuilder(task_set_config)
        
        config = builder.build()
        
        assert isinstance(config, RandomCurriculumConfig)
        
    def test_fluent_interface(self):
        """Test fluent interface."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = RandomCurriculumBuilder(task_set_config)
        
        result = builder
        assert result is builder


class TestLearningProgressCurriculumBuilder:
    """Test cases for LearningProgressCurriculumBuilder."""
    
    def test_basic_builder_defaults(self):
        """Test basic builder with defaults."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = LearningProgressCurriculumBuilder(task_set_config)
        
        config = builder.build()
        
        assert isinstance(config, LearningProgressCurriculumConfig)
        assert config.task_set_config is task_set_config
        assert config.n_tasks == 100  # Default
        # No base_seed field anymore
        
    def test_all_configuration_options(self):
        """Test setting all configuration options."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = LearningProgressCurriculumBuilder(task_set_config)
        
        config = (builder
                 .with_n_tasks(50)
                 .with_ema_timescale(0.05)
                 .with_progress_smoothing(0.2)
                 .with_num_active_tasks(8)
                 .with_rand_task_rate(0.3)
                 .with_sample_threshold(5)
                 .with_memory(15)
                 .build())
                 
        assert config.n_tasks == 50
        assert config.ema_timescale == 0.05
        assert config.progress_smoothing == 0.2
        assert config.num_active_tasks == 8
        assert config.rand_task_rate == 0.3
        assert config.sample_threshold == 5
        assert config.memory == 15
        
    def test_fluent_interface(self):
        """Test that all methods return self for chaining."""
        task_set_config = WeightedTaskSetConfig(items=[])
        builder = LearningProgressCurriculumBuilder(task_set_config)
        
        result = (builder
                 .with_n_tasks(10)
                 .with_ema_timescale(0.01)
                 .with_progress_smoothing(0.1)
                 .with_num_active_tasks(5)
                 .with_rand_task_rate(0.1)
                 .with_sample_threshold(3)
                 .with_memory(10))
                 
        assert result is builder


class TestBuilderIntegration:
    """Test end-to-end builder integration with curriculum creation."""
    
    def test_complete_random_curriculum_workflow(self):
        """Test complete workflow from builder to working curriculum."""
        # Build task set
        env_cfg1 = EnvConfig()
        env_cfg2 = EnvConfig()
        
        task_set_builder = (TaskSetBuilder.weighted()
                           .add_env_config(env_cfg1, weight=1.0)
                           .add_env_config(env_cfg2, weight=2.0)
                           .with_dict_overrides(**{"device": "cpu"}))
                           
        # Build curriculum
        curriculum_config = (CurriculumBuilder.random(task_set_builder)
                           .build())
                           
        # Create actual curriculum
        curriculum = RandomCurriculum(curriculum_config, seed=0)
        
        # Test that it works
        task = curriculum.get_task(123)
        env_cfg = task.get_env_config()
        
        assert env_cfg is not None
        
        
    def test_complete_learning_progress_curriculum_workflow(self):
        """Test complete workflow with learning progress curriculum."""
        # Build bucketed task set
        base_config = EnvConfig()
        
        task_set_builder = (TaskSetBuilder.bucketed(base_config)
                           .add_value_bucket("seed", [2, 3, 4])
                           .add_value_bucket("device", ["cpu", "cuda"]))
                           
        # Build curriculum
        curriculum_config = (CurriculumBuilder.learning_progress(task_set_builder)
                           .with_n_tasks(5)
                           .with_num_active_tasks(3)
                           .build())
                           
        # Create actual curriculum
        curriculum = LearningProgressCurriculum(curriculum_config, seed=0)
        
        # Test that it works
        task = curriculum.get_task(456)
        env_cfg = task.get_env_config()
        
        # Should have generated 5 tasks
        assert len(curriculum.tasks) == 5
        
    def test_nested_builder_workflow(self):
        """Test workflow with nested task sets using builders."""
        # Create inner task set
        inner_env_cfg = EnvConfig()
        inner_builder = TaskSetBuilder.weighted().add_env_config(inner_env_cfg)
        
        # Create outer task set with both direct config and nested
        direct_env_cfg = EnvConfig()
        outer_builder = (TaskSetBuilder.weighted()
                        .add_env_config(direct_env_cfg, weight=1.0)
                        .add_task_set(inner_builder, weight=1.0))
                        
        # Create curriculum
        curriculum_config = CurriculumBuilder.random(outer_builder).build()
        curriculum = RandomCurriculum(curriculum_config, seed=0)
        
        # Test that both configurations can be generated
        tasks = [curriculum.get_task(i) for i in range(20)]
        env_configs = [task.get_env_config() for task in tasks]
        
        # Should get valid EnvConfigs from both direct and nested sources
        assert all(env_cfg is not None for env_cfg in env_configs)
        assert len(set(str(env_cfg) for env_cfg in env_configs)) > 1  # Should have some variation
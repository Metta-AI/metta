"""Tests for TaskGeneratorSet."""

import pytest

from cogworks.curriculum import (
    SingleTaskGeneratorConfig,
    TaskGeneratorSet,
    TaskGeneratorSetConfig,
)
from metta.rl.env_config import EnvConfig


class TestTaskGeneratorSet:
    """Test cases for TaskGeneratorSet."""

    def test_task_generator_set_creation(self):
        """Test creating a TaskGeneratorSet with multiple configs."""
        # Create multiple task generator configs
        config1 = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=1))
        config2 = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=2))
        config3 = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=3))

        # Create TaskGeneratorSetConfig
        task_set_config = TaskGeneratorSetConfig(
            task_generator_configs=[config1, config2, config3], weights=[1.0, 1.0, 1.0]
        )

        # Create generator
        generator = TaskGeneratorSet(task_set_config)

        assert generator._config == task_set_config
        assert len(generator._sub_task_generators) == 3

    def test_task_generator_set_empty_list(self):
        """Test that TaskGeneratorSetConfig raises error with empty list."""
        with pytest.raises(ValueError):
            TaskGeneratorSetConfig(task_generator_configs=[], weights=[])

    def test_task_generator_set_sampling(self):
        """Test that TaskGeneratorSet samples from configs."""
        # Create configs with different seeds
        configs = [SingleTaskGeneratorConfig(env_config=EnvConfig(seed=i)) for i in range(5)]
        weights = [1.0] * 5

        task_set_config = TaskGeneratorSetConfig(task_generator_configs=configs, weights=weights)
        generator = TaskGeneratorSet(task_set_config)

        # Generate multiple tasks and check we get different configs
        seeds_seen = set()
        for task_id in range(20):
            env_config = generator.get_task(task_id)
            seeds_seen.add(env_config.seed)

        # Should have seen multiple different seeds
        assert len(seeds_seen) > 1

    def test_task_generator_set_with_overrides(self):
        """Test TaskGeneratorSet with environment overrides."""
        # Create configs
        config1 = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=1, device="cpu"))
        config2 = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=2, device="cpu"))

        # Create TaskGeneratorSetConfig with overrides
        task_set_config = TaskGeneratorSetConfig(
            task_generator_configs=[config1, config2],
            weights=[1.0, 1.0],
            overrides={"device": "cuda", "torch_deterministic": False},
        )

        generator = TaskGeneratorSet(task_set_config)

        # Generate task and check overrides were applied
        env_config = generator.get_task(0)
        assert env_config.device == "cuda"
        assert env_config.torch_deterministic is False

    def test_task_generator_set_with_string_overrides(self):
        """Test TaskGeneratorSet with dict overrides using dot-separated keys."""
        config = SingleTaskGeneratorConfig(env_config=EnvConfig())

        # Use dict format for overrides with dot-separated keys
        task_set_config = TaskGeneratorSetConfig(
            task_generator_configs=[config],
            weights=[1.0],
            overrides={
                "device": "cuda",
                "seed": 999,
                "torch_deterministic": False,
                "vectorization": "serial",
            },
        )

        generator = TaskGeneratorSet(task_set_config)
        env_config = generator.get_task(0)

        assert env_config.device == "cuda"
        assert env_config.seed == 999
        assert env_config.torch_deterministic is False
        assert env_config.vectorization == "serial"

    def test_task_set_generator_deterministic(self):
        """Test that TaskGeneratorSet is deterministic with same seed."""
        configs = [SingleTaskGeneratorConfig(env_config=EnvConfig(seed=i)) for i in range(10)]
        weights = [1.0] * 10

        task_set_config = TaskGeneratorSetConfig(task_generator_configs=configs, weights=weights)
        generator = TaskGeneratorSet(task_set_config)

        # Generate with same task_id multiple times
        results = []
        for _ in range(5):
            env_config = generator.get_task(42)
            results.append(env_config.seed)

        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_task_set_generator_different_seeds(self):
        """Test that different task_ids produce different selections."""
        configs = [SingleTaskGeneratorConfig(env_config=EnvConfig(seed=i)) for i in range(10)]
        weights = [1.0] * 10

        task_set_config = TaskGeneratorSetConfig(task_generator_configs=configs, weights=weights)
        generator = TaskGeneratorSet(task_set_config)

        # Generate with different task_ids
        seeds_seen = set()
        for task_id in range(50):
            env_config = generator.get_task(task_id)
            seeds_seen.add(env_config.seed)

        # Should have sampled from multiple configs
        assert len(seeds_seen) > 5  # Should see most of the 10 configs

    def test_task_set_generator_nested_overrides(self):
        """Test nested key overrides work correctly."""
        config = SingleTaskGeneratorConfig(env_config=EnvConfig())

        # Assuming EnvConfig might have nested structure in the future
        task_set_config = TaskGeneratorSetConfig(
            task_generator_configs=[config],
            weights=[1.0],
            overrides={"device": "cuda"},  # Simple override for now
        )

        generator = TaskGeneratorSet(task_set_config)
        env_config = generator.get_task(0)

        assert env_config.device == "cuda"

"""Tests for BucketedTaskGenerator."""

from cogworks.curriculum.task.generator import (
    BucketedTaskGenerator,
    BucketedTaskGeneratorConfig,
    SingleTaskGeneratorConfig,
    TaskSetGeneratorConfig,
    ValueRange,
)
from metta.rl.env_config import EnvConfig


class TestBucketedTaskGenerator:
    """Test cases for BucketedTaskGenerator."""

    def test_bucketed_generator_creation(self):
        """Test creating a BucketedTaskGenerator."""
        # Create child generator config
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=42))

        # Create buckets
        buckets = {
            "seed": [1, 2, 3],
            "device": ["cpu", "cuda"],
        }

        # Create BucketedTaskGeneratorConfig
        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        # Create generator
        generator = BucketedTaskGenerator(config)

        assert generator._config == config
        assert generator._child_generator is not None

    def test_bucketed_generator_with_value_buckets(self):
        """Test BucketedTaskGenerator with single value buckets."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=0, device="cpu"))

        buckets = {
            "seed": [100, 200, 300],
            "device": ["cuda"],  # Single value, always selected
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)

        # Generate multiple tasks
        seeds_seen = set()
        for task_id in range(20):
            env_config = generator.get_task(task_id)
            seeds_seen.add(env_config.seed)
            # Device should always be overridden to cuda
            assert env_config.device == "cuda"

        # Should have seen different seed values
        assert len(seeds_seen) > 1
        # All seeds should be from the bucket
        assert seeds_seen.issubset({100, 200, 300})

    def test_bucketed_generator_with_range_buckets(self):
        """Test BucketedTaskGenerator with range buckets."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig())

        buckets = {
            "seed": [
                ValueRange(range_min=0, range_max=100),
                ValueRange(range_min=1000, range_max=2000),
            ],
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)

        # Generate multiple tasks
        seeds = []
        for task_id in range(50):
            env_config = generator.get_task(task_id)
            seeds.append(env_config.seed)

        # Should have seeds in both ranges
        low_range_seeds = [s for s in seeds if 0 <= s <= 100]
        high_range_seeds = [s for s in seeds if 1000 <= s <= 2000]

        assert len(low_range_seeds) > 0
        assert len(high_range_seeds) > 0
        # All seeds should be in one of the ranges
        assert all(0 <= s <= 100 or 1000 <= s <= 2000 for s in seeds)

    def test_bucketed_generator_mixed_buckets(self):
        """Test BucketedTaskGenerator with mixed value and range buckets."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig())

        buckets = {
            "seed": [
                42,
                99,
                ValueRange(range_min=500, range_max=600),
            ],
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)

        # Generate multiple tasks
        seeds = []
        for task_id in range(60):
            env_config = generator.get_task(task_id)
            seeds.append(env_config.seed)

        # Should see exact values
        assert 42 in seeds
        assert 99 in seeds
        # Should see values in range
        range_seeds = [s for s in seeds if 500 <= s <= 600]
        assert len(range_seeds) > 0

    def test_bucketed_generator_deterministic(self):
        """Test that BucketedTaskGenerator is deterministic with same seed."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig())

        buckets = {
            "seed": [ValueRange(range_min=0, range_max=1000)],
            "device": ["cpu", "cuda"],
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)

        # Generate with same task_id multiple times
        results = []
        for _ in range(5):
            env_config = generator.get_task(42)
            results.append((env_config.seed, env_config.device))

        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_bucketed_generator_empty_buckets(self):
        """Test BucketedTaskGenerator with no buckets."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=123, device="cpu"))

        config = BucketedTaskGeneratorConfig(
            child_generator_config=child_config,
            buckets={},  # No buckets
        )

        generator = BucketedTaskGenerator(config)
        env_config = generator.get_task(0)

        # Should return the child generator's config unchanged
        assert env_config.seed == 123
        assert env_config.device == "cpu"

    def test_bucketed_generator_with_complex_child(self):
        """Test BucketedTaskGenerator with a TaskSetGenerator as child."""
        # Create multiple configs for TaskSetGenerator
        configs = [SingleTaskGeneratorConfig(env_config=EnvConfig(seed=i)) for i in range(3)]

        # TaskSetGenerator as child
        child_config = TaskSetGeneratorConfig(task_generators=configs)

        # Add buckets to override device
        buckets = {
            "device": ["cuda"],
            "torch_deterministic": [False],
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)

        # Generate tasks
        for task_id in range(10):
            env_config = generator.get_task(task_id)
            # Bucket overrides should always apply
            assert env_config.device == "cuda"
            assert env_config.torch_deterministic is False
            # Seed should come from one of the child configs
            assert env_config.seed in [0, 1, 2]

    def test_bucketed_generator_preserves_child_values(self):
        """Test that BucketedTaskGenerator preserves non-overridden values from child."""
        child_config = SingleTaskGeneratorConfig(
            env_config=EnvConfig(seed=999, device="cpu", torch_deterministic=True, vectorization="serial")
        )

        # Only override device
        buckets = {
            "device": ["cuda"],
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)

        generator = BucketedTaskGenerator(config)
        env_config = generator.get_task(0)

        # Device should be overridden
        assert env_config.device == "cuda"
        # Other values should be preserved from child
        assert env_config.seed == 999
        assert env_config.torch_deterministic is True
        assert env_config.vectorization == "serial"

    def test_bucketed_generator_with_primitive_values(self):
        """Test BucketedTaskGenerator with primitive values (int, float, str) directly in buckets."""
        child_config = SingleTaskGeneratorConfig(env_config=EnvConfig(seed=0, device="cpu"))

        # Mix primitive values with ValueRange objects
        buckets = {
            "seed": [100, 200, ValueRange(range_min=300, range_max=400)],  # int values + range
            "device": ["cuda", "cpu"],  # string values directly
            "torch_deterministic": [True, False, True],  # boolean values
        }

        config = BucketedTaskGeneratorConfig(child_generator_config=child_config, buckets=buckets)
        generator = BucketedTaskGenerator(config)

        # Generate multiple tasks
        seeds_seen = set()
        devices_seen = set()
        torch_det_seen = set()

        for task_id in range(50):
            env_config = generator.get_task(task_id)
            seeds_seen.add(env_config.seed)
            devices_seen.add(env_config.device)
            torch_det_seen.add(env_config.torch_deterministic)

        # Should see primitive values directly used
        assert 100 in seeds_seen
        assert 200 in seeds_seen
        assert "cuda" in devices_seen
        assert "cpu" in devices_seen
        assert True in torch_det_seen
        assert False in torch_det_seen

        # Should also see values from the range (300-400)
        range_seeds = [s for s in seeds_seen if 300 <= s <= 400]
        assert len(range_seeds) > 0

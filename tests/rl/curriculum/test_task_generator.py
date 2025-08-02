"""Unit tests for TaskGenerator classes."""

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from metta.rl.curriculum import (
    BucketedTaskGenerator,
    CompositeTaskGenerator,
    RandomTaskGenerator,
    SampledTaskGenerator,
    TaskGenerator,
    create_task_generator_from_config,
)


class ConcreteTaskGenerator(TaskGenerator):
    """Concrete implementation of TaskGenerator for testing."""

    def _apply_task_params(self, env_cfg: DictConfig, rng: np.random.RandomState, task_id: int):
        """Simple implementation that adds a test parameter."""
        env_cfg.test_param = rng.rand()


class TestTaskGenerator:
    """Test suite for base TaskGenerator class."""

    def test_abstract_class(self):
        """Test that TaskGenerator is abstract."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}})

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            TaskGenerator(base_config)

    def test_concrete_implementation(self):
        """Test concrete implementation of TaskGenerator."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}, "map": {"width": 32, "height": 32}})

        generator = ConcreteTaskGenerator(base_config)

        # Generate a task
        env_cfg = generator.generate(task_id=12345)

        # Check base config is preserved
        assert env_cfg.game.num_agents == 1
        assert env_cfg.map.width == 32
        assert env_cfg.map.height == 32

        # Check task ID is set
        assert env_cfg.task.id == 12345

        # Check test parameter was added
        assert "test_param" in env_cfg
        assert 0 <= env_cfg.test_param <= 1

    def test_deterministic_generation(self):
        """Test that same task_id produces same config."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}})
        generator = ConcreteTaskGenerator(base_config)

        # Generate same task multiple times
        configs = []
        for _ in range(5):
            configs.append(generator.generate(task_id=54321))

        # All configs should be identical
        for i in range(1, 5):
            assert configs[i].test_param == configs[0].test_param


class TestBucketedTaskGenerator:
    """Test suite for BucketedTaskGenerator."""

    def test_initialization(self):
        """Test BucketedTaskGenerator initialization."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}, "map": {"width": 32}})

        buckets = {
            "game.num_agents": {"range": [1, 4], "bins": 3},
            "map.width": [16, 32, 64],
            "game.difficulty": {"range": [0.0, 1.0], "bins": 5},
        }

        generator = BucketedTaskGenerator(base_config, buckets, default_bins=2)

        # Check buckets were expanded correctly
        assert len(generator.buckets) == 3
        assert len(generator.buckets["game.num_agents"]) == 3  # 3 bins
        assert len(generator.buckets["map.width"]) == 3  # 3 discrete values
        assert len(generator.buckets["game.difficulty"]) == 5  # 5 bins

    def test_range_bucket_expansion(self):
        """Test expansion of range-based buckets."""
        base_config = OmegaConf.create({})

        buckets = {"param1": {"range": [0, 10], "bins": 5}, "param2": {"range": [1.0, 2.0], "bins": 2}}

        generator = BucketedTaskGenerator(base_config, buckets)

        # Check integer ranges
        param1_buckets = generator.buckets["param1"]
        assert len(param1_buckets) == 5
        assert param1_buckets[0]["range"] == (0, 2)
        assert param1_buckets[0]["want_int"] == True

        # Check float ranges
        param2_buckets = generator.buckets["param2"]
        assert len(param2_buckets) == 2
        assert param2_buckets[0]["range"] == (1.0, 1.5)
        assert param2_buckets[0]["want_int"] == False

    def test_parameter_sampling(self):
        """Test that parameters are sampled correctly."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}, "map": {"width": 32}})

        buckets = {
            "game.num_agents": {"range": [1, 4], "bins": 3},
            "map.width": [16, 32, 64, 128],
            "game.spawn_rate": {"range": [0.1, 0.9], "bins": 4},
        }

        generator = BucketedTaskGenerator(base_config, buckets)

        # Generate multiple tasks and check parameters
        for task_id in range(100):
            env_cfg = generator.generate(task_id)

            # Check num_agents is in valid range
            assert 1 <= env_cfg.game.num_agents <= 4
            assert isinstance(env_cfg.game.num_agents, int)

            # Check width is one of the discrete values
            assert env_cfg.map.width in [16, 32, 64, 128]

            # Check spawn_rate is in valid range
            assert 0.1 <= env_cfg.game.spawn_rate <= 0.9
            assert isinstance(env_cfg.game.spawn_rate, float)

    def test_nested_parameter_setting(self):
        """Test setting deeply nested parameters."""
        base_config = OmegaConf.create({})

        buckets = {"a.b.c.d": [1, 2, 3], "x.y.z": {"range": [0.0, 1.0], "bins": 2}}

        generator = BucketedTaskGenerator(base_config, buckets)
        env_cfg = generator.generate(task_id=789)

        # Check nested values were set
        assert env_cfg.a.b.c.d in [1, 2, 3]
        assert 0.0 <= env_cfg.x.y.z <= 1.0

    def test_invalid_bucket_spec(self):
        """Test that invalid bucket specifications raise errors."""
        base_config = OmegaConf.create({})

        # Invalid bucket spec (not dict or list)
        buckets = {"param": "invalid"}

        with pytest.raises(ValueError, match="must be.*or list"):
            BucketedTaskGenerator(base_config, buckets)


class TestRandomTaskGenerator:
    """Test suite for RandomTaskGenerator."""

    def test_initialization(self):
        """Test RandomTaskGenerator initialization."""
        task_configs = {
            "easy": OmegaConf.create({"difficulty": 0.3}),
            "medium": OmegaConf.create({"difficulty": 0.6}),
            "hard": OmegaConf.create({"difficulty": 0.9}),
        }

        weights = {"easy": 0.5, "medium": 0.3, "hard": 0.2}

        generator = RandomTaskGenerator(task_configs, weights)

        # Check weights are normalized
        total_weight = sum(generator.weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_uniform_weights(self):
        """Test initialization with uniform weights."""
        task_configs = {
            "task1": OmegaConf.create({"type": 1}),
            "task2": OmegaConf.create({"type": 2}),
            "task3": OmegaConf.create({"type": 3}),
        }

        generator = RandomTaskGenerator(task_configs)  # No weights provided

        # Check all weights are equal
        for weight in generator.weights.values():
            assert abs(weight - 1 / 3) < 1e-6

    def test_task_selection(self):
        """Test that tasks are selected according to weights."""
        task_configs = {"common": OmegaConf.create({"value": 1}), "rare": OmegaConf.create({"value": 2})}

        weights = {"common": 0.9, "rare": 0.1}
        generator = RandomTaskGenerator(task_configs, weights)

        # Generate many tasks and count types
        type_counts = {"common": 0, "rare": 0}
        num_samples = 1000

        for i in range(num_samples):
            env_cfg = generator.generate(task_id=i)
            task_type = env_cfg.task.type
            type_counts[task_type] += 1

        # Check distribution roughly matches weights
        common_ratio = type_counts["common"] / num_samples
        assert 0.85 < common_ratio < 0.95  # Should be around 0.9

    def test_config_merging(self):
        """Test that task configs are properly merged."""
        _ = OmegaConf.create({"game": {"num_agents": 1, "max_steps": 1000}})

        task_configs = {
            "combat": OmegaConf.create({"game": {"type": "combat", "num_agents": 2}, "weapons": {"enabled": True}}),
            "exploration": OmegaConf.create({"game": {"type": "exploration"}, "map": {"size": "large"}}),
        }

        generator = RandomTaskGenerator(task_configs)

        # Force selection of combat task
        np.random.seed(42)  # Set seed to ensure combat is selected
        env_cfg = generator.generate(task_id=1)

        # Check merging worked correctly
        if env_cfg.task.type == "combat":
            assert env_cfg.game.type == "combat"
            assert env_cfg.game.num_agents == 2  # Overridden
            assert env_cfg.weapons.enabled
        elif env_cfg.task.type == "exploration":
            assert env_cfg.game.type == "exploration"
            assert env_cfg.map.size == "large"


class TestSampledTaskGenerator:
    """Test suite for SampledTaskGenerator."""

    def test_initialization(self):
        """Test SampledTaskGenerator initialization."""
        base_config = OmegaConf.create({"game": {"num_agents": 1}})

        sampling_params = {
            "game.difficulty": {"range": (0.3, 0.7), "want_int": False},
            "map.terrain": "forest",
            "spawn.rate": 0.5,
        }

        generator = SampledTaskGenerator(base_config, sampling_params)
        assert generator.sampling_parameters == sampling_params

    def test_parameter_application(self):
        """Test that sampling parameters are applied correctly."""
        base_config = OmegaConf.create({})

        sampling_params = {
            "game.mode": "survival",
            "difficulty.level": {"range": (1, 10), "want_int": True},
            "physics.gravity": {"range": (9.0, 10.0), "want_int": False},
        }

        generator = SampledTaskGenerator(base_config, sampling_params)

        # Generate multiple configs
        for task_id in range(50):
            env_cfg = generator.generate(task_id)

            # Check static values
            assert env_cfg.game.mode == "survival"

            # Check ranged values
            assert 1 <= env_cfg.difficulty.level <= 10
            assert isinstance(env_cfg.difficulty.level, int)

            assert 9.0 <= env_cfg.physics.gravity <= 10.0
            assert isinstance(env_cfg.physics.gravity, float)


class TestCompositeTaskGenerator:
    """Test suite for CompositeTaskGenerator."""

    def test_initialization(self):
        """Test CompositeTaskGenerator initialization."""
        base_config = OmegaConf.create({})

        # Create sub-generators
        gen1 = BucketedTaskGenerator(base_config, {"param1": [1, 2, 3]})
        gen2 = RandomTaskGenerator({"a": OmegaConf.create({"type": "a"}), "b": OmegaConf.create({"type": "b"})})

        composite = CompositeTaskGenerator([gen1, gen2], weights=[0.7, 0.3])

        assert len(composite.generators) == 2
        assert abs(sum(composite.weights) - 1.0) < 1e-6

    def test_empty_generators_list(self):
        """Test that empty generators list raises error."""
        with pytest.raises(ValueError, match="at least one generator"):
            CompositeTaskGenerator([])

    def test_generator_selection(self):
        """Test that generators are selected according to weights."""
        base_config = OmegaConf.create({})

        # Create generators with distinct outputs
        gen1 = ConcreteTaskGenerator(base_config)
        gen2 = BucketedTaskGenerator(base_config, {"marker": [1]})
        gen3 = BucketedTaskGenerator(base_config, {"marker": [2]})

        composite = CompositeTaskGenerator([gen1, gen2, gen3], weights=[0.5, 0.3, 0.2])

        # Generate many tasks and count which generator was used
        generator_counts = {0: 0, 1: 0, 2: 0}
        num_samples = 1000

        for i in range(num_samples):
            env_cfg = composite.generate(task_id=i)
            gen_idx = env_cfg.task.generator_idx
            generator_counts[gen_idx] += 1

        # Check distribution roughly matches weights
        assert 0.45 < generator_counts[0] / num_samples < 0.55
        assert 0.25 < generator_counts[1] / num_samples < 0.35
        assert 0.15 < generator_counts[2] / num_samples < 0.25


class TestFactoryFunction:
    """Test suite for create_task_generator_from_config."""

    def test_bucketed_generator_creation(self):
        """Test creating BucketedTaskGenerator from config."""
        config = OmegaConf.create(
            {
                "_target_": "bucketed",
                "base_config": {"game": {"type": "test"}},
                "buckets": {"param": [1, 2, 3]},
                "default_bins": 5,
            }
        )

        generator = create_task_generator_from_config(config)
        assert isinstance(generator, BucketedTaskGenerator)

    def test_random_generator_creation(self):
        """Test creating RandomTaskGenerator from config."""
        config = OmegaConf.create(
            {
                "_target_": "random",
                "task_configs": {"easy": {"difficulty": 0.3}, "hard": {"difficulty": 0.9}},
                "weights": {"easy": 0.7, "hard": 0.3},
            }
        )

        generator = create_task_generator_from_config(config)
        assert isinstance(generator, RandomTaskGenerator)

    def test_unknown_generator_type(self):
        """Test that unknown generator type raises error."""
        config = OmegaConf.create({"_target_": "unknown_type"})

        with pytest.raises(ValueError, match="Unknown generator type"):
            create_task_generator_from_config(config)

    def test_case_insensitive_matching(self):
        """Test that generator type matching is case-insensitive."""
        config = OmegaConf.create(
            {
                "_target_": "BuCkEtEd",  # Mixed case
                "base_config": {},
                "buckets": {},
            }
        )

        generator = create_task_generator_from_config(config)
        assert isinstance(generator, BucketedTaskGenerator)


if __name__ == "__main__":
    pytest.main([__file__])

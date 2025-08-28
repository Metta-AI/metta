import numpy as np
import pytest
from pydantic import BaseModel

from metta.map.random.float import (
    FloatConstantDistribution,
    FloatDistribution,
    FloatLognormalDistribution,
    FloatUniformDistribution,
)


class TestFloatConstantDistribution:
    def test_sample_returns_constant_value(self):
        dist = FloatConstantDistribution(value=3.14)
        rng = np.random.default_rng(seed=123)

        # Should always return the same value regardless of rng
        assert dist.sample(rng) == 3.14
        assert dist.sample(rng) == 3.14
        assert dist.sample(rng) == 3.14

    def test_sample_with_negative_value(self):
        dist = FloatConstantDistribution(value=-2.5)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == -2.5

    def test_sample_with_zero(self):
        dist = FloatConstantDistribution(value=0.0)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == 0.0

    def test_sample_with_integer_value(self):
        dist = FloatConstantDistribution(value=42.0)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == 42.0


class TestFloatUniformDistribution:
    def test_sample_within_range(self):
        dist = FloatUniformDistribution(low=1.0, high=10.0)
        rng = np.random.default_rng(seed=123)

        # Sample many times and check all are within range
        samples = [dist.sample(rng) for _ in range(100)]

        assert all(1.0 <= sample <= 10.0 for sample in samples)
        assert len(set(samples)) > 50  # Should get many different values

    def test_sample_small_range(self):
        dist = FloatUniformDistribution(low=5.0, high=5.1)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(5.0 <= sample <= 5.1 for sample in samples)

    def test_sample_negative_range(self):
        dist = FloatUniformDistribution(low=-10.5, high=-1.2)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(-10.5 <= sample <= -1.2 for sample in samples)

    def test_sample_zero_crossing_range(self):
        dist = FloatUniformDistribution(low=-2.5, high=2.5)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(-2.5 <= sample <= 2.5 for sample in samples)

    def test_sample_deterministic_with_seed(self):
        dist = FloatUniformDistribution(low=0.0, high=1.0)

        # Same seed should produce same sequence
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)

        samples1 = [dist.sample(rng1) for _ in range(10)]
        samples2 = [dist.sample(rng2) for _ in range(10)]

        assert samples1 == samples2


class TestFloatLognormalDistribution:
    def test_sample_basic_range(self):
        dist = FloatLognormalDistribution(low=1.0, high=10.0)
        rng = np.random.default_rng(seed=123)

        # Sample many times - should be generally within reasonable bounds
        samples = [dist.sample(rng) for _ in range(1000)]

        # Most samples should be positive (lognormal property)
        assert all(sample > 0 for sample in samples)

        # Should have some variety
        assert len(set(samples)) > 100

    def test_sample_with_max_constraint(self):
        dist = FloatLognormalDistribution(low=1.0, high=10.0, max=15.0)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(1000)]

        # All samples should be <= max
        assert all(sample <= 15.0 for sample in samples)
        assert all(sample > 0 for sample in samples)

    def test_sample_deterministic_with_seed(self):
        dist = FloatLognormalDistribution(low=1.0, high=5.0)

        # Same seed should produce same sequence
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)

        samples1 = [dist.sample(rng1) for _ in range(10)]
        samples2 = [dist.sample(rng2) for _ in range(10)]

        assert samples1 == samples2

    def test_invalid_parameters_low_greater_than_high(self):
        dist = FloatLognormalDistribution(low=10.0, high=1.0)
        rng = np.random.default_rng(seed=123)

        with pytest.raises(ValueError, match="Low value must be less than high value"):
            dist.sample(rng)

    def test_invalid_parameters_low_zero_or_negative(self):
        dist = FloatLognormalDistribution(low=0.0, high=10.0)
        rng = np.random.default_rng(seed=123)

        with pytest.raises(ValueError, match="Low value must be above 0"):
            dist.sample(rng)

        dist = FloatLognormalDistribution(low=-1.0, high=10.0)

        with pytest.raises(ValueError, match="Low value must be above 0"):
            dist.sample(rng)

    def test_lognormal_from_90_percentile_edge_case(self):
        # Test with very small range
        dist = FloatLognormalDistribution(low=0.1, high=0.2)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(sample > 0 for sample in samples)


class TestFloatDistributionTypes:
    def test_constant_distribution_from_float(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": 3.14})
        assert isinstance(model.dist, FloatConstantDistribution)
        assert model.dist.value == 3.14

    def test_uniform_distribution_from_tuple(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("uniform", 1.0, 10.0)})
        assert isinstance(model.dist, FloatUniformDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0

    def test_lognormal_distribution_two_args(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 10.0)})
        assert isinstance(model.dist, FloatLognormalDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0
        assert model.dist.max is None

    def test_lognormal_distribution_three_args(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 10.0, 15.0)})
        assert isinstance(model.dist, FloatLognormalDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0
        assert model.dist.max == 15.0

    def test_invalid_distribution_format(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        # Wrong distribution type
        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("normal", 1.0, 10.0)})

        # Wrong number of arguments for uniform
        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("uniform", 1.0)})

    def test_integration_constant_distribution(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": 2.718})
        rng = np.random.default_rng(seed=123)

        assert model.dist.sample(rng) == 2.718

    def test_integration_uniform_distribution(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("uniform", 0.5, 1.5)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(0.5 <= sample <= 1.5 for sample in samples)

    def test_integration_lognormal_distribution(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 5.0)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(sample > 0 for sample in samples)

    def test_integration_lognormal_with_max(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 5.0, 10.0)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(100)]
        assert all(0 < sample <= 10.0 for sample in samples)

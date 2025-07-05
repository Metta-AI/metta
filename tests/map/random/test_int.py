import numpy as np
import pytest

from metta.map.random.int import (
    IntConstantDistribution,
    IntDistribution,
    IntUniformDistribution,
)


class TestIntConstantDistribution:
    def test_sample_returns_constant_value(self):
        dist = IntConstantDistribution(value=42)
        rng = np.random.default_rng(seed=123)

        # Should always return the same value regardless of rng
        assert dist.sample(rng) == 42
        assert dist.sample(rng) == 42
        assert dist.sample(rng) == 42

    def test_sample_with_negative_value(self):
        dist = IntConstantDistribution(value=-10)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == -10

    def test_sample_with_zero(self):
        dist = IntConstantDistribution(value=0)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == 0


class TestIntUniformDistribution:
    def test_sample_within_range(self):
        dist = IntUniformDistribution(low=1, high=10)
        rng = np.random.default_rng(seed=123)

        # Sample many times and check all are within range
        samples = [dist.sample(rng) for _ in range(100)]

        assert all(1 <= sample <= 10 for sample in samples)
        assert len(set(samples)) > 1  # Should get different values

    def test_sample_single_value_range(self):
        # When low == high, should always return that value
        dist = IntUniformDistribution(low=5, high=5)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(10)]
        assert all(sample == 5 for sample in samples)

    def test_sample_negative_range(self):
        dist = IntUniformDistribution(low=-10, high=-1)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(-10 <= sample <= -1 for sample in samples)

    def test_sample_zero_crossing_range(self):
        dist = IntUniformDistribution(low=-5, high=5)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(-5 <= sample <= 5 for sample in samples)

    def test_sample_deterministic_with_seed(self):
        dist = IntUniformDistribution(low=1, high=100)

        # Same seed should produce same sequence
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)

        samples1 = [dist.sample(rng1) for _ in range(10)]
        samples2 = [dist.sample(rng2) for _ in range(10)]

        assert samples1 == samples2


class TestIntDistributionTypes:
    def test_constant_distribution_from_int(self):
        # Test that int values are converted to IntConstantDistribution
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": 42})
        assert isinstance(model.dist, IntConstantDistribution)
        assert model.dist.value == 42

    def test_uniform_distribution_from_tuple(self):
        # Test that tuple values are converted to IntUniformDistribution
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": ("uniform", 1, 10)})
        assert isinstance(model.dist, IntUniformDistribution)
        assert model.dist.low == 1
        assert model.dist.high == 10

    def test_invalid_uniform_tuple_format(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: IntDistribution

        # Wrong number of elements
        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("uniform", 1)})

        # Wrong first element
        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("normal", 1, 10)})

    def test_integration_constant_distribution(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": 7})
        rng = np.random.default_rng(seed=123)

        assert model.dist.sample(rng) == 7

    def test_integration_uniform_distribution(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": ("uniform", 5, 15)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(5 <= sample <= 15 for sample in samples)

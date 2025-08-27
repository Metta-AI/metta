import numpy as np
import pytest
from pydantic import BaseModel

from metta.map.random.float import (
    FloatConstantDistribution,
    FloatDistribution,
    FloatLognormalDistribution,
    FloatUniformDistribution,
)
from metta.map.random.int import (
    IntConstantDistribution,
    IntDistribution,
    IntUniformDistribution,
)


class TestFloatConstantDistribution:
    def test_sample_returns_constant_value(self):
        dist = FloatConstantDistribution(value=3.14)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == 3.14
        assert dist.sample(rng) == 3.14


class TestFloatUniformDistribution:
    def test_sample_within_range(self):
        dist = FloatUniformDistribution(low=1.0, high=10.0)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(1.0 <= sample <= 10.0 for sample in samples)
        assert len(set(samples)) > 50

    def test_sample_deterministic_with_seed(self):
        dist = FloatUniformDistribution(low=0.0, high=1.0)

        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)

        samples1 = [dist.sample(rng1) for _ in range(10)]
        samples2 = [dist.sample(rng2) for _ in range(10)]

        assert samples1 == samples2


class TestFloatLognormalDistribution:
    def test_sample_with_max_constraint(self):
        dist = FloatLognormalDistribution(low=1.0, high=10.0, max=15.0)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(sample <= 15.0 for sample in samples)
        assert all(sample > 0 for sample in samples)

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


class TestFloatDistributionTypes:
    def test_constant_distribution_from_float(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": 3.14})
        assert isinstance(model.dist, FloatConstantDistribution)
        assert model.dist.value == 3.14

    def test_uniform_distribution_from_tuple(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("uniform", 1.0, 10.0)})
        assert isinstance(model.dist, FloatUniformDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0

    def test_lognormal_distribution_two_args(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 10.0)})
        assert isinstance(model.dist, FloatLognormalDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0
        assert model.dist.max is None

    def test_lognormal_distribution_three_args(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 10.0, 15.0)})
        assert isinstance(model.dist, FloatLognormalDistribution)
        assert model.dist.low == 1.0
        assert model.dist.high == 10.0
        assert model.dist.max == 15.0

    def test_invalid_distribution_format(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("normal", 1.0, 10.0)})

        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("uniform", 1.0)})

    def test_integration_constant_distribution(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": 2.718})
        rng = np.random.default_rng(seed=123)

        assert model.dist.sample(rng) == 2.718

    def test_integration_uniform_distribution(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("uniform", 0.5, 1.5)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(0.5 <= sample <= 1.5 for sample in samples)

    def test_integration_lognormal_distribution(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 5.0)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(sample > 0 for sample in samples)

    def test_integration_lognormal_with_max(self):
        class TestModel(BaseModel):
            dist: FloatDistribution

        model = TestModel.model_validate({"dist": ("lognormal", 1.0, 5.0, 10.0)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(100)]
        assert all(0 < sample <= 10.0 for sample in samples)


class TestIntConstantDistribution:
    def test_sample_returns_constant_value(self):
        dist = IntConstantDistribution(value=42)
        rng = np.random.default_rng(seed=123)

        assert dist.sample(rng) == 42
        assert dist.sample(rng) == 42


class TestIntUniformDistribution:
    def test_sample_within_range(self):
        dist = IntUniformDistribution(low=1, high=10)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(1 <= sample <= 10 for sample in samples)
        assert len(set(samples)) > 1

    def test_sample_single_value_range(self):
        dist = IntUniformDistribution(low=5, high=5)
        rng = np.random.default_rng(seed=123)

        samples = [dist.sample(rng) for _ in range(10)]
        assert all(sample == 5 for sample in samples)

    def test_sample_deterministic_with_seed(self):
        dist = IntUniformDistribution(low=1, high=100)

        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)

        samples1 = [dist.sample(rng1) for _ in range(10)]
        samples2 = [dist.sample(rng2) for _ in range(10)]

        assert samples1 == samples2


class TestIntDistributionTypes:
    def test_constant_distribution_from_int(self):
        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": 42})
        assert isinstance(model.dist, IntConstantDistribution)
        assert model.dist.value == 42

    def test_uniform_distribution_from_tuple(self):
        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": ("uniform", 1, 10)})
        assert isinstance(model.dist, IntUniformDistribution)
        assert model.dist.low == 1
        assert model.dist.high == 10

    def test_invalid_uniform_tuple_format(self):
        class TestModel(BaseModel):
            dist: IntDistribution

        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("uniform", 1)})

        with pytest.raises(TypeError):
            TestModel.model_validate({"dist": ("normal", 1, 10)})

    def test_integration_constant_distribution(self):
        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": 7})
        rng = np.random.default_rng(seed=123)

        assert model.dist.sample(rng) == 7

    def test_integration_uniform_distribution(self):
        class TestModel(BaseModel):
            dist: IntDistribution

        model = TestModel.model_validate({"dist": ("uniform", 5, 15)})
        rng = np.random.default_rng(seed=123)

        samples = [model.dist.sample(rng) for _ in range(50)]
        assert all(5 <= sample <= 15 for sample in samples)
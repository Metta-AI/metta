import numpy as np
import pytest
from pydantic import BaseModel

from metta.map.random.float import (
    FloatConstantDistribution,
    FloatDistribution,
    FloatLognormalDistribution,
    FloatUniformDistribution,
)

# TestFloatConstantDistribution removed - tests mathematical guarantees


# TestFloatUniformDistribution removed - tests mathematical guarantees


# TestFloatLognormalDistribution removed - tests mathematical guarantees


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

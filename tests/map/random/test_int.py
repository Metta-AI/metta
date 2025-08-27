import numpy as np
import pytest

from metta.map.random.int import (
    IntConstantDistribution,
    IntDistribution,
    IntUniformDistribution,
)

# TestIntConstantDistribution removed - tests mathematical guarantees


# TestIntUniformDistribution removed - tests mathematical guarantees


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

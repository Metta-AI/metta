from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, GetPydanticSchema
from pydantic_core import core_schema

# Useful for scene classes - they want to take an optional seed, but sometimes we
# pass in a generator from another scene.
MaybeSeed = Union[int, np.random.Generator, None]


class BaseIntDistribution(BaseModel):
    def sample(self, rng: np.random.Generator) -> int: ...


class IntConstantDistribution(BaseIntDistribution):
    value: int

    def sample(self, _) -> int:
        return self.value


class IntUniformDistribution(BaseIntDistribution):
    low: int
    high: int

    def sample(self, rng: np.random.Generator) -> int:
        return rng.integers(self.low, self.high, endpoint=True, dtype=int)


IntConstantDistributionType = Annotated[
    int,
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda x: IntConstantDistribution(value=x), handler(tp)
        )
    ),
]

IntUniformDistributionType = Annotated[
    tuple[Literal["uniform"], int, int],
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda v: IntUniformDistribution(low=v[1], high=v[2]), handler(tp)
        )
    ),
]

IntDistribution = IntConstantDistributionType | IntUniformDistributionType

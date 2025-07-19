from typing import Annotated, Union

import numpy as np
from pydantic import BaseModel, BeforeValidator

# Useful for scene classes - they want to take an optional seed, but sometimes we
# pass in a generator from another scene.
MaybeSeed = Union[int, np.random.Generator, None]


class BaseIntDistribution(BaseModel):
    def sample(self, rng: np.random.Generator) -> int: ...


class IntConstantDistribution(BaseIntDistribution):
    value: int

    def sample(self, rng) -> int:
        return self.value


class IntUniformDistribution(BaseIntDistribution):
    low: int
    high: int

    def sample(self, rng) -> int:
        return rng.integers(self.low, self.high, endpoint=True, dtype=int)


def _to_int_distribution(v) -> BaseIntDistribution:
    """
    Accept:
      • an existing distribution object
      • an int                    → Constant
      • ("uniform", low, high)    → Uniform
    """
    if isinstance(v, BaseIntDistribution):
        return v
    if isinstance(v, int):
        return IntConstantDistribution(value=v)
    if isinstance(v, (list, tuple)) and len(v) == 3 and v[0] == "uniform":
        _, low, high = v
        return IntUniformDistribution(low=low, high=high)
    raise TypeError("value must be an int or ('uniform', low, high) tuple")


IntDistribution = Annotated[
    BaseIntDistribution,
    BeforeValidator(_to_int_distribution),
]

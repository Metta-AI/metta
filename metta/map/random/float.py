from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, GetPydanticSchema
from pydantic_core import core_schema
from scipy import stats


class FloatBaseDistribution(BaseModel):
    def sample(self, _) -> float: ...


class FloatConstantDistribution(FloatBaseDistribution):
    value: float

    def sample(self, _) -> float:
        return self.value


FloatConstantDistributionType = Annotated[
    float,
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda x: FloatConstantDistribution(value=x), handler(tp)
        )
    ),
]


class FloatUniformDistribution(FloatBaseDistribution):
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.high)


FloatUniformDistributionType = Annotated[
    tuple[Literal["uniform"], float, float],
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda x: FloatUniformDistribution(low=x[1], high=x[2]), handler(tp)
        )
    ),
]


class FloatLognormalDistribution(FloatBaseDistribution):
    low: float
    high: float
    max: float | None = None

    def _lognormal_from_90_percentile(self, rng: np.random.Generator) -> float:
        """
        Calculate the mean and standard deviation of a lognormal distribution that has a 90%
        probability of being between low and high.
        """

        if self.low >= self.high:
            raise ValueError("Low value must be less than high value")
        if self.low <= 0:
            raise ValueError("Low value must be above 0")

        # Default to 90% probability, for now
        probability = 0.9

        if probability <= 0 or probability >= 1:
            raise ValueError("Probability must be in (0, 1) interval")

        log_low = np.log(self.low)
        log_high = np.log(self.high)

        # Calculate normalized sigmas using the inverse of the normal CDF
        normalized_sigmas = stats.norm.ppf(1 - (1 - probability) / 2)
        mu = (log_low + log_high) / 2
        sigma = (log_high - log_low) / (2 * normalized_sigmas)

        # Return a sample from the lognormal distribution
        return rng.lognormal(mean=mu, sigma=sigma)

    def sample(self, rng: np.random.Generator) -> float:
        percentage = self._lognormal_from_90_percentile(rng)
        abs_max = self.max if self.max is not None else None
        if abs_max is not None:
            percentage = min(percentage, abs_max)
        return percentage


FloatLognormalDistributionTypeTwoArgs = Annotated[
    tuple[Literal["lognormal"], float, float],
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda x: FloatLognormalDistribution(low=x[1], high=x[2]), handler(tp)
        )
    ),
]

FloatLognormalDistributionTypeThreeArgs = Annotated[
    tuple[Literal["lognormal"], float, float, float],
    GetPydanticSchema(
        lambda tp, handler: core_schema.no_info_after_validator_function(
            lambda x: FloatLognormalDistribution(low=x[1], high=x[2], max=x[3]), handler(tp)
        )
    ),
]

FloatDistribution = (
    # `float`: just return the value
    FloatConstantDistributionType
    # `["uniform", low: float, high: float]`: any float in the range
    | FloatUniformDistributionType
    # `["lognormal", p5: float, p95: float]`: any float in the lognormal distribution
    # with 90% probability of being between p5 and p95
    | FloatLognormalDistributionTypeTwoArgs
    # `["lognormal", p5: float, p95: float, max: float]`: any float in the lognormal distribution
    # with 90% probability of being between p5 and p95, and max (absolute limit) is optional
    | FloatLognormalDistributionTypeThreeArgs
    # Direct distribution instances
    | FloatConstantDistribution
    | FloatUniformDistribution
    | FloatLognormalDistribution
)

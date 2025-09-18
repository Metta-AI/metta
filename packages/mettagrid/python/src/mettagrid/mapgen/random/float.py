from typing import Annotated

import numpy as np
from pydantic import BaseModel, BeforeValidator
from scipy import stats


class BaseFloatDistribution(BaseModel):
    def sample(self, rng: np.random.Generator) -> float: ...


class FloatConstantDistribution(BaseFloatDistribution):
    value: float

    def sample(self, rng) -> float:
        return self.value


class FloatUniformDistribution(BaseFloatDistribution):
    low: float
    high: float

    def sample(self, rng) -> float:
        return rng.uniform(self.low, self.high)


class FloatLognormalDistribution(BaseFloatDistribution):
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

        probability = 0.9

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


def _to_float_distribution(v) -> BaseFloatDistribution:
    """
    Accept:
    - an existing distribution object
    - a float                     -> Constant
    - ("uniform", low, high)      -> Uniform
    - ("lognormal", p5, p95)      -> Lognormal with 90% probability of being between p5 and p95
    - ("lognormal", p5, p95, max) -> Lognormal + absolute limit
                                     and max (absolute limit) is optional
    """
    if isinstance(v, BaseFloatDistribution):
        return v
    if isinstance(v, float):
        return FloatConstantDistribution(value=v)
    if isinstance(v, (list, tuple)) and len(v) == 3 and v[0] == "uniform":
        _, low, high = v
        return FloatUniformDistribution(low=low, high=high)
    if isinstance(v, (list, tuple)) and len(v) == 3 and v[0] == "lognormal":
        _, low, high = v
        return FloatLognormalDistribution(low=low, high=high)
    if isinstance(v, (list, tuple)) and len(v) == 4 and v[0] == "lognormal":
        _, low, high, max = v
        return FloatLognormalDistribution(low=low, high=high, max=max)
    raise TypeError(
        "value must be one of: "
        "- a float,\n"
        "- a tuple of the form (uniform, low, high),\n"
        "- a tuple of the form (lognormal, p5, p95),\n"
        "- a tuple of the form (lognormal, p5, p95, max)"
    )


FloatDistribution = Annotated[
    BaseFloatDistribution,
    BeforeValidator(_to_float_distribution),
]

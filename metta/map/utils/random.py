from typing import Literal, Union

import numpy as np
from omegaconf import ListConfig
from scipy import stats

# Useful for scene classes - they want to take an optional seed, but sometimes we
# pass in a generator from another scene.
MaybeSeed = Union[int, np.random.Generator, None]


def lognormal_from_90_percentile(low: float, high: float, rng: np.random.Generator) -> float:
    """
    Calculate the mean and standard deviation of a lognormal distribution that has a 90%
    probability of being between low and high.
    """

    if low >= high:
        raise ValueError("Low value must be less than high value")
    if low <= 0:
        raise ValueError("Low value must be above 0")

    # Default to 90% probability, for now
    probability = 0.9

    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be in (0, 1) interval")

    log_low = np.log(low)
    log_high = np.log(high)

    # Calculate normalized sigmas using the inverse of the normal CDF
    normalized_sigmas = stats.norm.ppf(1 - (1 - probability) / 2)
    mu = (log_low + log_high) / 2
    sigma = (log_high - log_low) / (2 * normalized_sigmas)

    # Return a sample from the lognormal distribution
    return rng.lognormal(mean=mu, sigma=sigma)


FloatDistribution = Union[
    float,
    tuple[Literal["uniform"], float, float],
    tuple[Literal["lognormal"], float, float, float],
]


def sample_float_distribution(cfg: FloatDistribution, rng: np.random.Generator) -> float:
    """
    Valid config values:
    - `float`: just return the value
    - `["uniform", low: float, high: float]`: any float in the range
    - `["lognormal", p5: float, p95: float, max: float]`: any float in the range, max (absolute limit) is optional
    """
    if isinstance(cfg, float):
        return cfg
    elif isinstance(cfg, tuple) or isinstance(cfg, ListConfig):
        (dist_type, *args) = cfg
        if dist_type == "uniform":
            assert len(args) == 2, "Uniform distribution requires [low, high]"
            return rng.uniform(args[0], args[1])
        elif dist_type == "lognormal":
            assert len(args) == 2 or len(args) == 3, "Lognormal distribution requires [mu, sigma] or [mu, sigma, max]"
            percentage = lognormal_from_90_percentile(args[0], args[1], rng)
            abs_max = args[2] if len(args) == 3 else None
            if abs_max is not None:
                percentage = min(percentage, abs_max)
            return percentage
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    else:
        raise ValueError(f"Invalid distribution: {cfg}")


IntDistribution = Union[int, tuple[Literal["uniform"], int, int]]


def sample_int_distribution(cfg: IntDistribution, rng: np.random.Generator) -> int:
    """
    Valid config values:
    - `int`: just return the value
    - `["uniform", low: int, high: int]`: any integer in the range, high is inclusive
    """
    if isinstance(cfg, int):
        return cfg
    elif isinstance(cfg, tuple) or isinstance(cfg, ListConfig):
        (dist_type, *args) = cfg
        if dist_type == "uniform":
            assert len(args) == 2, "Uniform int distribution requires [low, high]"
            return rng.integers(args[0], args[1], endpoint=True, dtype=int)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    else:
        raise ValueError(f"Invalid distribution: {cfg}")

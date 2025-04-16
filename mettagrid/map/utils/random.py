from typing import Union

import numpy as np
from omegaconf import DictConfig
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


def sample_distribution(cfg: DictConfig, rng: np.random.Generator) -> float:
    if cfg.distribution_type == "uniform":
        return rng.uniform(cfg.min, cfg.max)
    elif cfg.distribution_type == "lognormal":
        percentage = lognormal_from_90_percentile(cfg.p5, cfg.p95, rng)
        abs_max = cfg.get("max", None)
        if abs_max is not None:
            percentage = min(percentage, abs_max)
        return percentage
    else:
        raise ValueError(f"Unknown distribution type: {cfg.distribution_type}")



import numpy as np
from omegaconf import DictConfig, ListConfig


def sample_config(value, sampling: float):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, DictConfig):
        return {
            key: sample_config(value, sampling)
            for key, value in value.items()
        }
    if isinstance(value, ListConfig):
        if len(value) == 0:
            return value
        if isinstance(value[0], (int, float)):
            assert len(value) == 3, f"Expected (min, max, mean), but found {value}"
            
            min, mean, variance = value
            if sampling == 0:
                return mean
            else:
                # Sample from normal distribution with mean=center and std=sampling
                val = max(min, np.random.normal(loc=mean, scale=variance))

                return int(round(val)) if isinstance(value[0], int) else val
        return value
    return value

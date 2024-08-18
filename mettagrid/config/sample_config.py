

import numpy as np
from omegaconf import DictConfig, ListConfig


def sample_config(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, DictConfig):
        return {
            key: sample_config(value)
            for key, value in value.items()
        }
    if isinstance(value, ListConfig):
        if len(value) == 0:
            return value
        if isinstance(value[0], int):
            assert len(value) == 2, f"Found a list with length != 2 {value}"
            return np.random.randint(value[0], value[1])
        if isinstance(value[0], float):
            assert len(value) == 2, f"Found a list with length != 2 {value}"
            return np.random.uniform(value[0], value[1])
        return value
    return value

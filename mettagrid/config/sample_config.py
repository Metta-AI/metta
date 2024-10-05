

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
            min_val, max_val, center = value
            if sampling == 0:
                return center
            else:
                range_width = max_val - min_val
                scaled_stdev = sampling * range_width * 6
                val = np.random.normal(center, scaled_stdev)
                val = np.clip(val, min_val, max_val)
                return int(round(val)) if isinstance(value[0], int) else val
        return value
    return value

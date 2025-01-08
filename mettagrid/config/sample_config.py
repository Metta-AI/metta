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
                # Calculate the available range on both sides of the center
                left_range = center - min_val
                right_range = max_val - center

                # Scale the ranges based on the sampling parameter
                scaled_left = min(left_range, sampling * left_range)
                scaled_right = min(right_range, sampling * right_range)

                # Generate a random value within the scaled range
                val = np.random.uniform(center - scaled_left, center + scaled_right)

                # Clip to ensure we stay within [min_val, max_val]
                val = np.clip(val, min_val, max_val)

                # Return integer if the original values were integers
                return int(round(val)) if isinstance(value[0], int) else val
        return value
    return value
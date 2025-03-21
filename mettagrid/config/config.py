import os
import random
import signal

import numpy as np
import torch
from omegaconf import OmegaConf
from rich import traceback
import warnings

warnings.warn("This config.py file is deprecated", DeprecationWarning)

def uniform(min_val, max_val, center, *, _root_):
    sampling = _root_["sampling"]
    if sampling == 0:
        return center

    center = (max_val + min_val) // 2
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
    return int(round(val)) if isinstance(center, int) else val

def choose(*args):
    return random.choice(args)

def div(a, b):
    return a // b

def sub(a, b):
    return a - b

def make_odd(a):
    return max(3, a // 2 * 2 + 1)


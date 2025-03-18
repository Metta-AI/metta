import os
import random
import signal

import numpy as np
import torch
from omegaconf import OmegaConf
from rich import traceback
import warnings

warnings.warn("This config.py file is deprecated", DeprecationWarning)

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def uniform(min_val, max_val, center, *, _root_):
    sampling = _root_.get("sampling", 0)
    if sampling == 0:
        return center
    else:

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

def setup_omega_conf():
    OmegaConf.register_new_resolver("div", div, replace=True)
    OmegaConf.register_new_resolver("uniform", uniform, replace=True)
    OmegaConf.register_new_resolver("sub", sub, replace=True)
    OmegaConf.register_new_resolver("make_odd", make_odd, replace=True)
    OmegaConf.register_new_resolver("choose", choose, replace=True)

def setup_metta_environment(cfg):
    # Set environment variables to run without display
    os.environ['GLFW_PLATFORM'] = 'osmesa'  # Use OSMesa as the GLFW backend
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    os.environ['DISPLAY'] = ''

    # Suppress deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pygame.pkgdata')

    setup_omega_conf()
    # print(OmegaConf.to_yaml(cfg))
    traceback.install(show_locals=False)
    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import logging
import os
import random
import signal
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf
from rich import traceback

logger = logging.getLogger("runtime_configuration")


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def setup_mettagrid_environment(cfg):
    # Import mettagrid_env to ensure OmegaConf resolvers are registered before Hydra loads
    import mettagrid.mettagrid_env  # noqa: F401

    # Set environment variables to run without display
    os.environ["GLFW_PLATFORM"] = "osmesa"  # Use OSMesa as the GLFW backend
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["DISPLAY"] = ""

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")

    if cfg.dist_cfg_path is not None:
        dist_cfg = OmegaConf.load(cfg.dist_cfg_path)
        cfg.run = dist_cfg.run
        cfg.wandb.run_id = dist_cfg.wandb_run_id

    # print(OmegaConf.to_yaml(cfg))
    traceback.install(show_locals=False)
    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

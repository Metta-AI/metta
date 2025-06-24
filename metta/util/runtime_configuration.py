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


def seed_everything(seed, torch_deterministic, rank: int = 0):
    # Despite these efforts, we still don't get deterministic behavior. But presumably
    # this is better than nothing.
    # https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility

    # Add rank offset to base seed for distributed training to ensure different
    # processes generate uncorrelated random sequences
    if seed is not None:
        rank_specific_seed = seed + rank
    else:
        rank_specific_seed = rank

    random.seed(rank_specific_seed)
    np.random.seed(rank_specific_seed)
    if seed is not None:
        torch.manual_seed(rank_specific_seed)
        torch.cuda.manual_seed_all(rank_specific_seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic
    torch.use_deterministic_algorithms(torch_deterministic)

    if torch_deterministic:
        # Set CuBLAS workspace config for deterministic behavior on CUDA >= 10.2
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_mettagrid_environment(cfg):
    # Import mettagrid_env to ensure OmegaConf resolvers are registered before Hydra loads
    import metta.mettagrid.mettagrid_env  # noqa: F401

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

    # Get rank for distributed training seeding
    rank = int(os.environ.get("RANK", 0))
    seed_everything(cfg.seed, cfg.torch_deterministic, rank)

    os.makedirs(cfg.run_dir, exist_ok=True)
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

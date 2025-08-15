import logging
import os
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from metta.rl.system_config import SystemConfig

logger = logging.getLogger(__name__)


def init_mettagrid_system_environment(cfg: DictConfig, system_cfg: SystemConfig | None = None) -> None:
    """
    Configure the runtime environment for MettaGrid simulations.
    Initializes CUDA, sets thread counts, and handles reproducibility settings.

    Parameters:
    -----------
    cfg : DictConfig
        Configuration containing torch_deterministic flag and other runtime settings
    system_cfg : SystemConfig | None
        Environment configuration. If not provided, will be extracted from cfg.
    """

    # Validate device configuration
    device = cfg.get("device", "cpu")

    # Check if CUDA is requested but not available
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Overriding device to 'cpu'.")
            cfg.device = "cpu"

        # If device is cuda:X, check that the specific device exists
        if ":" in device:
            device_id = device.split(":")[1]
            try:
                device_id = int(device_id)
                if device_id >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"Device '{device}' was requested but only {torch.cuda.device_count()} "
                        f"CUDA devices are available (0-{torch.cuda.device_count() - 1})."
                    )
            except ValueError:
                raise ValueError(f"Invalid device ID in '{device}'. Device ID must be an integer.") from None

    # Validate device format
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(
            f"Invalid device '{device}'. Device must be 'cpu' or start with 'cuda' (e.g., 'cuda', 'cuda:0')."
        )

    OmegaConf.set_struct(cfg, False)
    cfg.device = device
    OmegaConf.set_struct(cfg, True)

    # Set CUDA launch blocking for better error messages in development
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

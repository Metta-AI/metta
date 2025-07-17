"""Decorators for Metta scripts."""

import functools
import inspect
import logging
import multiprocessing
import os
import random
import signal
import sys
import warnings
from types import FrameType
from typing import Callable

import hydra
import numpy as np
import rich.traceback
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.fs import get_repo_root
from metta.common.util.logging_helpers import setup_mettagrid_logger

logger = logging.getLogger("metta_script")


def metta_script(main: Callable[[DictConfig], int | None], config_name: str) -> None:
    """
    Wrapper for Metta script entry points that performs environment setup and
    configuration before calling the `main` function.

    Example usage:
    ```python
    from metta.util.metta_script import metta_script

    def main(cfg: DictConfig):
        ...

    # call main() with the config from configs/my_job.yaml
    metta_script(main, "my_job")
    ```

    Calling this function will do nothing if the script is loaded as a module.

    This wrapper:
    1. Configures Hydra to load the `config_name` config and pass it to the `main` function
    2. Sets up logging to both stdout and run_dir/logs/
    3. Calls setup_mettagrid_environment() to:
       - Create required directories (including run_dir)
       - Configure CUDA settings
       - Set up environment variables
       - Initialize random seeds
       - Register OmegaConf resolvers
    4. Performs device validation and sets the device to "cpu" if CUDA is not available
    """

    # If not running as a script, there's nothing to do.
    caller_frame: FrameType = inspect.stack()[1].frame
    caller_globals = caller_frame.f_globals
    if caller_globals.get("__name__") != "__main__":
        return

    script_path = caller_globals["__file__"]

    # Wrapped main function that we want to run.
    # This code runs after the Hydra was configured. Depending on CLI args such as `--help`, it may not run at all.
    def extended_main(cfg: ListConfig | DictConfig) -> None:
        if not isinstance(cfg, DictConfig):
            raise ValueError("Metta scripts must be run with a DictConfig")

        # Set up console logging first
        setup_mettagrid_logger()

        # Then add file logging (after console handlers are set up)
        run_dir = cfg.get("run_dir")
        if run_dir:
            setup_file_logging(run_dir)

        logger.info(f"Starting {main.__name__} from {script_path} with run_dir: {cfg.get('run_dir', 'not set')}")

        # Set up the full mettagrid environment (includes device validation)
        setup_mettagrid_environment(cfg)

        logger.info("Environment setup completed")

        # Call the original function
        result = main(cfg)
        if result is not None:
            sys.exit(result)

    # Hydra analyzes the wrapped function, and the function must come from the
    # `__main__` name for hydra to work correctly.
    # So we have to pretend that we wrap the original function from the script,
    # not the `extended_main()` function defined above.
    functools.update_wrapper(extended_main, main)

    # Hydra needs the config path to be relative to the original script.
    script_dir = os.path.abspath(os.path.dirname(script_path))
    abs_config_path = str(get_repo_root() / "configs")
    relative_config_path = os.path.relpath(abs_config_path, script_dir)

    # Calling `hydra.main` as a function instead of a decorator, because `extended_main` function
    # needs to be patched with `functools.update_wrapper` first.
    configured_main = hydra.main(config_path=relative_config_path, config_name=config_name, version_base=None)(
        extended_main
    )

    configured_main()


def setup_file_logging(run_dir: str) -> None:
    """Set up file logging in addition to stdout logging."""
    # Create logs directory
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    node_index = os.environ.get("RANK", "0")
    if node_index == "0":
        log_file = "script.log"
    else:
        log_file = f"script_{node_index}.log"

    # Set up file handler for the root logger
    log_file = os.path.join(logs_dir, log_file)
    file_handler = logging.FileHandler(log_file, mode="a")

    # Use the same formatter as the existing console handler
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)  # Ensure file handler level is set

    # Add to root logger so all log messages go to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Force a flush to make sure the file is created properly
    file_handler.flush()


def is_multiprocessing_available() -> bool:
    try:
        # Test if we can create a multiprocessing context with spawn method
        # (spawn is the safest and most compatible method across platforms)
        multiprocessing.get_context("spawn")
        return True
    except Exception:
        return False


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


def setup_mettagrid_environment(cfg: DictConfig) -> None:
    """
    Configure the runtime environment for MettaGrid simulations.
    Initializes CUDA, sets thread counts, and handles reproducibility settings.

    Parameters:
    -----------
    cfg : DictConfig
        Configuration containing torch_deterministic flag and other runtime settings
    """
    OmegaConf.set_struct(cfg, False)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Overriding device to 'cpu'.")
        cfg.device = "cpu"

    if cfg.vectorization == "multiprocessing" and not is_multiprocessing_available():
        logger.warning(
            "Vectorization 'multiprocessing' was requested but multiprocessing is not "
            "available in this environment. Overriding to 'serial'."
        )
        cfg.vectorization = "serial"

    OmegaConf.set_struct(cfg, True)

    # Validate device configuration
    device = cfg.get("device", "cpu")

    # Check if CUDA is requested but not available
    if device.startswith("cuda"):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{device}' was requested but CUDA is not available. "
                    "Please either install CUDA/PyTorch with GPU support or set device: cpu in your config."
                )

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
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed but CUDA device was requested. "
                "Please install PyTorch or set device: cpu in your config."
            ) from None

    # Validate device format
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(
            f"Invalid device '{device}'. Device must be 'cpu' or start with 'cuda' (e.g., 'cuda', 'cuda:0')."
        )

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

    # print(OmegaConf.to_yaml(cfg))
    rich.traceback.install(show_locals=False)

    # Get rank for distributed training seeding
    rank = int(os.environ.get("RANK", 0))
    seed_everything(cfg.seed, cfg.torch_deterministic, rank)

    os.makedirs(cfg.run_dir, exist_ok=True)
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

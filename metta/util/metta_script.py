"""Decorators for Metta scripts."""

import functools
import inspect
import logging
import multiprocessing
import os
import sys
from types import FrameType
from typing import Callable

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.fs import get_repo_root
from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment


def metta_script(main: Callable[[DictConfig], int | None], config_name: str) -> None:
    """
    Wrapper for Metta script entry points that performs environment setup and configuration.

    This wrapper:
    1. Configures Hydra to load the `config_name` config
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

    # Wrapped main function that we want to run.
    # This code runs after the Hydra was configured. Depending on args such as `--help`, it may not run at all.
    def extended_main(cfg: ListConfig | DictConfig) -> None:
        if not isinstance(cfg, DictConfig):
            raise ValueError("Metta scripts must be run with a DictConfig")

        # Set up console logging first
        logger = setup_mettagrid_logger("metta_script")

        # Then add file logging (after console handlers are set up)
        setup_file_logging(cfg)

        logger.info(f"Starting {main.__name__} with run_dir: {cfg.get('run_dir', 'not set')}")

        # Patch the config to set the device to "cpu" if CUDA is not available
        set_hardware_configurations(cfg, logger)

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
    script_dir = os.path.abspath(os.path.dirname(caller_globals["__file__"]))
    abs_config_path = str(get_repo_root() / "configs")
    relative_config_path = os.path.relpath(abs_config_path, script_dir)

    decorated_main = hydra.main(config_path=relative_config_path, config_name=config_name, version_base=None)(
        extended_main
    )

    decorated_main()


def setup_file_logging(cfg: DictConfig) -> None:
    """Set up file logging in addition to stdout logging."""
    run_dir = cfg.get("run_dir")
    if not run_dir:
        return

    # Create logs directory
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up file handler for the root logger
    log_file = os.path.join(logs_dir, "script.log")
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


def set_hardware_configurations(cfg: DictConfig, logger: logging.Logger) -> None:
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


def is_multiprocessing_available() -> bool:
    try:
        # Test if we can create a multiprocessing context with spawn method
        # (spawn is the safest and most compatible method across platforms)
        multiprocessing.get_context("spawn")
        return True
    except Exception:
        return False

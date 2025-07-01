"""Decorators for Metta scripts."""

import functools
import logging
import os
from typing import Callable, TypeVar

from omegaconf import DictConfig, ListConfig

from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment

T = TypeVar("T")


def metta_script(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for Metta script entry points that sets up logging and environment.

    This decorator:
    - Sets up dual logging to both run_dir and stdout
    - Calls setup_mettagrid_environment for device validation and environment setup
    - Provides a logger instance to the decorated function
    """

    @functools.wraps(func)
    def wrapper(cfg: DictConfig | ListConfig, *args, **kwargs) -> T:
        # Setup mettagrid environment (includes device validation)
        setup_mettagrid_environment(cfg)
        
        # Setup logging to both stdout and file
        logger = setup_dual_logging(cfg)
        
        logger.info(f"Starting {func.__name__} with run_dir: {cfg.get('run_dir', 'not specified')}")

        # Call the original function with logger
        return func(cfg, *args, **kwargs)

    return wrapper


def setup_dual_logging(cfg: DictConfig | ListConfig) -> logging.Logger:
    """
    Setup logging to both stdout (via Rich handler) and file in run_dir.
    
    Returns:
        Logger instance for the script
    """
    # First setup the standard mettagrid logger (Rich handler to stdout)
    logger = setup_mettagrid_logger("metta_script")
    
    # Add file handler if run_dir is specified
    run_dir = cfg.get("run_dir")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        
        # Create file handler for the log file
        log_file_path = os.path.join(run_dir, "script.log")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        
        # Use same formatter as Rich handler but without colors
        formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        # Add file handler to root logger so all loggers use it
        root_logger = logging.getLogger()
        file_handler.setLevel(root_logger.level)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file_path}")
    
    return logger

"""Decorators for Metta scripts."""

import functools
import logging
import os
import sys
from typing import Callable, TypeVar

import torch
from omegaconf import DictConfig, ListConfig

from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment

T = TypeVar("T")


def metta_script(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for Metta script entry points that performs environment setup.

    This decorator:
    1. Sets up logging to both stdout and run_dir/logs/
    2. Calls setup_mettagrid_environment for full environment initialization
    3. Performs device validation
    """

    @functools.wraps(func)
    def wrapper(cfg: DictConfig | ListConfig, *args, **kwargs) -> T:
        # Set up console logging first
        logger = setup_mettagrid_logger("metta_script")
        
        # Then add file logging (after console handlers are set up)
        _setup_script_logging(cfg)
        
        logger.info(f"Starting {func.__name__} with run_dir: {cfg.get('run_dir', 'not set')}")

        # Set up the full mettagrid environment (includes device validation)
        setup_mettagrid_environment(cfg)
        
        logger.info("Environment setup completed")

        # Call the original function
        return func(cfg, *args, **kwargs)

    return wrapper


def _setup_script_logging(cfg: DictConfig | ListConfig) -> None:
    """Set up file logging in addition to stdout logging."""
    run_dir = cfg.get("run_dir")
    if not run_dir:
        return
    
    # Create logs directory
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up file handler for the root logger
    log_file = os.path.join(logs_dir, "script.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    
    # Use the same formatter as the existing console handler
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)  # Ensure file handler level is set
    
    # Add to root logger so all log messages go to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Force a flush to make sure the file is created properly
    file_handler.flush()

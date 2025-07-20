"""Decorators for Metta scripts."""

import functools
import logging
import multiprocessing
from contextvars import ContextVar
from typing import Callable, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf

from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment

T = TypeVar("T")

# Context variable to store the logger
_metta_logger: ContextVar["logging.Logger | None"] = ContextVar("_metta_logger", default=None)


def get_metta_logger():
    """
    Get the logger instance from the metta_script decorator.

    Returns:
        The logger instance set up by the metta_script decorator.

    Raises:
        RuntimeError: If called outside of a metta_script decorated function.
    """
    logger = _metta_logger.get()
    if logger is None:
        raise RuntimeError(
            "No metta logger available. Ensure you're running within a @metta_script decorated function."
        )
    return logger


def metta_script(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for Metta script entry points that performs full environment setup.

    This decorator:
    1. Sets up logging
    2. Sets config values related to hardware configuration
    3. Calls setup_mettagrid_environment() to:
       - Create required directories (including run_dir)
       - Configure CUDA settings
       - Set up environment variables
       - Initialize random seeds
       - Register OmegaConf resolvers

    The decorated function can access the logger using get_metta_logger().
    """

    @functools.wraps(func)
    def wrapper(cfg: DictConfig, *args, **kwargs) -> T:
        logger = setup_mettagrid_logger("metta_script")

        set_hardware_configurations(cfg, logger)

        # Call setup_mettagrid_environment first - it handles all environment setup
        # including device validation, directory creation, and seed initialization
        setup_mettagrid_environment(cfg)

        # Log that setup completed successfully
        logger.info(f"MetaGrid environment setup completed for device: {cfg.device}")

        # Set the logger in context
        token = _metta_logger.set(logger)
        try:
            # Call the original function
            return func(cfg, *args, **kwargs)
        finally:
            # Reset the context
            _metta_logger.reset(token)

    return wrapper


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

"""Decorators for Metta scripts."""

import functools
import logging
from contextvars import ContextVar
from typing import Callable, TypeVar

import torch
from omegaconf import DictConfig

from metta.common.util.logging import setup_mettagrid_logger

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
    Decorator for Metta script entry points that performs device validation.

    This decorator checks that the requested device is available and valid.
    It will raise an error if CUDA is requested but not available.

    The decorated function can access the logger using get_metta_logger().
    """

    @functools.wraps(func)
    def wrapper(cfg: DictConfig, *args, **kwargs) -> T:
        logger = setup_mettagrid_logger("metta_script")

        # Get the device from config
        device = cfg.get("device", "cpu")

        # Check if CUDA is requested but not available
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device}' was requested but CUDA is not available. "
                "Please either install CUDA/PyTorch with GPU support or set device: cpu in your config."
            )

        # Validate device format
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(
                f"Invalid device '{device}'. Device must be 'cpu' or start with 'cuda' (e.g., 'cuda', 'cuda:0')."
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

        logger.info(f"Device check passed: {device}")

        # Set the logger in context
        token = _metta_logger.set(logger)
        try:
            # Call the original function
            return func(cfg, *args, **kwargs)
        finally:
            # Reset the context
            _metta_logger.reset(token)

    return wrapper

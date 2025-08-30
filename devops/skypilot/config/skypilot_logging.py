#!/usr/bin/env python3
"""
Centralized logging module for SkyPilot configuration scripts.
Provides consistent logging with node index information.
"""

import logging

from metta.common.util.logging_helpers import init_logging
from metta.common.util.logging_helpers import log_master as _log_master


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with node index in format.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Initialize logging with rank display
    init_logging(level=logging.getLevelName(level), show_rank=True)

    # Return the skypilot logger
    logger = logging.getLogger("skypilot")
    logger.setLevel(level)

    return logger


# Create a default logger instance
skypilot_logger = setup_logger()


def log_master(message: str):
    """Log message only on master node (rank 0)."""
    _log_master(message, skypilot_logger)


def log_all(message: str):
    """Log message on all nodes."""
    skypilot_logger.info(message)


def log_error(message: str):
    """Log error message on all nodes."""
    skypilot_logger.error(message)


def log_warning(message: str):
    """Log warning message on all nodes."""
    skypilot_logger.warning(message)


def log_debug(message: str):
    """Log debug message on all nodes."""
    skypilot_logger.debug(message)

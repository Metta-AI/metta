#!/usr/bin/env python3
"""
Centralized logging module for SkyPilot configuration scripts.
Provides consistent logging with node index information.
"""

import datetime
import logging
import os
import sys


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with node index in format.

    Args:
        name: Logger name. If None, uses 'skypilot'
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))

    # Use a specific name to avoid conflicts
    logger = logging.getLogger("skypilot")

    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(level)

    # Create console handler with custom formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [{node_index}] %(message)s", datefmt="%H:%M:%S.%f")

    # Override the default time formatting to show milliseconds
    formatter.formatTime = lambda record, datefmt=None: datetime.datetime.fromtimestamp(record.created).strftime(
        "%H:%M:%S.%f"
    )[:-3]

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Create a default logger instance
skypilot_logger = setup_logger()


def log_master(message: str):
    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    if node_index == 0:
        skypilot_logger.info(message)


def log_all(message: str):
    skypilot_logger.info(message)


def log_error(message: str):
    skypilot_logger.error(message)


def log_warning(message: str):
    skypilot_logger.warning(message)


def log_debug(message: str):
    skypilot_logger.debug(message)

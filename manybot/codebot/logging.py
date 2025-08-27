"""
Logging configuration for the codebot package.

Provides centralized logging setup to be used across all modules.
"""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the codebot application.

    Args:
        verbose: If True, set logging level to DEBUG. Otherwise, use INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

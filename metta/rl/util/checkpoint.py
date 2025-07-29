"""Checkpoint-related utility functions."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def log_recent_checkpoints(run_dir: str, max_recent: int = 3) -> None:
    """Log the most recent checkpoint files in a run directory.

    This is useful for debugging to see what checkpoints are available
    when resuming training.

    Args:
        run_dir: The run directory containing checkpoints
        max_recent: Maximum number of recent checkpoints to log (default: 3)
    """
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        files = sorted(os.listdir(checkpoints_dir))
        recent_files = files[-max_recent:] if len(files) >= max_recent else files
        if recent_files:
            logger.info(f"Recent checkpoints: {', '.join(recent_files)}")
        else:
            logger.info("No checkpoints found in directory")
    else:
        logger.info("Checkpoints directory does not exist yet")

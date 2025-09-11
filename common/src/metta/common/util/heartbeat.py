import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def record_heartbeat() -> None:
    """Record a heartbeat by updating the file's modification time."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")
    if not heartbeat_file_path:
        return  # No heartbeat file configured

    try:
        Path(heartbeat_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(heartbeat_file_path).touch()
    except Exception as e:
        logger.error(f"Failed to record heartbeat at {heartbeat_file_path}: {e}")
        raise

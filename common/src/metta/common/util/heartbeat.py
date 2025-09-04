import logging
import os
import time

logger = logging.getLogger(__name__)


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the globally configured file path."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")
    if not heartbeat_file_path:
        raise RuntimeError("HEARTBEAT_FILE environment variable not set")

    # Ensure the directory for the heartbeat file exists
    os.makedirs(os.path.dirname(heartbeat_file_path), exist_ok=True)
    with open(heartbeat_file_path, "w") as f:
        f.write(str(time.time()))

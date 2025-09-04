import logging
import os
import time

logger = logging.getLogger(__name__)

# Shared IPC filename, co-located with the heartbeat signal file (must match wandb_context.py)
WANDB_IPC_FILENAME = "wandb_ipc.json"


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the globally configured file path."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")

    if heartbeat_file_path:
        try:
            # Ensure the directory for the heartbeat file exists
            os.makedirs(os.path.dirname(heartbeat_file_path), exist_ok=True)
            with open(heartbeat_file_path, "w") as f:
                f.write(str(time.time()))
        except Exception as exc:
            logger.warning("Failed to write heartbeat: %s", exc)

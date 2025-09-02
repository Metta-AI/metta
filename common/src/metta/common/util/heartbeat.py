import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def record_heartbeat() -> None:
    """Record a heartbeat by updating the file's modification time.

    This function only touches the file to update mtime, avoiding file recreation.
    """
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")

    if not heartbeat_file_path:
        return  # No heartbeat file configured

    try:
        # Use pathlib.Path.touch() - most efficient for updating mtime
        # exist_ok=True prevents errors if file exists
        Path(heartbeat_file_path).touch(exist_ok=True)

    except FileNotFoundError:
        # Parent directory doesn't exist
        logger.error(f"Heartbeat directory missing: {os.path.dirname(heartbeat_file_path)}")

        # Try to recreate the directory structure and file
        try:
            heartbeat_path = Path(heartbeat_file_path)
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.touch()
            logger.warning(f"Recreated missing heartbeat file at {heartbeat_file_path}")
        except Exception as recreate_error:
            logger.error(f"Failed to recreate heartbeat file: {recreate_error}")

    except PermissionError as e:
        logger.error(f"Permission denied updating heartbeat at {heartbeat_file_path}: {e}")

    except OSError as e:
        # Log errno for debugging filesystem-specific issues
        errno_num = getattr(e, "errno", "unknown")
        logger.error(f"OS error updating heartbeat (errno={errno_num}): {e}")

        # For S3/network filesystems, errno 116 (ESTALE) indicates stale handle
        if errno_num == 116:
            logger.error("Stale file handle - filesystem mount may be unstable")

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error recording heartbeat: {type(e).__name__}: {e}", exc_info=True)


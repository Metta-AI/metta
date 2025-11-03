#!/usr/bin/env -S uv run python3

"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import os
import subprocess

from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


def terminate_process_group(job: subprocess.Popen, timeout: int = 30) -> None:
    """Terminate a subprocess and its entire process group."""
    import signal

    if job.poll() is not None:
        # Process already terminated
        return

    try:
        # Send SIGTERM to the entire process group
        # Since we use start_new_session=True, the PID is the PGID
        os.killpg(job.pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process group {job.pid}")

        # Wait for graceful termination
        job.wait(timeout=timeout)
        logger.info(f"Process group {job.pid} terminated gracefully")

    except subprocess.TimeoutExpired:
        # Force kill if graceful termination failed
        logger.warning(f"Process group {job.pid} didn't terminate gracefully, forcing kill")
        try:
            os.killpg(job.pid, signal.SIGKILL)
            job.wait()
            logger.info(f"Process group {job.pid} killed forcefully")
        except ProcessLookupError:
            # Process already dead
            pass

    except ProcessLookupError:
        # Process already dead (can happen between poll and killpg)
        logger.info(f"Process group {job.pid} already terminated")

    except PermissionError as e:
        logger.error(f"Permission denied when trying to kill process group {job.pid}: {e}")
        # Try to at least kill the parent process
        try:
            job.kill()
            job.wait()
        except Exception:
            pass

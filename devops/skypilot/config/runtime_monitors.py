#!/usr/bin/env python3
"""
Runtime monitors for SkyPilot jobs including heartbeat and timeout monitoring.
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class JobMonitor(ABC):
    def __init__(
        self,
        name: str,
        shutdown_callback: Callable[[str], None],
        check_interval_sec: float = 30.0,
    ):
        self.name = name
        self.shutdown_callback = shutdown_callback
        self.check_interval_sec = check_interval_sec

    @abstractmethod
    def check_condition(self) -> tuple[bool, Optional[str]]:
        pass

    def run(self):
        while True:
            try:
                should_terminate, reason = self.check_condition()

                if should_terminate and reason:
                    logger.error(f"{self.name} monitor triggered: {reason}")
                    self.shutdown_callback(reason)
                    break

            except Exception as e:
                logger.warning(f"{self.name} monitor error: {e}")

            time.sleep(self.check_interval_sec)


class HeartbeatMonitor(JobMonitor):
    """Monitor for checking heartbeat file updates."""

    def __init__(
        self,
        heartbeat_timeout_sec: int,
        heartbeat_file: Path,
        shutdown_callback: Callable[[str], None],
    ):
        super().__init__(
            name="heartbeat",
            shutdown_callback=shutdown_callback,
            check_interval_sec=15.0,  # Check every 15 seconds
        )
        self.heartbeat_timeout = heartbeat_timeout_sec
        self.heartbeat_file = heartbeat_file
        self.last_heartbeat = time.time()

    def check_condition(self) -> tuple[bool, Optional[str]]:
        """Check if heartbeat has timed out."""
        if self.heartbeat_file.exists():
            try:
                stat = os.stat(self.heartbeat_file)
                self.last_heartbeat = stat.st_mtime
            except OSError:
                pass  # File might have been deleted

        elapsed = time.time() - self.last_heartbeat

        if elapsed > self.heartbeat_timeout:
            return True, "heartbeat_timeout"

        return False, None

    def run(self):
        logger.info(f"Heartbeat monitor started (timeout: {self.heartbeat_timeout}s)")
        super().run()


class TimeoutMonitor(JobMonitor):
    """Monitor for checking maximum runtime."""

    def __init__(
        self,
        max_runtime_hours: float,
        shutdown_callback: Callable[[str], None],
    ):
        super().__init__(
            name="timeout",
            shutdown_callback=shutdown_callback,
            check_interval_sec=30.0,  # Check every 30 seconds
        )
        self.max_runtime_hours = max_runtime_hours
        self.max_seconds = max_runtime_hours * 3600 if max_runtime_hours else 0
        self.start_time = time.time()
        self.accumulated_runtime = 0.0

        # Get accumulated runtime file path from environment
        job_metadata_dir = Path(os.environ.get("JOB_METADATA_DIR", "/tmp/metta"))
        self.accumulated_runtime_file = job_metadata_dir / "accumulated_runtime"

        # Load accumulated runtime if file exists, otherwise create it
        if self.accumulated_runtime_file.exists():
            try:
                self.accumulated_runtime = float(self.accumulated_runtime_file.read_text())
                logger.info(f"Loaded accumulated runtime: {self.accumulated_runtime:.0f}s")
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to load accumulated runtime: {e}")
                self.accumulated_runtime = 0.0
        else:
            # Create the file with initial value
            try:
                self.accumulated_runtime_file.parent.mkdir(parents=True, exist_ok=True)
                self.accumulated_runtime_file.write_text("0.0")
                logger.info("Created accumulated runtime file with initial value: 0.0s")
            except Exception as e:
                logger.error(f"Failed to create accumulated runtime file: {e}")

    def get_current_runtime(self) -> float:
        """Get the runtime for the current session."""
        return time.time() - self.start_time

    def get_total_runtime(self) -> float:
        """Get the total runtime including accumulated time."""
        return self.accumulated_runtime + self.get_current_runtime()

    def save_accumulated_runtime(self):
        """Save the current total runtime to file."""
        try:
            total_runtime = self.get_total_runtime()
            self.accumulated_runtime_file.parent.mkdir(parents=True, exist_ok=True)
            self.accumulated_runtime_file.write_text(str(total_runtime))
            logger.debug(f"Updated accumulated runtime: {total_runtime:.0f}s")
        except Exception as e:
            logger.error(f"Failed to save accumulated runtime: {e}")

    def check_condition(self) -> tuple[bool, Optional[str]]:
        """Check if max runtime has been exceeded."""
        total_runtime = self.get_total_runtime()

        # Save accumulated runtime on every check
        self.save_accumulated_runtime()

        if total_runtime >= self.max_seconds:
            return True, "max_runtime_reached"

        return False, None

    def run(self):
        remaining = self.max_seconds - self.accumulated_runtime
        logger.info(f"Timeout monitor started (remaining: {remaining:.0f}s)")
        super().run()


def start_monitors(
    shutdown_callback: Callable[[str], None],
) -> None:
    """
    Start runtime monitors based on environment configuration.

    Reads configuration from environment variables:
    - HEARTBEAT_TIMEOUT: Timeout in seconds for heartbeat monitoring
    - MAX_RUNTIME_HOURS: Maximum runtime in hours
    - SKYPILOT_NODE_RANK: Node rank (0 = master)
    - JOB_METADATA_DIR: Directory for metadata files

    Args:
        shutdown_callback: Callback function to trigger shutdown with reason
    """
    # Read configuration from environment
    heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
    max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
    rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    is_master = rank == 0
    job_metadata_dir = Path(os.environ.get("JOB_METADATA_DIR", "/tmp/metta"))

    # Define paths
    heartbeat_file = job_metadata_dir / "heartbeat"

    # Start heartbeat monitor if configured
    if heartbeat_timeout:
        heartbeat_monitor = HeartbeatMonitor(
            heartbeat_timeout_sec=heartbeat_timeout,
            heartbeat_file=heartbeat_file,
            shutdown_callback=shutdown_callback,
        )
        threading.Thread(target=heartbeat_monitor.run, name="heartbeat_monitor", daemon=True).start()

    # Start timeout monitor if configured and on master
    if max_runtime_hours and is_master:
        timeout_monitor = TimeoutMonitor(
            max_runtime_hours=max_runtime_hours,
            shutdown_callback=shutdown_callback,
        )

        threading.Thread(target=timeout_monitor.run, name="timeout_monitor", daemon=True).start()

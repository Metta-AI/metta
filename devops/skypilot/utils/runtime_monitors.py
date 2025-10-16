#!/usr/bin/env python3
"""
Runtime monitors for SkyPilot jobs including heartbeat and timeout monitoring.
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class Monitor(ABC):
    """Abstract base class for runtime monitors."""

    @property
    def name(self) -> str:
        """Get the monitor name from the class name."""
        return self.__class__.__name__

    @abstractmethod
    def check_condition(self) -> str | None:
        """Check if the monitor condition is triggered.

        Returns the termination reason if triggered, None otherwise.
        """
        pass


class HeartbeatMonitor(Monitor):
    """Monitor for checking heartbeat file updates."""

    def __init__(
        self,
        rank: int,
        heartbeat_timeout_sec: int,
    ):
        self.heartbeat_timeout = heartbeat_timeout_sec
        self.rank = rank
        self.is_master = rank == 0

        # Get heartbeat file path from environment
        heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")
        if not heartbeat_file_path:
            raise ValueError("HEARTBEAT_FILE environment variable must be set")

        self.heartbeat_file = Path(heartbeat_file_path)

        try:
            self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat_file.touch()  # Updates mtime on restart
            logger.info(f"Updated heartbeat file at {self.heartbeat_file}")
        except Exception as e:
            logger.error(f"Failed to update heartbeat file: {e}")

    def check_condition(self) -> str | None:
        """Check if heartbeat has timed out.

        Returns the termination reason if triggered, None otherwise.
        """
        try:
            stat = self.heartbeat_file.stat()
            last_heartbeat_time = stat.st_mtime
            current_time = time.time()
            elapsed = current_time - last_heartbeat_time
            if elapsed > self.heartbeat_timeout:
                return "heartbeat_timeout"

            return None

        except FileNotFoundError:
            logger.error(f"Heartbeat file not found: {self.heartbeat_file}")
            if not self.heartbeat_file.parent.exists():
                logger.error(f"Parent directory also missing: {self.heartbeat_file.parent}")
                return "heartbeat_directory_missing"

            return "heartbeat_file_missing"

        except PermissionError as e:
            logger.error(f"Permission denied accessing heartbeat file: {e}")
            return "heartbeat_permission_denied"

        except OSError as e:
            errno_num = getattr(e, "errno", "unknown")
            logger.error(f"OS error accessing heartbeat file (errno={errno_num}): {e}")
            return f"heartbeat_os_error_{errno_num}"

        except Exception as e:
            logger.error(f"Unexpected error checking heartbeat: {type(e).__name__}: {e}")
            return f"heartbeat_unexpected_error_{type(e).__name__}"


class TimeoutMonitor(Monitor):
    """Monitor for checking maximum runtime."""

    def __init__(
        self,
        rank: int,
        max_runtime_hours: float,
    ):
        self.max_runtime_hours = max_runtime_hours
        self.max_seconds = max_runtime_hours * 3600 if max_runtime_hours else 0
        self.start_time = time.time()
        self.accumulated_runtime_sec: int = 0

        self.rank = rank
        self.is_master = rank == 0

        # Get accumulated runtime file path from environment
        accumulated_runtime_file_path = os.environ.get("ACCUMULATED_RUNTIME_FILE")
        if not accumulated_runtime_file_path:
            raise ValueError("ACCUMULATED_RUNTIME_FILE environment variable must be set")
        self.accumulated_runtime_file = Path(accumulated_runtime_file_path)

        # Load accumulated runtime if file exists, otherwise create it
        if self.accumulated_runtime_file.exists():
            try:
                self.accumulated_runtime_sec = int(self.accumulated_runtime_file.read_text())
                logger.info(f"Loaded accumulated runtime: {self.accumulated_runtime_sec:.0f}s")
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to load accumulated runtime: {e}")
                self.accumulated_runtime_sec = 0
        else:
            # Only master node creates the file
            if self.is_master:
                try:
                    self.accumulated_runtime_file.parent.mkdir(parents=True, exist_ok=True)
                    self.accumulated_runtime_file.write_text("0")
                    logger.info("Created accumulated runtime file with initial value: 0.0s")
                except Exception as e:
                    logger.error(f"Failed to create accumulated runtime file: {e}")
            else:
                logger.info("Accumulated runtime file not found (non-master node)")
                self.accumulated_runtime_sec = 0

    def get_current_runtime(self) -> int:
        """Get the runtime for the current session."""
        return int(time.time() - self.start_time)

    def get_total_runtime(self) -> int:
        """Get the total runtime including accumulated time."""
        return self.accumulated_runtime_sec + self.get_current_runtime()

    def save_accumulated_runtime(self):
        """Save the current total runtime to file (master node only)."""
        if not self.is_master:
            return

        try:
            total_runtime = self.get_total_runtime()
            self.accumulated_runtime_file.parent.mkdir(parents=True, exist_ok=True)
            self.accumulated_runtime_file.write_text(str(total_runtime))
            logger.debug(f"Updated accumulated runtime: {total_runtime:.0f}s")
        except Exception as e:
            logger.error(f"Failed to save accumulated runtime: {e}")

    def check_condition(self) -> str | None:
        """Check if max runtime has been exceeded.

        Returns the termination reason if triggered, None otherwise.
        """
        if self.get_total_runtime() > self.max_seconds:
            return "max_runtime_reached"

        # Save accumulated runtime on every check
        self.save_accumulated_runtime()

        return None


class ForceRestartTestMonitor(Monitor):
    """Monitor that simulates node failure for testing job recovery."""

    def __init__(
        self,
        restart_time_hours: float,
    ):
        self.start_time = time.time()
        self.failure_delay_sec = int(restart_time_hours * 3600)

    def check_condition(self) -> str | None:
        """Check if it's time to simulate a failure.

        Returns the termination reason if triggered, None otherwise.
        """
        elapsed = time.time() - self.start_time

        if elapsed >= self.failure_delay_sec:
            return "force_restart_test"

        return None

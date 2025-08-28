#!/usr/bin/env python3
"""
Runtime monitors for SkyPilot jobs including heartbeat and timeout monitoring.
"""

import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from skypilot_logging import setup_logger, log_all, log_error, log_warning, log_debug

# Initialize logger for this module
logger = setup_logger()


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
                    log_error(f"{self.name} monitor triggered: {reason}")
                    self.shutdown_callback(reason)
                    break

            except Exception as e:
                log_warning(f"{self.name} monitor error: {e}")

            time.sleep(self.check_interval_sec)


class HeartbeatMonitor(JobMonitor):
    """Monitor for checking heartbeat file updates."""

    def __init__(
        self,
        rank: int,
        heartbeat_timeout_sec: int,
        shutdown_callback: Callable[[str], None],
    ):
        super().__init__(
            name="heartbeat",
            shutdown_callback=shutdown_callback,
            check_interval_sec=15.0,
        )
        self.heartbeat_timeout = heartbeat_timeout_sec
        self.rank = rank
        self.is_master = rank == 0

        # Get heartbeat file path from environment
        heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")
        if not heartbeat_file_path:
            raise ValueError("HEARTBEAT_FILE environment variable must be set")

        self.heartbeat_file = Path(heartbeat_file_path)

        # Only master node manages the heartbeat file
        if self.is_master:
            try:
                self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
                self.heartbeat_file.touch()  # Updates mtime on restart
                log_all(f"Updated heartbeat file at {self.heartbeat_file}")
            except Exception as e:
                log_error(f"Failed to update heartbeat file: {e}")
        else:
            time.sleep(5) # give the master node time to update the file


    def check_condition(self) -> tuple[bool, Optional[str]]:
        """Check if heartbeat has timed out."""
        try:
            stat = os.stat(self.heartbeat_file)
            last_heartbeat_time = stat.st_mtime
            elapsed = time.time() - last_heartbeat_time

            if elapsed > self.heartbeat_timeout:
                return True, "heartbeat_timeout"

        except (OSError, FileNotFoundError):
            # If heartbeat file doesn't exist, that's a problem - trigger timeout
            log_warning(f"Heartbeat file not found: {self.heartbeat_file}")
            return True, "heartbeat_file_missing"

        return False, None

    def run(self):
        log_all(f"Heartbeat monitor started on node {self.rank} (timeout: {self.heartbeat_timeout}s)")
        super().run()


class TimeoutMonitor(JobMonitor):
    """Monitor for checking maximum runtime."""

    def __init__(
        self,
        rank: int,
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
        self.accumulated_runtime_sec : int = 0

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
                log_all(f"Loaded accumulated runtime: {self.accumulated_runtime_sec:.0f}s")
            except (ValueError, IOError) as e:
                log_warning(f"Failed to load accumulated runtime: {e}")
                self.accumulated_runtime_sec = 0
        else:
            # Only master node creates the file
            if self.is_master:
                try:
                    self.accumulated_runtime_file.parent.mkdir(parents=True, exist_ok=True)
                    self.accumulated_runtime_file.write_text("0")
                    log_all("Created accumulated runtime file with initial value: 0.0s")
                except Exception as e:
                    log_error(f"Failed to create accumulated runtime file: {e}")
            else:
                log_all("Accumulated runtime file not found (non-master node)")
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
            log_debug(f"Updated accumulated runtime: {total_runtime:.0f}s")
        except Exception as e:
            log_error(f"Failed to save accumulated runtime: {e}")

    def check_condition(self) -> tuple[bool, Optional[str]]:
        """Check if max runtime has been exceeded."""
        total_runtime = self.get_total_runtime()

        # Save accumulated runtime on every check
        self.save_accumulated_runtime()

        if total_runtime >= self.max_seconds:
            return True, "max_runtime_reached"

        return False, None

    def run(self):
        remaining = self.max_seconds - self.accumulated_runtime_sec
        log_all(f"Timeout monitor started on node {self.rank} (exit in {remaining:.0f}s)")
        super().run()


class ForceRestartTestMonitor(JobMonitor):
    """Monitor that simulates node failure for testing job recovery."""

    def __init__(
        self,
        rank: int,
        restart_time_hours: float,
        shutdown_callback: Callable[[str], None],
    ):
        super().__init__(
            name="force_restart_test",
            shutdown_callback=shutdown_callback,
            check_interval_sec=10.0,  # Check every 10 seconds
        )
        self.start_time = time.time()
        self.failure_delay_sec = int(restart_time_hours * 3600)
        self.rank = rank

    def check_condition(self) -> tuple[bool, Optional[str]]:
        """Check if it's time to simulate a failure."""

        elapsed = time.time() - self.start_time

        if elapsed >= self.failure_delay_sec:
            return True, "force_restart_test"

        return False, None

    def run(self):
        log_all(f"Test failure monitor started on node {self.rank} (will fail in {self.failure_delay_sec}s)")
        super().run()


def start_monitors(
    shutdown_callback: Callable[[str], None],
) -> None:
    """
    Start runtime monitors based on environment configuration.

    Reads configuration from environment variables:
    - HEARTBEAT_TIMEOUT: Timeout in seconds for heartbeat monitoring
    - HEARTBEAT_FILE: Path to the heartbeat file (required if HEARTBEAT_TIMEOUT is set)
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

    if heartbeat_timeout:
        heartbeat_monitor = HeartbeatMonitor(
            rank,
            heartbeat_timeout_sec=heartbeat_timeout,
            shutdown_callback=shutdown_callback,
        )
        threading.Thread(target=heartbeat_monitor.run, name="heartbeat_monitor", daemon=True).start()

    if max_runtime_hours:
        timeout_monitor = TimeoutMonitor(
            rank,
            max_runtime_hours=max_runtime_hours,
            shutdown_callback=shutdown_callback,
        )
        threading.Thread(target=timeout_monitor.run, name="timeout_monitor", daemon=True).start()

        if is_master:
            force_restart_test_monitor = ForceRestartTestMonitor(
                rank,
                restart_time_hours=max_runtime_hours / 2.0,
                shutdown_callback=shutdown_callback,
            )
            threading.Thread(
                target=force_restart_test_monitor.run, name="force_restart_test_monitor", daemon=True
            ).start()

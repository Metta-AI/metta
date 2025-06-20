#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil>=6.0.0",
# ]
# ///
"""
Benchmarking utilities for GitHub Actions workflows.
Provides memory and time monitoring for subprocess execution.
"""

import os
import subprocess
import sys
import threading
import time
from typing import Tuple, Union

import psutil


class ProcessMonitor:
    """Monitor process memory usage and execution time."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.peak_memory_mb = 0
        self.start_time = time.time()
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._memory_samples = []

    def start(self) -> None:
        """Start monitoring memory usage in a separate thread."""

        def monitor():
            try:
                proc = psutil.Process(self.process.pid)
                while not self._stop_monitoring.is_set() and self.process.poll() is None:
                    try:
                        # Get memory info for process and all children
                        memory_mb = proc.memory_info().rss / (1024 * 1024)
                        for child in proc.children(recursive=True):
                            try:
                                memory_mb += child.memory_info().rss / (1024 * 1024)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass

                        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                        self._memory_samples.append((time.time() - self.start_time, memory_mb))

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    time.sleep(0.1)
            except Exception as e:
                print(f"Memory monitoring error: {e}", file=sys.stderr)

        self._monitor_thread = threading.Thread(target=monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop(self) -> Tuple[float, float]:
        """Stop monitoring and return (duration, peak_memory_mb)."""
        self._stop_monitoring.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)

        duration = time.time() - self.start_time
        return duration, self.peak_memory_mb


def run_with_benchmark(
    cmd: Union[str, list[str]],
    name: str = "process",
    timeout: int | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict:
    """
    Run a command with benchmarking.

    Args:
        cmd: Command to run (string or list)
        name: Name for the benchmark (for logging)
        timeout: Maximum time to wait (seconds)
        shell: Whether to run through shell
        env: Environment variables
        cwd: Working directory

    Returns:
        Dictionary with keys:
        - success: bool
        - exit_code: int
        - duration: float (seconds)
        - memory_peak_mb: float
        - stdout: str
        - stderr: str
        - timeout: bool (True if process timed out)
    """
    print(f"Starting benchmark for: {name}")

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=shell, env=env, cwd=cwd
        )

        monitor = ProcessMonitor(process)
        monitor.start()

        # Wait for completion
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = -1
            timed_out = True
            print(f"Process timed out after {timeout}s")

        duration, peak_memory = monitor.stop()

        print(f"{name} completed in {duration:.1f}s")
        print(f"{name} peak memory: {peak_memory:.1f} MB")

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "duration": duration,
            "memory_peak_mb": peak_memory,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timeout": timed_out,
        }

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "duration": 0,
            "memory_peak_mb": 0,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
        }


def write_github_output(outputs: dict[str, str]) -> None:
    """Write multiple outputs for GitHub Actions."""
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")

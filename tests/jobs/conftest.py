"""Shared fixtures and utilities for job system tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import JobManager


class MockStream:
    """Mock stdout/stderr stream with read1() support."""

    def __init__(self):
        self._data = b""
        self._lines = []

    def read1(self, size=None):
        """Read and return up to size bytes (non-blocking read)."""
        if not self._data:
            return b""
        if size is None:
            result = self._data
            self._data = b""
        else:
            result = self._data[:size]
            self._data = self._data[size:]
        return result

    def write(self, data: bytes):
        """Add data to stream."""
        self._data += data

    def __iter__(self):
        """Iterate over lines in stream."""
        return iter(self._lines)


class MockProcess:
    """Mock subprocess that completes after N polls."""

    def __init__(self, exit_code: int = 0, complete_after_polls: int = 1):
        self.exit_code = exit_code
        self.complete_after = complete_after_polls
        self.poll_count = 0
        self.returncode = None
        self.pid = 12345
        self.terminated = False
        self.stdout = MockStream()
        self.stderr = MockStream()

    def poll(self):
        """Simulate poll() - returns None until complete."""
        self.poll_count += 1
        if self.poll_count >= self.complete_after:
            self.returncode = self.exit_code
            return self.exit_code
        return None

    def terminate(self):
        """Simulate terminate()."""
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        """Simulate wait()."""
        return self.returncode


@pytest.fixture
def temp_job_manager():
    """Create a JobManager with temporary database.

    Usage:
        def test_something(temp_job_manager):
            manager = temp_job_manager()
            # test code
    """
    managers = []

    def _create(
        max_local_jobs: int = 1,
        max_remote_jobs: int = 10,
        remote_poll_interval_s: float = 5.0,
    ) -> JobManager:
        tmp_dir = Path(tempfile.mkdtemp())
        # Create logs directory to prevent FileNotFoundError in monitoring threads
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        manager = JobManager(
            base_dir=tmp_dir,
            max_local_jobs=max_local_jobs,
            max_remote_jobs=max_remote_jobs,
            remote_poll_interval_s=remote_poll_interval_s,
        )
        managers.append(manager)
        return manager

    yield _create

    # Cleanup
    for manager in managers:
        if manager.base_dir.exists():
            shutil.rmtree(manager.base_dir)


def simple_job_config(
    name: str,
    module: str = "test.module",
    remote: bool = False,
    timeout_s: int = 60,
    **kwargs: Any,
) -> JobConfig:
    """Create a simple JobConfig for testing."""
    from metta.jobs.job_config import RemoteConfig

    return JobConfig(
        name=name,
        module=module,
        remote=RemoteConfig() if remote else None,
        timeout_s=timeout_s,
        **kwargs,
    )

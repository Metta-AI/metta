"""Tests for JobManager."""

import tempfile
import time
from pathlib import Path

import pytest

from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import ExitCode, JobManager
from metta.jobs.job_state import JobStatus


def test_job_manager_basic():
    """Test basic JobManager functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        manager = JobManager(base_dir, max_local_jobs=2, max_remote_jobs=5)

        # Submit a simple local job
        config = JobConfig(
            name="test_job",
            module="echo",
            args=["message=hello"],
            timeout_s=60,
            # remote=None (default) means local execution
        )

        # Submit job
        manager.submit(config)

        # Check status
        status = manager.get_status("test_job")
        assert status in ("pending", "running"), f"Expected pending or running, got {status}"

        # Job state should exist
        job_state = manager.get_job_state("test_job")
        assert job_state is not None
        assert job_state.name == "test_job"


def test_job_manager_group_operations():
    """Test group operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        manager = JobManager(base_dir)

        # Submit multiple jobs to same group
        for i in range(3):
            config = JobConfig(
                name=f"job_{i}",
                module="echo",
                args=[f"i={i}"],
                timeout_s=60,
                group="group_1",
                # remote=None (default) means local execution
            )
            manager.submit(config)

        # Get all jobs in group
        jobs = manager.get_group_jobs("group_1")
        assert len(jobs) == 3
        assert "job_0" in jobs
        assert "job_1" in jobs
        assert "job_2" in jobs


def test_job_manager_timeout_enforcement():
    """Test that JobManager enforces timeout_s and marks job as failed with correct exit code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        # Create logs directory that LocalJob expects
        (base_dir / "logs").mkdir(exist_ok=True)
        manager = JobManager(base_dir, max_local_jobs=1)

        # Submit a job with very short timeout that runs a long command
        config = JobConfig(
            name="timeout_test",
            module="__unused__",
            args=[],
            timeout_s=3,  # 3 second timeout
            metadata={"cmd": ["sleep", "60"]},  # Sleep for 60 seconds (will be killed)
        )
        manager.submit(config)

        # Wait for the job to start and then timeout
        # Poll for up to 15 seconds to give time for timeout to trigger
        start = time.time()
        max_wait = 15
        job_state = None

        while time.time() - start < max_wait:
            job_state = manager.get_job_state("timeout_test")
            if job_state and job_state.status == JobStatus.COMPLETED:
                break
            time.sleep(0.5)

        # Verify job was marked as timed out
        assert job_state is not None, "Job state should exist"
        assert job_state.status == JobStatus.COMPLETED, f"Job should be completed, got {job_state.status}"
        assert job_state.exit_code == ExitCode.TIMEOUT, (
            f"Exit code should be {ExitCode.TIMEOUT} (timeout), got {job_state.exit_code}"
        )
        assert job_state.acceptance_passed is False, "Timeout should mark acceptance as failed"


@pytest.mark.skip(reason="Remote jobs require SkyPilot infrastructure")
def test_job_manager_remote_timeout_enforcement():
    """Test that JobManager enforces timeout_s for remote jobs.

    Note: This test is skipped because it requires actual SkyPilot infrastructure.
    The timeout logic for remote jobs is tested via the local job test above,
    as both use the same _handle_job_completion_from_result() method and
    Job.wait() handles timeout for both local and remote jobs.
    """
    pass

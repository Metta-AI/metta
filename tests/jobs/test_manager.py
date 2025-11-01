"""Tests for JobManager."""

import tempfile
from pathlib import Path

from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import JobManager


def test_job_manager_basic():
    """Test basic JobManager functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        manager = JobManager(base_dir, max_local_jobs=2, max_remote_jobs=5)

        # Submit a simple local job
        config = JobConfig(
            name="test_job",
            module="echo",
            args={"message": "hello"},
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
                args={"i": i},
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

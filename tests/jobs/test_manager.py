"""Tests for JobManager."""

import tempfile
from pathlib import Path

from metta.jobs import JobConfig, JobManager


def test_job_manager_basic():
    """Test basic JobManager functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "jobs.db"
        manager = JobManager(db_path, max_local_jobs=2, max_remote_jobs=5)

        # Submit a simple local job
        config = JobConfig(
            name="test_job",
            module="echo",
            args={"message": "hello"},
            execution="local",
            timeout_s=60,
        )

        # Submit job
        manager.submit("test_batch", config)

        # Check status
        status = manager.get_status("test_batch", "test_job")
        assert status in ("pending", "running"), f"Expected pending or running, got {status}"

        # Job state should exist
        job_state = manager.get_job_state("test_batch", "test_job")
        assert job_state is not None
        assert job_state.name == "test_job"
        assert job_state.batch_id == "test_batch"


def test_job_manager_batch_operations():
    """Test batch operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "jobs.db"
        manager = JobManager(db_path)

        # Submit multiple jobs to same batch
        for i in range(3):
            config = JobConfig(
                name=f"job_{i}",
                module="echo",
                args={"i": i},
                execution="local",
                timeout_s=60,
            )
            manager.submit("batch_1", config)

        # Get all jobs in batch
        jobs = manager.get_batch_jobs("batch_1")
        assert len(jobs) == 3
        assert "job_0" in jobs
        assert "job_1" in jobs
        assert "job_2" in jobs

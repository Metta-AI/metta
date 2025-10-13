"""Behavioral tests for JobManager core functionality.

Combines tests for lifecycle, queueing, and error handling.
"""

import time
from unittest.mock import patch

import pytest

from tests.jobs.conftest import MockProcess, simple_job_config


def test_job_lifecycle_complete_flow(temp_job_manager):
    """Jobs flow through pending → running → completed with exit codes."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)
    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config("test_job")
        manager.submit(config)

        job_state = manager.get_job_state("test_job")
        assert job_state.status == "running"

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")
        assert job_state.status == "completed"
        assert job_state.exit_code == 0
        assert job_state.completed_at is not None


def test_job_with_non_zero_exit_code(temp_job_manager):
    """Jobs that fail capture non-zero exit code."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=42, complete_after_polls=1)
    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config("test_job")
        manager.submit(config)

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")
        assert job_state.status == "completed"
        assert job_state.exit_code == 42


def test_pending_job_when_no_slots(temp_job_manager):
    """Jobs stay pending when all worker slots full."""
    manager = temp_job_manager(max_local_jobs=0)
    config = simple_job_config("test_job")
    manager.submit(config)

    job_state = manager.get_job_state("test_job")
    assert job_state.status == "pending"
    assert job_state.exit_code is None


def test_respects_max_local_jobs_limit(temp_job_manager):
    """Never runs more than max_local_jobs concurrently."""
    manager = temp_job_manager(max_local_jobs=2)
    mock_processes = [MockProcess(exit_code=0, complete_after_polls=100) for _ in range(4)]
    with patch("subprocess.Popen", side_effect=mock_processes):
        for i in range(4):
            config = simple_job_config(f"job_{i}", remote=False)
            manager.submit(config)

        all_jobs = manager.get_all_jobs()
        running_count = sum(1 for job in all_jobs.values() if job.status == "running")
        pending_count = sum(1 for job in all_jobs.values() if job.status == "pending")

        assert running_count == 2
        assert pending_count == 2


def test_queued_job_starts_when_slot_frees(temp_job_manager):
    """Pending jobs start when slots become available."""
    manager = temp_job_manager(max_local_jobs=1)
    mock_proc1 = MockProcess(exit_code=0, complete_after_polls=1)
    mock_proc2 = MockProcess(exit_code=0, complete_after_polls=100)

    with patch("subprocess.Popen", side_effect=[mock_proc1, mock_proc2]):
        config1 = simple_job_config("job_1", remote=False)
        config2 = simple_job_config("job_2", remote=False)
        manager.submit(config1)
        manager.submit(config2)

        assert manager.get_job_state("job_1").status == "running"
        assert manager.get_job_state("job_2").status == "pending"

        time.sleep(0.3)

        for _ in range(5):
            manager.poll()
            time.sleep(0.01)

        assert manager.get_job_state("job_2").status == "running"


def test_cannot_submit_duplicate_job_name(temp_job_manager):
    """Submitting job with duplicate name raises error."""
    manager = temp_job_manager()
    config = simple_job_config("test_job")
    manager.submit(config)

    with pytest.raises(ValueError, match="already exists"):
        manager.submit(config)


def test_can_query_jobs(temp_job_manager):
    """Can retrieve job state and query all jobs."""
    manager = temp_job_manager()
    for i in range(3):
        config = simple_job_config(f"job_{i}")
        manager.submit(config)

    job_state = manager.get_job_state("job_1")
    assert job_state is not None
    assert job_state.name == "job_1"

    all_jobs = manager.get_all_jobs()
    assert len(all_jobs) == 3
    names = {job.name for job in all_jobs.values()}
    assert names == {"job_0", "job_1", "job_2"}

    assert manager.get_job_state("nonexistent") is None


def test_delete_job(temp_job_manager):
    """Can delete completed jobs."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)
    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config("test_job")
        manager.submit(config)
        assert manager.get_job_state("test_job") is not None

        time.sleep(0.2)

        manager.delete_job("test_job")
        assert manager.get_job_state("test_job") is None

        manager.delete_job("nonexistent")


def test_empty_manager(temp_job_manager):
    """Empty manager returns empty results."""
    manager = temp_job_manager()
    assert manager.get_all_jobs() == {}
    assert manager.get_job_state("nonexistent") is None
    assert manager.cancel_group("nonexistent_group") == 0
    assert manager.get_group_jobs("nonexistent_group") == {}


def test_job_timeout_configured(temp_job_manager):
    """Jobs can be configured with timeout."""
    manager = temp_job_manager()
    config = simple_job_config("test_job", timeout_s=30)
    manager.submit(config)

    job_state = manager.get_job_state("test_job")
    assert job_state.config.timeout_s == 30


def test_concurrent_poll_calls_safe(temp_job_manager):
    """Multiple poll() calls don't cause issues."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=5)
    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config("test_job")
        manager.submit(config)

        for _ in range(20):
            manager.poll()
            time.sleep(0.001)

        job_state = manager.get_job_state("test_job")
        assert job_state.status in ("running", "completed")


def test_local_and_remote_queues_independent(temp_job_manager):
    """Local and remote jobs have independent concurrency limits."""
    manager = temp_job_manager(max_local_jobs=1, max_remote_jobs=1)
    mock_local = MockProcess(exit_code=0, complete_after_polls=100)

    with patch("subprocess.Popen", return_value=mock_local):
        local_config = simple_job_config("local_job", remote=False)
        remote_config = simple_job_config("remote_job", remote=True)

        manager.submit(local_config)
        manager.submit(remote_config)

        assert manager.get_job_state("local_job").status == "running"
        assert manager.get_job_state("remote_job") is not None

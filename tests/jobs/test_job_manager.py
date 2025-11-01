"""Behavioral tests for JobManager core functionality.

Combines tests for lifecycle, queueing, and error handling.
"""

import time
from unittest.mock import patch

import pytest

from metta.jobs.job_manager import EXIT_CODE_SKIPPED
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


def test_job_waits_for_dependency_completion(temp_job_manager):
    """Jobs wait for dependencies to complete before starting."""
    manager = temp_job_manager()
    mock_proc1 = MockProcess(exit_code=0, complete_after_polls=2)
    mock_proc2 = MockProcess(exit_code=0, complete_after_polls=100)

    with patch("subprocess.Popen", side_effect=[mock_proc1, mock_proc2]):
        # Submit job1 first
        config1 = simple_job_config("job_1")
        manager.submit(config1)

        # Submit job2 with dependency on job1
        config2 = simple_job_config("job_2", dependency_names=["job_1"])
        manager.submit(config2)

        # Job1 should be running, job2 should be pending
        assert manager.get_job_state("job_1").status == "running"
        assert manager.get_job_state("job_2").status == "pending"

        # Wait for job1 to complete (monitoring happens in background thread)
        timeout = 2.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            manager.poll()  # Try to start job2
            time.sleep(0.1)
            job2_status = manager.get_job_state("job_2").status
            if job2_status == "running":
                break
            if time.time() - start_time > 1.0:
                # Debug: check if job1 actually completed
                job1_debug = manager.get_job_state("job_1")
                print(
                    f"DEBUG: job1 status={job1_debug.status}, "
                    f"exit_code={job1_debug.exit_code}, "
                    f"acceptance_passed={job1_debug.acceptance_passed}"
                )
                print(f"DEBUG: job2 status={job2_status}")

        # Job1 should be completed, job2 should now be running
        job1 = manager.get_job_state("job_1")
        assert job1.status == "completed", f"Expected job_1 completed, got {job1.status}"
        assert job1.exit_code == 0

        job2 = manager.get_job_state("job_2")
        assert job2.status == "running", f"Expected job_2 running after job_1 completed, got {job2.status}"


def test_job_skipped_when_dependency_fails(temp_job_manager):
    """Jobs are automatically skipped when dependency fails with non-zero exit code."""
    manager = temp_job_manager()
    mock_proc1 = MockProcess(exit_code=1, complete_after_polls=1)

    with patch("subprocess.Popen", return_value=mock_proc1):
        # Submit job1 that will fail
        config1 = simple_job_config("job_1")
        manager.submit(config1)

        # Submit job2 with dependency on job1
        config2 = simple_job_config("job_2", dependency_names=["job_1"])
        manager.submit(config2)

        # Wait for job1 to fail and job2 to be skipped
        time.sleep(0.3)
        for _ in range(10):
            completed = manager.poll()
            time.sleep(0.05)
            if "job_2" in completed:
                break

        # Job1 should be failed
        job1_state = manager.get_job_state("job_1")
        assert job1_state.status == "completed"
        assert job1_state.exit_code == 1

        # Job2 should be skipped (marked completed with EXIT_CODE_SKIPPED)
        job2_state = manager.get_job_state("job_2")
        assert job2_state.status == "completed", f"Expected job_2 completed (skipped), got {job2_state.status}"
        assert job2_state.exit_code == EXIT_CODE_SKIPPED, f"Expected EXIT_CODE_SKIPPED, got {job2_state.exit_code}"


def test_transitive_dependency_skip_and_retry(temp_job_manager):
    """Test transitive dependency skipping: A→B→C, if B fails, C is skipped. Retrying B resets C."""
    manager = temp_job_manager()
    mock_fail = MockProcess(exit_code=1, complete_after_polls=1)
    mock_success = MockProcess(exit_code=0, complete_after_polls=2)

    # Need mocks for: job_a (fail), job_a (retry success), job_b (success), job_c (success)
    with patch("subprocess.Popen", side_effect=[mock_fail, mock_success, mock_success, mock_success]):
        # Create chain: job_a → job_b → job_c
        config_a = simple_job_config("job_a")
        config_b = simple_job_config("job_b", dependency_names=["job_a"])
        config_c = simple_job_config("job_c", dependency_names=["job_b"])

        manager.submit(config_a)
        manager.submit(config_b)
        manager.submit(config_c)

        # Wait for job_a to fail, which should cascade skip job_b and job_c
        time.sleep(0.5)
        for _ in range(15):
            manager.poll()
            time.sleep(0.05)
            job_c_state = manager.get_job_state("job_c")
            if job_c_state and job_c_state.status == "completed":
                break

        # Verify initial failure cascade
        job_a_state = manager.get_job_state("job_a")
        assert job_a_state.status == "completed"
        assert job_a_state.exit_code == 1, "job_a should have failed"

        job_b_state = manager.get_job_state("job_b")
        assert job_b_state.status == "completed"
        assert job_b_state.exit_code == EXIT_CODE_SKIPPED, "job_b should be skipped due to job_a failure"

        job_c_state = manager.get_job_state("job_c")
        assert job_c_state.status == "completed"
        assert job_c_state.exit_code == EXIT_CODE_SKIPPED, "job_c should be skipped due to job_b being skipped"

        # Now retry job_a - this should transitively reset both job_b and job_c
        manager.delete_job("job_a")
        manager.submit(config_a)  # Resubmit with success mock

        # Wait for all jobs to complete successfully
        time.sleep(0.8)
        for _ in range(20):
            manager.poll()
            time.sleep(0.05)
            job_c_final = manager.get_job_state("job_c")
            if job_c_final and job_c_final.status == "running":
                break

        # Verify transitive reset worked
        job_a_final = manager.get_job_state("job_a")
        assert job_a_final.status == "completed"
        assert job_a_final.exit_code == 0, "job_a should succeed on retry"

        job_b_final = manager.get_job_state("job_b")
        # job_b should either be running or completed successfully (not skipped)
        assert job_b_final.exit_code != EXIT_CODE_SKIPPED, "job_b should be reset and run, not skipped"

        # job_c should also be reset and either pending/running/completed (not skipped)
        job_c_final = manager.get_job_state("job_c")
        if job_c_final.status == "completed":
            assert job_c_final.exit_code != EXIT_CODE_SKIPPED, "job_c should be reset and run, not remain skipped"

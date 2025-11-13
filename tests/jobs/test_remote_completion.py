"""Test remote job completion handling.

Tests that remote jobs properly transition to completed status when terminal states are reached.
"""

import time
from unittest.mock import MagicMock, patch

from tests.jobs.conftest import simple_job_config


def test_remote_job_completes_on_succeeded_status(temp_job_manager):
    """Remote job transitions to completed when SkyPilot reports SUCCEEDED."""
    manager = temp_job_manager(remote_poll_interval_s=0.1)  # Fast polling for tests
    config = simple_job_config("remote_job", remote=True)

    # Mock check_job_statuses to return SUCCEEDED status
    mock_statuses = {6767: {"status": "SUCCEEDED"}}
    mock_check_statuses = MagicMock(return_value=mock_statuses)

    with patch("devops.skypilot.utils.job_helpers.check_job_statuses", mock_check_statuses):
        with patch("metta.jobs.job_runner.RemoteJob") as MockRemoteJob:
            # Create mock remote job instance that properly sets job_id on submit()
            mock_job_instance = MagicMock()
            mock_job_instance.submit = MagicMock()  # submit() doesn't need to do anything
            mock_job_instance.job_id = "6767"  # Return as string (like real RemoteJob)
            mock_job_instance.request_id = None
            mock_job_instance.run_name = None
            mock_job_instance.log_path = "/tmp/test.log"
            mock_job_instance.get_logs = MagicMock()
            MockRemoteJob.return_value = mock_job_instance

            # Submit the job (this will start monitoring thread)
            manager.submit(config)

            # Wait for monitoring thread to pick up SUCCEEDED status
            # With 0.1s poll interval, 0.5s should be enough for several polls
            time.sleep(0.5)

            # Debug: check if mock was called and job_id is set
            print(f"check_job_statuses called {mock_check_statuses.call_count} times")
            print(f"mock submit() called {mock_job_instance.submit.call_count} times")

            # Verify job is marked as completed
            job_state = manager.get_job_state("remote_job")
            assert job_state is not None, "Job should exist in database"
            print(
                f"Job state: status={job_state.status}, exit_code={job_state.exit_code}, "
                f"skypilot_status={job_state.skypilot_status}, job_id={job_state.job_id}"
            )
            assert job_state.status == "completed", f"Expected completed, got {job_state.status}"
            assert job_state.exit_code == 0, f"Expected exit_code=0, got {job_state.exit_code}"
            assert job_state.skypilot_status == "SUCCEEDED", f"Expected SUCCEEDED, got {job_state.skypilot_status}"
            assert job_state.completed_at is not None, "Job should have completed_at timestamp"


def test_remote_job_completes_on_failed_status(temp_job_manager):
    """Remote job transitions to completed with non-zero exit code when SkyPilot reports FAILED."""
    manager = temp_job_manager()
    config = simple_job_config("remote_job", remote=True)

    # Mock check_job_statuses to return FAILED status
    mock_statuses = {6767: {"status": "FAILED"}}

    with patch("devops.skypilot.utils.job_helpers.check_job_statuses", return_value=mock_statuses):
        with patch("metta.jobs.job_runner.RemoteJob") as MockRemoteJob:
            mock_job_instance = MagicMock()
            mock_job_instance.job_id = 6767
            mock_job_instance.get_logs = MagicMock()
            MockRemoteJob.return_value = mock_job_instance

            manager.submit(config)
            time.sleep(0.5)

            job_state = manager.get_job_state("remote_job")
            assert job_state is not None
            assert job_state.status == "completed"
            assert job_state.exit_code == 1
            assert job_state.skypilot_status == "FAILED"


def test_remote_job_with_acceptance_criteria(temp_job_manager):
    """Remote job evaluates acceptance criteria on completion."""
    manager = temp_job_manager()
    config = simple_job_config("remote_job", remote=True)
    # Add acceptance criterion (will fail because no metrics)
    config.acceptance_criteria = []  # Empty = pass

    mock_statuses = {6767: {"status": "SUCCEEDED"}}

    with patch("devops.skypilot.utils.job_helpers.check_job_statuses", return_value=mock_statuses):
        with patch("metta.jobs.job_runner.RemoteJob") as MockRemoteJob:
            mock_job_instance = MagicMock()
            mock_job_instance.job_id = 6767
            mock_job_instance.get_logs = MagicMock()
            MockRemoteJob.return_value = mock_job_instance

            manager.submit(config)
            time.sleep(0.5)

            job_state = manager.get_job_state("remote_job")
            assert job_state is not None
            assert job_state.status == "completed"
            assert job_state.exit_code == 0
            # Empty criteria should result in None (no criteria to evaluate)
            assert job_state.acceptance_passed is None


def test_remote_job_stays_running_on_running_status(temp_job_manager):
    """Remote job stays in running state when SkyPilot reports RUNNING."""
    manager = temp_job_manager()
    config = simple_job_config("remote_job", remote=True)

    mock_statuses = {6767: {"status": "RUNNING"}}

    with patch("devops.skypilot.utils.job_helpers.check_job_statuses", return_value=mock_statuses):
        with patch("metta.jobs.job_runner.RemoteJob") as MockRemoteJob:
            mock_job_instance = MagicMock()
            mock_job_instance.job_id = 6767
            mock_job_instance.get_logs = MagicMock()
            MockRemoteJob.return_value = mock_job_instance

            manager.submit(config)
            time.sleep(0.5)

            job_state = manager.get_job_state("remote_job")
            assert job_state is not None
            assert job_state.status == "running"
            assert job_state.exit_code is None
            assert job_state.skypilot_status == "RUNNING"

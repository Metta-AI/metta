"""Test remote job completion handling.

Tests that remote jobs properly transition to completed status when terminal states are reached.
"""

from unittest.mock import MagicMock, patch

from tests.jobs.conftest import simple_job_config


def test_remote_job_marks_failed_when_launch_fails(temp_job_manager):
    """Remote job marked as failed immediately when launch fails (no job_id)."""
    manager = temp_job_manager()
    config = simple_job_config("remote_job", remote=True)

    # Mock _spawn_job to return a job with no job_id (launch failure)
    mock_job = MagicMock()
    mock_job.exit_code = 1

    with patch.object(manager, "_spawn_job", return_value=mock_job):
        # Submit and immediately try to start (poll does this)
        manager.submit(config)
        manager.poll()

        # Verify job is marked as completed with failure
        job_state = manager.get_job_state("remote_job")
        assert job_state is not None, "Job should exist in database"
        assert job_state.status == "completed", f"Expected completed, got {job_state.status}"
        assert job_state.exit_code == 1, f"Expected exit_code=1, got {job_state.exit_code}"
        assert job_state.completed_at is not None, "Job should have completed_at timestamp"

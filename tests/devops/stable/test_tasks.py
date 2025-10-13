"""Tests for Task execution logic."""

from operator import ge

from devops.stable.runner import TaskRunner
from devops.stable.state import ReleaseState
from devops.stable.tasks import Task
from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import JobManager
from metta.jobs.job_state import JobState


def make_job_state(
    name: str,
    exit_code: int = 0,
    job_id: str | None = None,
    wandb_run_id: str | None = None,
    wandb_url: str | None = None,
    checkpoint_uri: str | None = None,
    metrics: dict[str, float] | None = None,
    acceptance_passed: bool | None = None,
) -> JobState:
    """Create a fake JobState for testing."""
    job_state = JobState(
        name=name,
        config_json="{}",  # Not used in tests
        status="completed" if exit_code == 0 else "failed",
        job_id=job_id,
        started_at="2025-01-01T00:00:00",
        completed_at="2025-01-01T00:01:00",
        exit_code=exit_code,
        logs_path="/fake/log.txt",
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
        checkpoint_uri=checkpoint_uri,
        acceptance_passed=acceptance_passed,
    )
    if metrics:
        job_state.metrics = metrics
    return job_state


def make_task_runner_with_state(job_states: dict[str, JobState]) -> tuple[TaskRunner, JobManager]:
    """Create TaskRunner with JobManager pre-populated with job states."""
    from datetime import datetime
    from unittest.mock import Mock

    state = ReleaseState(
        version="v2025.01.01-test",
        created_at=datetime.utcnow().isoformat(),
    )

    # Mock JobManager that returns our test states
    job_manager = Mock(spec=JobManager)
    job_manager.get_job_state.side_effect = lambda name: job_states.get(name)

    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)
    return runner, job_manager


def test_passed_with_exit_code_zero():
    """Test that _passed returns True for exit code 0 with no acceptance criteria."""
    task = Task(JobConfig(name="test", module="test"))
    job_state = make_job_state(name="v2025.01.01-test_test", exit_code=0)

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is True


def test_passed_with_nonzero_exit():
    """Test that _passed returns False for non-zero exit code."""
    task = Task(JobConfig(name="test", module="test"))
    job_state = make_job_state(name="v2025.01.01-test_test", exit_code=1)

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is False


def test_passed_with_timeout():
    """Test that _passed returns False for timeout (exit code 124)."""
    task = Task(JobConfig(name="test", module="test"))
    job_state = make_job_state(name="v2025.01.01-test_test", exit_code=124)

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is False


def test_passed_records_job_id():
    """Test that job_id is available from JobState."""
    job_state = make_job_state(name="v2025.01.01-test_test", exit_code=0, job_id="sky-job-123")

    assert job_state.job_id == "sky-job-123"


def test_acceptance_criteria_pass():
    """Test that task passes when metrics meet acceptance thresholds."""
    task = Task(
        JobConfig(name="test", module="test"),
        acceptance=[("overview/sps", ge, 10000)],
    )
    job_state = make_job_state(
        name="v2025.01.01-test_test",
        exit_code=0,
        metrics={"overview/sps": 15000.0},
        acceptance_passed=True,  # Job-level acceptance evaluation passed
    )

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is True


def test_acceptance_criteria_fail():
    """Test that task fails when metrics don't meet acceptance thresholds."""
    task = Task(
        JobConfig(name="test", module="test"),
        acceptance=[("overview/sps", ge, 10000)],
    )
    job_state = make_job_state(
        name="v2025.01.01-test_test",
        exit_code=0,
        metrics={"overview/sps": 5000.0},
        acceptance_passed=False,  # Job-level acceptance evaluation failed
    )

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is False


def test_acceptance_criteria_missing_metric():
    """Test that task fails when required metric is missing."""
    task = Task(
        JobConfig(name="test", module="test"),
        acceptance=[("overview/sps", ge, 10000)],
    )
    job_state = make_job_state(
        name="v2025.01.01-test_test",
        exit_code=0,
        metrics={},  # Missing metric
        acceptance_passed=False,  # Job-level acceptance evaluation failed due to missing metric
    )

    runner, _ = make_task_runner_with_state({"v2025.01.01-test_test": job_state})

    assert runner._passed("v2025.01.01-test_test", task) is False


def test_checkpoint_uri_extraction():
    """Test that checkpoint_uri is available from JobState."""
    job_state = make_job_state(
        name="v2025.01.01-test_train_task",
        exit_code=0,
        wandb_run_id="abc123",
        wandb_url="https://wandb.ai/entity/project/runs/abc123",
        checkpoint_uri="wandb://run/abc123",
    )

    # Verify artifacts are in JobState
    assert job_state.checkpoint_uri == "wandb://run/abc123"
    assert job_state.wandb_run_id == "abc123"
    assert job_state.wandb_url == "https://wandb.ai/entity/project/runs/abc123"

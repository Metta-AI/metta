"""Tests for Task execution logic."""

from operator import ge

from devops.stable.tasks import Task, TrainingTask
from metta.jobs.models import JobConfig
from metta.jobs.state import JobState


def make_job_state(
    name: str,
    exit_code: int = 0,
    job_id: str | None = None,
    wandb_run_id: str | None = None,
    wandb_url: str | None = None,
    checkpoint_uri: str | None = None,
    metrics: dict[str, float] | None = None,
) -> JobState:
    """Create a fake JobState for testing."""
    job_state = JobState(
        batch_id="test_batch",
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
    )
    if metrics:
        job_state.metrics = metrics
    return job_state


class ConcreteTestTask(Task):
    """Concrete Task implementation for testing base Task (exit code only)."""

    def __init__(self, name: str, exit_code: int = 0, job_id: str | None = None):
        # Create minimal JobConfig for testing
        job_config = JobConfig(name=name, module="test", execution="local")
        super().__init__(job_config)
        self._exit_code = exit_code
        self._job_id = job_id

    def make_job_state(self) -> JobState:
        """Create JobState for testing."""
        return make_job_state(
            name=self.name,
            exit_code=self._exit_code,
            job_id=self._job_id,
        )


class ConcreteTrainingTask(TrainingTask):
    """Concrete TrainingTask implementation for testing (with metrics/acceptance)."""

    def __init__(self, name: str, logs: str = "", exit_code: int = 0, **kwargs):
        # Create JobConfig from args
        job_config = JobConfig(name=name, module="fake.module", execution="local")
        acceptance = kwargs.pop("acceptance", None)
        wandb_metrics = kwargs.pop("wandb_metrics", None)
        super().__init__(
            job_config=job_config,
            acceptance=acceptance,
            wandb_metrics=wandb_metrics,
        )
        self._logs = logs
        self._exit_code = exit_code

    def make_job_state(
        self,
        wandb_run_id: str | None = None,
        wandb_url: str | None = None,
        checkpoint_uri: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> JobState:
        """Create JobState for testing."""
        return make_job_state(
            name=self.name,
            exit_code=self._exit_code,
            wandb_run_id=wandb_run_id,
            wandb_url=wandb_url,
            checkpoint_uri=checkpoint_uri,
            metrics=metrics,
        )


def test_evaluate_result_timeout():
    """Test that exit code 124 is recognized as timeout."""
    task = ConcreteTestTask(name="test", exit_code=124)
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.outcome == "failed"
    assert result.error == "Timeout exceeded"
    assert result.exit_code == 124


def test_evaluate_result_nonzero_exit():
    """Test that non-zero exit codes result in failure."""
    task = ConcreteTestTask(name="test", exit_code=1)
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.outcome == "failed"
    assert result.error is not None and len(result.error) > 0
    assert result.exit_code == 1


def test_evaluate_result_success():
    """Test successful result for base Task (exit code 0)."""
    task = ConcreteTestTask(name="test", exit_code=0)
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.outcome == "passed"
    assert result.error is None


def test_evaluate_result_records_job_id():
    """Test that job_id is captured from JobState."""
    task = ConcreteTestTask(name="test", exit_code=0, job_id="sky-job-123")
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.job_id == "sky-job-123"


# TrainingTask-specific tests


def test_training_task_acceptance_pass():
    """Test that TrainingTask with acceptance criteria passes when metrics meet thresholds."""
    task = ConcreteTrainingTask(name="test", acceptance=[])
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.outcome == "passed"
    assert result.error is None


def test_training_task_acceptance_fail():
    """Test that TrainingTask with failing acceptance criteria result in failure."""
    task = ConcreteTrainingTask(
        name="test",
        acceptance=[("overview/sps", ge, 10000)],
        wandb_metrics=["overview/sps"],
    )
    # Job state without metrics (simulates missing metric)
    job_state = task.make_job_state()

    result = task.evaluate_result(job_state)

    assert result.outcome == "failed"
    assert "overview/sps" in result.error
    assert "metric missing" in result.error


def test_training_task_extracts_checkpoint_uri():
    """Test that TrainingTask extracts checkpoint_uri from JobState."""
    task = ConcreteTrainingTask(name="train_task")
    job_state = task.make_job_state(
        wandb_run_id="abc123",
        wandb_url="https://wandb.ai/entity/project/runs/abc123",
        checkpoint_uri="wandb://run/abc123",
    )

    result = task.evaluate_result(job_state)

    assert "checkpoint_uri" in result.artifacts
    assert result.artifacts["checkpoint_uri"] == "wandb://run/abc123"
    assert result.artifacts["wandb_run_id"] == "abc123"

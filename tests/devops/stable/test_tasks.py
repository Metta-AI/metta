"""Tests for Task execution logic."""

from operator import ge

from devops.job_runner import JobResult
from devops.stable.tasks import Task, TrainingTask


class FakeJob:
    """Fake job for testing Task._evaluate_result."""

    def __init__(self, exit_code=0, logs="", job_id=None):
        self.exit_code = exit_code
        self._logs = logs
        self.job_id = job_id
        self.logs_path = "/fake/log.txt"

    def wait(self, stream_output=False):
        return JobResult(
            name="fake",
            exit_code=self.exit_code,
            logs_path=self.logs_path,
            duration_s=1.0,
            job_id=self.job_id,
        )

    def get_logs(self):
        return self._logs


class ConcreteTestTask(Task):
    """Concrete Task implementation for testing base Task (exit code only)."""

    def __init__(self, name: str, exit_code: int = 0, job_id: str | None = None):
        super().__init__(name)
        self._exit_code = exit_code
        self._job_id = job_id

    def execute(self) -> JobResult:
        return JobResult(
            name=self.name,
            exit_code=self._exit_code,
            logs_path="/fake/log.txt",
            duration_s=1.0,
            job_id=self._job_id,
        )


class ConcreteTrainingTask(TrainingTask):
    """Concrete TrainingTask implementation for testing (with metrics/acceptance)."""

    def __init__(self, name: str, logs: str = "", exit_code: int = 0, **kwargs):
        super().__init__(name=name, module="fake.module", args=[], **kwargs)
        self._logs = logs
        self._exit_code = exit_code

    def execute(self) -> JobResult:
        """Execute fake job and return result with logs."""

        class FakeJobResult(JobResult):
            def __init__(self, name, exit_code, logs, logs_path, duration_s):
                super().__init__(name, exit_code, logs_path, duration_s=duration_s)
                self._logs = logs

            def get_logs(self):
                return self._logs

        return FakeJobResult(
            name=self.name,
            exit_code=self._exit_code,
            logs=self._logs,
            logs_path="/fake/log.txt",
            duration_s=1.0,
        )


def test_evaluate_result_timeout():
    """Test that exit code 124 is recognized as timeout."""
    task = ConcreteTestTask(name="test", exit_code=124)
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.outcome == "failed"
    assert result.error == "Timeout exceeded"
    assert result.exit_code == 124


def test_evaluate_result_nonzero_exit():
    """Test that non-zero exit codes result in failure."""
    task = ConcreteTestTask(name="test", exit_code=1)
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.outcome == "failed"
    assert result.error is not None and len(result.error) > 0
    assert result.exit_code == 1


def test_evaluate_result_success():
    """Test successful result for base Task (exit code 0)."""
    task = ConcreteTestTask(name="test", exit_code=0)
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.outcome == "passed"
    assert result.error is None


def test_evaluate_result_records_job_id():
    """Test that job_id is captured from JobResult."""
    task = ConcreteTestTask(name="test", exit_code=0, job_id="sky-job-123")
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.job_id == "sky-job-123"


# TrainingTask-specific tests


def test_training_task_acceptance_pass():
    """Test that TrainingTask with acceptance criteria passes when metrics meet thresholds."""
    task = ConcreteTrainingTask(name="test", logs="Some log output\n", acceptance=[])
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.outcome == "passed"
    assert result.error is None


def test_training_task_acceptance_fail():
    """Test that TrainingTask with failing acceptance criteria result in failure."""
    task = ConcreteTrainingTask(
        name="test",
        logs="Some log output\n",
        acceptance=[("overview/sps", ge, 10000)],
        wandb_metrics=["overview/sps"],
    )
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert result.outcome == "failed"
    assert "overview/sps" in result.error
    assert "metric missing" in result.error


def test_training_task_extracts_checkpoint_uri():
    """Test that TrainingTask extracts checkpoint_uri from wandb info."""
    task = ConcreteTrainingTask(
        name="train_task",
        logs="wandb: View run at https://wandb.ai/entity/project/runs/abc123\n",
    )
    job_result = task.execute()

    result = task._convert_result(job_result)

    assert "checkpoint_uri" in result.artifacts
    assert result.artifacts["checkpoint_uri"] == "wandb://run/abc123"
    assert result.artifacts["wandb_run_id"] == "abc123"

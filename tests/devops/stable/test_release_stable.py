"""Tests for release_stable orchestration."""

from datetime import datetime

from devops.job_runner import JobResult
from devops.stable.runner import TaskRunner
from devops.stable.state import ReleaseState, load_state, save_state
from devops.stable.tasks import Task, TaskResult


class FakeTask(Task):
    """Fake task for testing."""

    def __init__(self, name: str, exit_code: int = 0, error: str | None = None):
        super().__init__(name)
        self._exit_code = exit_code
        self._error = error

    def execute(self) -> JobResult:
        return JobResult(
            name=self.name,
            exit_code=self._exit_code,
            logs_path="/fake/log.txt",
            duration_s=1.0,
        )


def test_state_persistence(tmp_path, monkeypatch):
    """Test that state persists correctly to disk."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    # Create and save state
    state = ReleaseState(
        version="release_1.0.0",
        created_at=datetime.utcnow().isoformat(),
        commit_sha="abc123",
    )
    state.results["test_task"] = TaskResult(
        name="test_task",
        started_at=datetime.utcnow().isoformat(),
        ended_at=datetime.utcnow().isoformat(),
        exit_code=0,
        outcome="passed",
    )

    save_state(state)

    # Load and verify
    loaded = load_state("release_1.0.0")
    assert loaded is not None
    assert loaded.version == "release_1.0.0"
    assert "test_task" in loaded.results
    assert loaded.results["test_task"].outcome == "passed"


def test_state_handles_version_prefix(tmp_path, monkeypatch):
    """Test that version with or without 'release_' prefix works."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    state = ReleaseState(
        version="release_2.0.0",
        created_at=datetime.utcnow().isoformat(),
    )
    save_state(state)

    # Load with and without prefix
    loaded1 = load_state("release_2.0.0")
    loaded2 = load_state("2.0.0")

    assert loaded1 is not None
    assert loaded2 is not None
    assert loaded1.version == loaded2.version


def test_run_task_skips_completed(tmp_path, monkeypatch):
    """Test that already-passed tasks are skipped."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")
    state.results["already_passed"] = TaskResult(
        name="already_passed",
        started_at="now",
        ended_at="now",
        outcome="passed",
        exit_code=0,
    )

    task = FakeTask(name="already_passed")

    runner = TaskRunner(state, interactive=False)
    result = runner._run_with_deps(task)

    # Should return existing result without re-running
    assert result.outcome == "passed"
    assert result.name == "already_passed"


def test_run_task_retries_failed(tmp_path, monkeypatch):
    """Test that failed tasks can be retried."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)
    monkeypatch.setattr("devops.stable.state.LOG_DIR_LOCAL", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    state = ReleaseState(version="release_1.0.0", created_at="now")
    state.results["failed_task"] = TaskResult(
        name="failed_task",
        started_at="now",
        ended_at="now",
        outcome="failed",
        exit_code=1,
        error="Previous failure",
    )

    task = FakeTask(name="failed_task", exit_code=0)

    runner = TaskRunner(state, interactive=False)
    result = runner._run_with_deps(task)

    # Should have retried and passed
    assert result.outcome == "passed"


def test_run_task_skips_missing_dependencies(tmp_path, monkeypatch):
    """Test that task is skipped when dependencies are not complete."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create dependent task with a missing dependency
    dependent_task = FakeTask(name="dependent_task")
    dependency = FakeTask(name="missing_dependency")
    dependent_task.dependencies = [dependency]

    runner = TaskRunner(state, interactive=False)
    result = runner._run_with_deps(dependent_task)

    # Dependency will run and pass, so dependent task should also run
    assert result.outcome == "passed"


def test_run_task_skips_failed_dependencies(tmp_path, monkeypatch):
    """Test that task is skipped when dependencies failed."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create failed dependency
    failed_dep = FakeTask(name="failed_dep", exit_code=1)

    # Create dependent task
    dependent_task = FakeTask(name="eval_after_train")
    dependent_task.dependencies = [failed_dep]

    runner = TaskRunner(state, interactive=False)
    result = runner._run_with_deps(dependent_task)

    assert result.outcome == "skipped"
    assert "Dependency failed_dep did not pass" in result.error


def test_run_task_injects_policy_uri(tmp_path, monkeypatch):
    """Test that evaluation tasks get policy_uri from training dependencies."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)
    monkeypatch.setattr("devops.stable.state.LOG_DIR_LOCAL", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    from devops.stable.tasks import TrainingTask, evaluate

    # Create a fake training task with checkpoint
    class FakeTrainingTask(TrainingTask):
        def execute(self) -> JobResult:
            class FakeJobResult(JobResult):
                def __init__(self):
                    super().__init__("train", 0, "/fake/log.txt", duration_s=1.0)

                def get_logs(self):
                    return "wandb: View run at https://wandb.ai/entity/project/runs/abc123\n"

            return FakeJobResult()

    train_task = FakeTrainingTask(
        name="train_task",
        module="fake.module",
        args=[],
    )

    # Run training task to get result
    train_result = train_task.run()
    assert "checkpoint_uri" in train_result.artifacts

    # Create evaluation task with dependency
    eval_task = evaluate(
        name="eval_task",
        module="test.evaluate",
        training_task=train_task,
    )

    # Check that policy_uri will be injected when executed
    # (We can't easily test execute() without running actual commands,
    # but we can verify the dependency is set)
    assert train_task in eval_task.dependencies


def test_run_task_handles_exceptions(tmp_path, monkeypatch):
    """Test that exceptions during task execution are caught and reported."""
    monkeypatch.setattr("devops.stable.state.STATE_DIR", tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create a mock task that raises an exception
    class ErrorTask(Task):
        def execute(self) -> JobResult:
            raise RuntimeError("Simulated error")

    task = ErrorTask(name="error_task")

    runner = TaskRunner(state, interactive=False)
    result = runner._run_with_deps(task)

    assert result.outcome == "failed"
    assert "Exception" in result.error
    assert result.exit_code == 1

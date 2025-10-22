"""Tests for release_stable orchestration."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from devops.stable.runner import TaskRunner
from devops.stable.state import ReleaseState, load_state, save_state
from devops.stable.tasks import Task, TaskResult
from metta.jobs.manager import JobManager
from metta.jobs.models import JobConfig
from metta.jobs.state import JobState


class FakeTask(Task):
    """Fake task for testing."""

    def __init__(self, name: str, exit_code: int = 0, error: str | None = None):
        job_config = JobConfig(name=name, module="fake.module", execution="local")
        super().__init__(job_config)
        self._exit_code = exit_code
        self._error = error

    def make_job_state(self) -> JobState:
        """Create a fake JobState for testing."""
        return JobState(
            batch_id="test_batch",
            name=self.name,
            config=self.job_config,
            status="completed" if self._exit_code == 0 else "failed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            exit_code=self._exit_code,
            logs_path="/fake/log.txt",
        )


def make_mock_job_manager(tmp_path: Path):
    """Create a mock JobManager that returns pre-configured JobStates."""
    job_manager = Mock(spec=JobManager)
    job_manager._job_states = {}

    def mock_submit(batch_id: str, config: JobConfig):
        # Store config for later retrieval
        job_manager._job_states[(batch_id, config.name)] = config

    def mock_poll():
        # Mark all submitted jobs as complete
        return list(job_manager._job_states.keys())

    def mock_get_job_state(batch_id: str, name: str):
        config = job_manager._job_states.get((batch_id, name))
        if not config:
            return None
        return JobState(
            batch_id=batch_id,
            name=name,
            config=config,
            status="completed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            exit_code=0,
            logs_path=str(tmp_path / f"{name}.log"),
        )

    def mock_wait_for_job(batch_id: str, name: str, poll_interval_s: float = 1.0):
        # Immediately return completed job state
        return mock_get_job_state(batch_id, name)

    job_manager.submit = mock_submit
    job_manager.poll = mock_poll
    job_manager.get_job_state = mock_get_job_state
    job_manager.wait_for_job = mock_wait_for_job

    return job_manager


def test_state_persistence(tmp_path, monkeypatch):
    """Test that state persists correctly to disk."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

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


def test_state_handles_version_exact_match(tmp_path, monkeypatch):
    """Test that version strings must match exactly."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(
        version="release_2.0.0",
        created_at=datetime.utcnow().isoformat(),
    )
    save_state(state)

    # Load with exact version string
    loaded1 = load_state("release_2.0.0")
    assert loaded1 is not None
    assert loaded1.version == "release_2.0.0"

    # Different version string won't match
    loaded2 = load_state("2.0.0")
    assert loaded2 is None


def test_run_task_skips_completed(tmp_path, monkeypatch):
    """Test that already-passed tasks are skipped."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")
    state.results["already_passed"] = TaskResult(
        name="already_passed",
        started_at="now",
        ended_at="now",
        outcome="passed",
        exit_code=0,
    )

    task = FakeTask(name="already_passed")
    job_manager = make_mock_job_manager(tmp_path)

    runner = TaskRunner(state, job_manager, interactive=False)
    result = runner._run_with_deps(task)

    # Should return existing result without re-running
    assert result.outcome == "passed"
    assert result.name == "already_passed"


def test_run_task_retries_failed(tmp_path, monkeypatch):
    """Test that failed tasks can be retried."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

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
    job_manager = make_mock_job_manager(tmp_path)

    runner = TaskRunner(state, job_manager, interactive=False)
    result = runner._run_with_deps(task)

    # Should have retried and passed
    assert result.outcome == "passed"


def test_run_task_skips_missing_dependencies(tmp_path, monkeypatch):
    """Test that task is skipped when dependencies are not complete."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create dependent task with a missing dependency
    dependent_task = FakeTask(name="dependent_task")
    dependency = FakeTask(name="missing_dependency")
    dependent_task.dependencies = [dependency]

    job_manager = make_mock_job_manager(tmp_path)
    runner = TaskRunner(state, job_manager, interactive=False)
    result = runner._run_with_deps(dependent_task)

    # Dependency will run and pass, so dependent task should also run
    assert result.outcome == "passed"


def test_run_task_skips_failed_dependencies(tmp_path, monkeypatch):
    """Test that task is skipped when dependencies failed."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create mock job manager that returns failed state for failed_dep
    job_manager = Mock(spec=JobManager)
    job_manager._job_states = {}

    def mock_submit(batch_id: str, config: JobConfig):
        job_manager._job_states[(batch_id, config.name)] = config

    def mock_poll():
        return list(job_manager._job_states.keys())

    def mock_get_job_state(batch_id: str, name: str):
        config = job_manager._job_states.get((batch_id, name))
        if not config:
            return None
        # Return failed state for failed_dep
        exit_code = 1 if name == "failed_dep" else 0
        return JobState(
            batch_id=batch_id,
            name=name,
            config=config,
            status="completed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            exit_code=exit_code,
            logs_path=f"/fake/{name}.log",
        )

    def mock_wait_for_job(batch_id: str, name: str, poll_interval_s: float = 1.0):
        # Immediately return completed job state
        return mock_get_job_state(batch_id, name)

    job_manager.submit = mock_submit
    job_manager.poll = mock_poll
    job_manager.get_job_state = mock_get_job_state
    job_manager.wait_for_job = mock_wait_for_job

    # Create failed dependency
    failed_dep = FakeTask(name="failed_dep", exit_code=1)

    # Create dependent task
    dependent_task = FakeTask(name="eval_after_train")
    dependent_task.dependencies = [failed_dep]

    runner = TaskRunner(state, job_manager, interactive=False)
    result = runner._run_with_deps(dependent_task)

    assert result.outcome == "skipped"
    assert "Dependency failed_dep did not pass" in result.error


def test_run_task_injects_policy_uri(tmp_path, monkeypatch):
    """Test that evaluation tasks get policy_uri from training dependencies."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Mark training task as already completed in state with checkpoint
    train_result = TaskResult(
        name="train_task",
        started_at="now",
        ended_at="now",
        outcome="passed",
        exit_code=0,
        artifacts={"checkpoint_uri": "wandb://run/abc123"},
    )
    state.results["train_task"] = train_result

    # Create training task (will be skipped since already completed)
    train_config = JobConfig(name="train_task", module="fake.module", execution="local")
    train_task = Task(train_config)

    # Create evaluation task that depends on training
    eval_config = JobConfig(
        name="eval_task",
        module="test.evaluate",
        args={},  # No policy_uri yet
        execution="local",
    )
    eval_task = Task(eval_config)
    eval_task.dependencies = [train_task]

    # Create mock job manager
    job_manager = make_mock_job_manager(tmp_path)

    # Run evaluation task - should inject policy_uri from cached train result
    runner = TaskRunner(state, job_manager, interactive=False)
    runner._run_with_deps(eval_task)

    # Verify policy_uri was injected into the task's job_config
    assert "policy_uri" in eval_task.job_config.args
    assert eval_task.job_config.args["policy_uri"] == "wandb://run/abc123"


def test_run_task_handles_exceptions(tmp_path, monkeypatch):
    """Test that exceptions during task execution are caught and reported."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create mock job manager that raises exception on poll
    job_manager = Mock(spec=JobManager)

    def mock_submit(batch_id: str, config: JobConfig):
        pass

    def mock_poll():
        raise RuntimeError("Simulated error")

    job_manager.submit = mock_submit
    job_manager.poll = mock_poll

    task = FakeTask(name="error_task")

    runner = TaskRunner(state, job_manager, interactive=False)
    result = runner._run_with_deps(task)

    assert result.outcome == "failed"
    assert "Exception" in result.error
    assert result.exit_code == 1

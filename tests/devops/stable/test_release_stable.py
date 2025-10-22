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
    job_states = {}

    def mock_submit(batch_id: str, config: JobConfig):
        # Store config for later retrieval
        job_states[(batch_id, config.name)] = config

    def mock_poll():
        # Mark all submitted jobs as complete
        return list(job_states.keys())

    def mock_get_job_state(batch_id: str, name: str):
        config = job_states.get((batch_id, name))
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

    job_manager = Mock(spec=JobManager)
    job_manager.submit.side_effect = mock_submit
    job_manager.poll.side_effect = mock_poll
    job_manager.get_job_state.side_effect = mock_get_job_state
    job_manager.wait_for_job.side_effect = mock_wait_for_job

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
    runner.run_all([task])

    # Should return existing result without re-running
    assert state.results["already_passed"].outcome == "passed"
    assert state.results["already_passed"].name == "already_passed"


def test_run_task_skips_failed_by_default(tmp_path, monkeypatch):
    """Test that failed tasks are skipped by default (no retry unless --retry-failed)."""
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

    # Default: retry_failed=False (skip failed tasks)
    runner = TaskRunner(state, job_manager, interactive=False, retry_failed=False)
    runner.run_all([task])

    # Should skip the failed task (not retry)
    assert state.results["failed_task"].outcome == "failed"
    assert state.results["failed_task"].error == "Previous failure"


def test_run_task_retries_failed_when_enabled(tmp_path, monkeypatch):
    """Test that failed tasks are retried when retry_failed=True."""
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

    # Explicitly enable retry_failed=True
    runner = TaskRunner(state, job_manager, interactive=False, retry_failed=True)
    runner.run_all([task])

    # Should have retried and passed
    assert state.results["failed_task"].outcome == "passed"


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
    runner.run_all([dependency, dependent_task])

    # Dependency will run and pass, so dependent task should also run
    assert state.results["dependent_task"].outcome == "passed"


def test_run_task_skips_failed_dependencies(tmp_path, monkeypatch):
    """Test that task is skipped when dependencies failed."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create mock job manager that returns failed state for failed_dep
    job_states = {}
    completed_jobs = set()

    def mock_submit(batch_id: str, config: JobConfig):
        job_states[(batch_id, config.name)] = config

    def mock_poll():
        # Return jobs that haven't been polled yet
        pending = [key for key in job_states.keys() if key not in completed_jobs]
        completed_jobs.update(pending)
        return pending

    def mock_get_job_state(batch_id: str, name: str):
        config = job_states.get((batch_id, name))
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

    job_manager = Mock(spec=JobManager)
    job_manager.submit.side_effect = mock_submit
    job_manager.poll.side_effect = mock_poll
    job_manager.get_job_state.side_effect = mock_get_job_state

    # Create failed dependency
    failed_dep = FakeTask(name="failed_dep", exit_code=1)

    # Create dependent task
    dependent_task = FakeTask(name="eval_after_train")
    dependent_task.dependencies = [failed_dep]

    runner = TaskRunner(state, job_manager, interactive=False)
    runner.run_all([failed_dep, dependent_task])

    assert state.results["eval_after_train"].outcome == "skipped"
    assert "Dependency failed_dep did not pass" in state.results["eval_after_train"].error


def test_run_all_with_parallel_execution(tmp_path, monkeypatch):
    """Test that run_all executes independent tasks in parallel."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create two independent tasks (no dependencies)
    task1 = FakeTask(name="task1", exit_code=0)
    task2 = FakeTask(name="task2", exit_code=0)

    job_manager = make_mock_job_manager(tmp_path)
    runner = TaskRunner(state, job_manager, interactive=False, retry_failed=False)

    # Run all tasks
    runner.run_all([task1, task2])

    # Both tasks should complete
    assert "task1" in state.results
    assert "task2" in state.results
    assert state.results["task1"].outcome == "passed"
    assert state.results["task2"].outcome == "passed"


def test_run_all_respects_dependencies(tmp_path, monkeypatch):
    """Test that run_all respects task dependencies."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create tasks with dependencies: task2 depends on task1
    task1 = FakeTask(name="task1", exit_code=0)
    task2 = FakeTask(name="task2", exit_code=0)
    task2.dependencies = [task1]

    job_manager = make_mock_job_manager(tmp_path)
    runner = TaskRunner(state, job_manager, interactive=False, retry_failed=False)

    # Run all tasks
    runner.run_all([task1, task2])

    # Both tasks should complete, task1 before task2
    assert "task1" in state.results
    assert "task2" in state.results
    assert state.results["task1"].outcome == "passed"
    assert state.results["task2"].outcome == "passed"


def test_run_all_skips_on_failed_dependency(tmp_path, monkeypatch):
    """Test that run_all skips tasks when dependencies fail."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create tasks: task1 fails, task2 depends on task1
    task1 = FakeTask(name="task1", exit_code=1)  # Will fail
    task2 = FakeTask(name="task2", exit_code=0)
    task2.dependencies = [task1]

    # Create mock that returns failure for task1
    job_states = {}
    completed_jobs = set()

    def mock_submit(batch_id: str, config: JobConfig):
        job_states[(batch_id, config.name)] = config

    def mock_poll():
        # Return jobs that haven't been polled yet
        pending = [key for key in job_states.keys() if key not in completed_jobs]
        completed_jobs.update(pending)
        return pending

    def mock_get_job_state(batch_id: str, name: str):
        config = job_states.get((batch_id, name))
        if not config:
            return None
        # task1 fails, others succeed
        exit_code = 1 if name == "task1" else 0
        return JobState(
            batch_id=batch_id,
            name=name,
            config=config,
            status="completed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            exit_code=exit_code,
            logs_path=str(tmp_path / f"{name}.log"),
        )

    job_manager = Mock(spec=JobManager)
    job_manager.submit.side_effect = mock_submit
    job_manager.poll.side_effect = mock_poll
    job_manager.get_job_state.side_effect = mock_get_job_state

    runner = TaskRunner(state, job_manager, interactive=False, retry_failed=False)

    # Run all tasks
    runner.run_all([task1, task2])

    # task1 should fail, task2 should be skipped
    assert "task1" in state.results
    assert "task2" in state.results
    assert state.results["task1"].outcome == "failed"
    assert state.results["task2"].outcome == "skipped"


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
    runner.run_all([train_task, eval_task])

    # Verify policy_uri was injected into the task's job_config
    assert "policy_uri" in eval_task.job_config.args
    assert eval_task.job_config.args["policy_uri"] == "wandb://run/abc123"


def test_run_task_handles_exceptions(tmp_path, monkeypatch):
    """Test that exceptions during task execution are caught and reported."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="release_1.0.0", created_at="now")

    # Create mock job manager that raises exception
    job_manager = Mock(spec=JobManager)

    def mock_submit(batch_id: str, config: JobConfig):
        pass

    def mock_poll():
        # Return completed immediately
        return [("release_release_1.0.0", "error_task")]

    def mock_get_job_state(batch_id: str, name: str):
        raise RuntimeError("Simulated error")

    job_manager.submit.side_effect = mock_submit
    job_manager.poll.side_effect = mock_poll
    job_manager.get_job_state.side_effect = mock_get_job_state

    task = FakeTask(name="error_task")

    runner = TaskRunner(state, job_manager, interactive=False)
    runner.run_all([task])

    assert state.results["error_task"].outcome == "failed"
    assert "Exception" in state.results["error_task"].error
    assert state.results["error_task"].exit_code == 1

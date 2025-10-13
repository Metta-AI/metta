"""Tests for release_stable orchestration."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from devops.stable.runner import TaskRunner
from devops.stable.state import ReleaseState, load_state, save_state
from devops.stable.tasks import Task
from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import JobManager
from metta.jobs.job_state import JobState


class FakeTask(Task):
    """Fake task for testing."""

    def __init__(self, name: str, exit_code: int = 0, error: str | None = None):
        job_config = JobConfig(name=name, module="fake.module")  # remote=None (default) = local
        super().__init__(job_config)
        self._exit_code = exit_code
        self._error = error

    def make_job_state(self) -> JobState:
        """Create a fake JobState for testing."""
        return JobState(
            name=self.name,
            config=self.job_config,
            status="completed" if self._exit_code == 0 else "failed",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            exit_code=self._exit_code,
            logs_path="/fake/log.txt",
        )


def make_mock_job_manager(tmp_path: Path, existing_states: dict[str, JobState] | None = None):
    """Create a mock JobManager that returns pre-configured JobStates.

    Args:
        tmp_path: Temporary path for log files
        existing_states: Pre-existing job states (for testing resumption)
    """
    job_states = dict(existing_states) if existing_states else {}

    def mock_submit(config: JobConfig):
        # Store config for later retrieval
        job_states[config.name] = JobState(
            name=config.name,
            config=config,
            status="running",
            started_at=datetime.utcnow().isoformat(),
            exit_code=None,
        )

    def mock_poll():
        # Mark all running jobs as complete
        completed = []
        for name, state in job_states.items():
            if state.status == "running":
                state.status = "completed"
                state.completed_at = datetime.utcnow().isoformat()
                state.exit_code = 0
                state.logs_path = str(tmp_path / f"{name}.log")
                completed.append(name)
        return completed

    def mock_get_job_state(name: str):
        return job_states.get(name)

    def mock_wait_for_job(name: str, poll_interval_s: float = 1.0):
        # Immediately return completed job state
        return mock_get_job_state(name)

    def mock_get_group_jobs(group: str):
        # Return all jobs in the group (for JobMonitor)
        result = {}
        for name, state in job_states.items():
            if state.config.group == group:
                result[name] = state
        return result

    def mock_get_all_jobs():
        # Return all jobs (for JobMonitor)
        return dict(job_states)

    def mock_cancel_group(group: str):
        # Cancel all jobs in the group
        cancelled = 0
        for _name, state in job_states.items():
            if state.config.group == group and state.status in ("pending", "running"):
                state.status = "cancelled"
                cancelled += 1
        return cancelled

    def mock_get_status_summary(group: str | None = None):
        # Return aggregated status summary
        jobs_to_summarize = {}
        if group:
            for name, state in job_states.items():
                if state.config.group == group:
                    jobs_to_summarize[name] = state
        else:
            jobs_to_summarize = job_states

        completed = sum(1 for js in jobs_to_summarize.values() if js.status == "completed")
        running = sum(1 for js in jobs_to_summarize.values() if js.status == "running")
        pending = sum(1 for js in jobs_to_summarize.values() if js.status == "pending")
        succeeded = sum(1 for js in jobs_to_summarize.values() if js.status == "completed" and js.exit_code == 0)
        failed = sum(1 for js in jobs_to_summarize.values() if js.status == "completed" and js.exit_code != 0)

        job_list = []
        for name, job_state in jobs_to_summarize.items():
            job_dict = {
                "name": name,
                "status": job_state.status,
                "exit_code": job_state.exit_code if job_state.status == "completed" else None,
                "job_id": job_state.job_id,
                "request_id": job_state.request_id,
                "logs_path": job_state.logs_path,
                "metrics": job_state.metrics or {},
                "wandb_url": job_state.wandb_url,
                "checkpoint_uri": job_state.checkpoint_uri,
                "started_at": job_state.started_at,
                "completed_at": job_state.completed_at,
            }
            job_list.append(job_dict)

        return {
            "total": len(jobs_to_summarize),
            "completed": completed,
            "running": running,
            "pending": pending,
            "succeeded": succeeded,
            "failed": failed,
            "jobs": job_list,
        }

    job_manager = Mock(spec=JobManager)
    job_manager.submit.side_effect = mock_submit
    job_manager.poll.side_effect = mock_poll
    job_manager.get_job_state.side_effect = mock_get_job_state
    job_manager.wait_for_job.side_effect = mock_wait_for_job
    job_manager.get_group_jobs.side_effect = mock_get_group_jobs
    job_manager.get_all_jobs.side_effect = mock_get_all_jobs
    job_manager.cancel_group.side_effect = mock_cancel_group
    job_manager.get_status_summary.side_effect = mock_get_status_summary

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

    save_state(state)

    # Load and verify
    loaded = load_state("release_1.0.0")
    assert loaded is not None
    assert loaded.version == "release_1.0.0"
    assert loaded.commit_sha == "abc123"


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
    """Test that already-completed jobs are skipped."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="v1.0.0", created_at="now")

    # Create a job that's already completed
    existing_job_state = JobState(
        name="v1.0.0_already_passed",
        config=JobConfig(name="v1.0.0_already_passed", module="fake"),
        status="completed",
        started_at="now",
        completed_at="now",
        exit_code=0,
        logs_path="/fake/log.txt",
    )

    job_manager = make_mock_job_manager(tmp_path, existing_states={"v1.0.0_already_passed": existing_job_state})
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    task = FakeTask(name="already_passed", exit_code=0)
    runner.run_all([task])

    # Verify job_manager.submit was never called (task was skipped)
    job_manager.submit.assert_not_called()


def test_run_task_retries_failed_when_requested(tmp_path, monkeypatch):
    """Test that failed tasks are retried when retry_failed=True."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="v1.0.0", created_at="now")

    # Create a job that failed
    existing_job_state = JobState(
        name="v1.0.0_failed_task",
        config=JobConfig(name="v1.0.0_failed_task", module="fake"),
        status="completed",
        started_at="now",
        completed_at="now",
        exit_code=1,
        logs_path="/fake/log.txt",
    )

    job_manager = make_mock_job_manager(tmp_path, existing_states={"v1.0.0_failed_task": existing_job_state})
    runner = TaskRunner(state=state, job_manager=job_manager, retry_failed=True, enable_monitor=False)

    task = FakeTask(name="failed_task", exit_code=1)
    runner.run_all([task])

    # Verify job_manager.submit was called (task was retried)
    job_manager.submit.assert_called_once()


def test_run_task_skips_failed_by_default(tmp_path, monkeypatch):
    """Test that failed tasks are skipped when retry_failed=False (default)."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="v1.0.0", created_at="now")

    # Create a job that failed
    existing_job_state = JobState(
        name="v1.0.0_failed_task",
        config=JobConfig(name="v1.0.0_failed_task", module="fake"),
        status="completed",
        started_at="now",
        completed_at="now",
        exit_code=1,
        logs_path="/fake/log.txt",
    )

    job_manager = make_mock_job_manager(tmp_path, existing_states={"v1.0.0_failed_task": existing_job_state})
    runner = TaskRunner(state=state, job_manager=job_manager, retry_failed=False, enable_monitor=False)

    task = FakeTask(name="failed_task", exit_code=1)
    runner.run_all([task])

    # Verify job_manager.submit was never called (task was skipped)
    job_manager.submit.assert_not_called()


def test_run_all_respects_dependencies(tmp_path, monkeypatch):
    """Test that tasks wait for dependencies to complete."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="v1.0.0", created_at="now")
    job_manager = make_mock_job_manager(tmp_path)
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    # Create tasks with dependency
    train_task = FakeTask(name="train", exit_code=0)
    eval_task = FakeTask(name="eval", exit_code=0)
    eval_task.dependency_names = ["train"]

    runner.run_all([train_task, eval_task])

    # Verify both tasks were submitted
    assert job_manager.submit.call_count == 2


def test_dependency_injection(tmp_path, monkeypatch):
    """Test that checkpoint_uri is injected from dependencies."""
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    state = ReleaseState(version="v1.0.0", created_at="now")

    # Create a training job with checkpoint_uri
    train_job_state = JobState(
        name="v1.0.0_train",
        config=JobConfig(name="v1.0.0_train", module="fake"),
        status="completed",
        started_at="now",
        completed_at="now",
        exit_code=0,
        logs_path="/fake/log.txt",
        checkpoint_uri="wandb://run/abc123",
    )

    job_manager = make_mock_job_manager(tmp_path, existing_states={"v1.0.0_train": train_job_state})
    runner = TaskRunner(state=state, job_manager=job_manager, enable_monitor=False)

    # Create tasks
    train_task = FakeTask(name="train", exit_code=0)
    eval_task = FakeTask(name="eval", exit_code=0)
    eval_task.dependency_names = ["train"]

    runner.run_all([train_task, eval_task])

    # Verify checkpoint_uri was injected into eval task
    submitted_config = job_manager.submit.call_args[0][0]
    assert submitted_config.args.get("policy_uri") == "wandb://run/abc123"


def test_jobs_from_same_group_are_included_in_summary(tmp_path, monkeypatch):
    """Jobs with the same group are all included in status summary.

    This verifies that JobManager correctly includes all jobs when filtering by group,
    regardless of their completion status. This is important for the release system
    which uses version as the group identifier.
    """
    monkeypatch.setattr("metta.common.util.fs.get_repo_root", lambda: tmp_path)

    version = "v2025.10.30-111945"

    # Create a completed job (from previous task)
    completed_job_state = JobState(
        name=f"{version}_python_ci",
        config=JobConfig(name=f"{version}_python_ci", module="fake", group=version),
        status="completed",
        started_at="2025-10-30T11:19:00",
        completed_at="2025-10-30T11:19:30",
        exit_code=0,
        logs_path="/fake/log.txt",
    )

    job_manager = make_mock_job_manager(tmp_path, existing_states={f"{version}_python_ci": completed_job_state})

    # Submit a new running job in the same group
    new_task = FakeTask(name="arena_smoke", exit_code=0)
    new_task.job_config.group = version
    new_task.job_config.name = f"{version}_arena_smoke"

    job_manager.submit(new_task.job_config)

    # Get summary for this version/group
    summary = job_manager.get_status_summary(group=version)

    # Both jobs should appear in the summary
    assert summary["total"] == 2, "Should include both jobs in the group"
    assert summary["succeeded"] == 1, "Completed job with exit_code=0"
    assert summary["running"] == 1, "New job is running"

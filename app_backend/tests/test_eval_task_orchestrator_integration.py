import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimResponse,
    TaskResponse,
    TasksResponse,
    TaskUpdateResponse,
)


class MockContainerManager(AbstractContainerManager):
    """Mock implementation of AbstractContainerManager for testing."""

    def __init__(self):
        self.workers: dict[str, WorkerInfo] = {}  # container_id -> WorkerInfo
        self.cleanup_calls: list[str] = []
        self.start_worker_calls: list[tuple[str, str, str]] = []

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        return list(self.workers.values())

    def start_worker_container(
        self, git_hash: str, backend_url: str, docker_image: str, machine_token: str | None = None
    ) -> WorkerInfo:
        worker = WorkerInfo(
            git_hash=git_hash,
            container_id=f"container_{git_hash}_{len(self.start_worker_calls)}",
            container_name=f"worker_{git_hash}_{len(self.start_worker_calls)}",
        )
        self.workers[worker.container_id] = worker
        self.start_worker_calls.append((git_hash, backend_url, docker_image))
        return worker

    def cleanup_container(self, container_id: str) -> None:
        self.cleanup_calls.append(container_id)
        if container_id in self.workers:
            del self.workers[container_id]


class MockEvalTaskClient:
    """Mock implementation of EvalTaskClient for testing."""

    def __init__(self):
        self.tasks: dict[uuid.UUID, TaskResponse] = {}
        self.claim_responses: list[TaskClaimResponse] = []
        self.claim_calls: list[Any] = []
        self.update_calls: list[Any] = []
        self.get_latest_calls: list[str] = []

    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        from copy import deepcopy

        available = [deepcopy(t) for t in self.tasks.values() if t.status == "unprocessed" and not t.assignee]
        return TasksResponse(tasks=available[:limit])

    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        from copy import deepcopy

        # Return copies of tasks to avoid modifying the orchestrator's task objects
        claimed = [deepcopy(t) for t in self.tasks.values() if t.status == "unprocessed" and t.assignee]
        if assignee:
            claimed = [t for t in claimed if t.assignee == assignee]
        return TasksResponse(tasks=claimed)

    async def claim_tasks(self, request: Any) -> TaskClaimResponse:
        self.claim_calls.append(request)
        if self.claim_responses:
            return self.claim_responses.pop(0)

        # Default behavior: claim all requested tasks
        claimed = []
        for task_id in request.tasks:
            if task_id in self.tasks and not self.tasks[task_id].assignee:
                self.tasks[task_id].assignee = request.assignee
                self.tasks[task_id].assigned_at = datetime.now(timezone.utc)
                claimed.append(task_id)
        return TaskClaimResponse(claimed=claimed)

    async def update_task_status(self, request: Any) -> TaskUpdateResponse:
        self.update_calls.append(request)
        statuses = {}
        for task_id, status_update in request.updates.items():
            if task_id in self.tasks:
                self.tasks[task_id].status = status_update.status
                if hasattr(status_update, "attributes") and status_update.attributes:
                    self.tasks[task_id].attributes.update(status_update.attributes)
                if hasattr(status_update, "clear_assignee") and status_update.clear_assignee:
                    self.tasks[task_id].assignee = None
                statuses[task_id] = status_update.status
        return TaskUpdateResponse(statuses=statuses)

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> TaskResponse | None:
        self.get_latest_calls.append(assignee)
        tasks = [t for t in self.tasks.values() if t.assignee == assignee and t.assigned_at]
        return max(tasks, key=lambda t: t.assigned_at or datetime.min) if tasks else None

    async def close(self):
        pass


@pytest.fixture
def mock_container_manager():
    return MockContainerManager()


@pytest.fixture
def mock_task_client():
    return MockEvalTaskClient()


@pytest.fixture
def orchestrator(mock_container_manager, mock_task_client, monkeypatch):
    """Create an orchestrator with mocked dependencies."""
    orchestrator = EvalTaskOrchestrator(
        backend_url="http://test-backend",
        machine_token="test-machine-token",
        docker_image="test-image:latest",
        poll_interval=1.0,
        worker_idle_timeout=60.0,
        container_manager=mock_container_manager,
    )
    # Replace the task client with our mock
    monkeypatch.setattr(orchestrator, "_task_client", mock_task_client)
    return orchestrator


@pytest.fixture
def sample_task_factory():
    """Factory for creating test TaskResponse objects."""

    def create_task(
        task_id: uuid.UUID | None = None,
        git_hash: str = "abc123",
        status: str = "unprocessed",
        assignee: str | None = None,
        assigned_at: datetime | None = None,
        workers_spawned: int = 0,
        retries: int | None = None,
    ) -> TaskResponse:
        if task_id is None:
            task_id = uuid.uuid4()

        return TaskResponse(
            id=task_id,
            policy_id=uuid.uuid4(),
            sim_suite="test_suite",
            status=status,  # type: ignore
            assigned_at=assigned_at,
            assignee=assignee,
            created_at=datetime.now(timezone.utc),
            attributes={"git_hash": git_hash, "workers_spawned": workers_spawned},
            policy_name="test_policy",
            retries=retries if retries is not None else workers_spawned,
        )

    return create_task


@pytest.fixture
def sample_worker_factory():
    """Factory for creating test WorkerInfo objects."""

    def create_worker(
        git_hash: str = "abc123",
        container_id: str | None = None,
        container_name: str | None = None,
    ) -> WorkerInfo:
        if container_id is None:
            container_id = f"container_{git_hash}"
        if container_name is None:
            container_name = f"worker_{git_hash}"

        return WorkerInfo(
            git_hash=git_hash,
            container_id=container_id,
            container_name=container_name,
        )

    return create_worker


class TestEvalTaskOrchestratorIntegration:
    """Integration tests for the EvalTaskOrchestrator."""

    @pytest.mark.asyncio
    async def test_claim_task_success(self, orchestrator, mock_task_client, sample_task_factory, sample_worker_factory):
        """Test successful task claiming."""
        task = sample_task_factory()
        worker = sample_worker_factory()
        mock_task_client.tasks[task.id] = task

        result = await orchestrator._attempt_claim_task(task, worker)

        assert result is True
        assert task.assignee == worker.container_name

    @pytest.mark.asyncio
    async def test_claim_task_failure(self, orchestrator, mock_task_client, sample_task_factory, sample_worker_factory):
        """Test handling when task claim fails."""
        task = sample_task_factory()
        worker = sample_worker_factory()
        mock_task_client.claim_responses = [TaskClaimResponse(claimed=[])]

        result = await orchestrator._attempt_claim_task(task, worker)

        assert result is False

    @pytest.mark.asyncio
    async def test_task_timeout_kills_worker_and_unclaims_task(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory, sample_worker_factory
    ):
        """Test that tasks running for more than 10 minutes are killed and unclaimed."""
        # Create a task that has been running for 11 minutes
        task = sample_task_factory(
            git_hash="timeout_hash",
            assignee="container_timeout",
            assigned_at=datetime.now(timezone.utc) - timedelta(minutes=11),
            retries=1,
        )
        mock_task_client.tasks[task.id] = task

        worker = sample_worker_factory(
            git_hash="timeout_hash",
            container_id="container_timeout_id",
            container_name="container_timeout",
        )
        mock_container_manager.workers[worker.container_id] = worker

        # Verify setup is correct
        assert task.id in mock_task_client.tasks
        assert task.status == "unprocessed"
        assert task.assignee == "container_timeout"
        assert worker.container_id in mock_container_manager.workers

        await orchestrator.run_cycle()

        # Verify worker was killed
        cleanup_count = len(mock_container_manager.cleanup_calls)
        assert cleanup_count == 1, (
            f"Expected 1 cleanup call, got {cleanup_count}: {mock_container_manager.cleanup_calls}"
        )
        assert mock_container_manager.cleanup_calls[0] == "container_timeout_id"

        # Verify task was unclaimed
        assert len(mock_task_client.update_calls) >= 1, (
            f"Expected at least 1 update call, got {len(mock_task_client.update_calls)}"
        )
        # Find the update for our task
        task_update = None
        for update_call in mock_task_client.update_calls:
            if task.id in update_call.updates:
                task_update = update_call.updates[task.id]
                break
        assert task_update is not None, "Task update not found"
        assert task_update.status == "unprocessed"
        assert task_update.clear_assignee is True
        assert "unassign_reason_1" in task_update.attributes

    @pytest.mark.asyncio
    async def test_max_retries_marks_task_error(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory, sample_worker_factory
    ):
        """Test task marked as error after max retries."""
        # Create task with max retries that has timed out
        task = sample_task_factory(
            git_hash="retry_hash",
            assignee="worker_retry",
            assigned_at=datetime.now(timezone.utc) - timedelta(minutes=11),
            retries=3,
        )
        mock_task_client.tasks[task.id] = task

        worker = sample_worker_factory(
            git_hash="retry_hash",
            container_id="container_retry_id",
            container_name="worker_retry",
        )
        mock_container_manager.workers[worker.container_id] = worker

        await orchestrator.run_cycle()

        # Verify task marked as error
        assert len(mock_task_client.update_calls) == 1
        update_call = mock_task_client.update_calls[0]
        assert task.id in update_call.updates
        assert update_call.updates[task.id].status == "error"
        assert update_call.updates[task.id].attributes["reason"] == "max_retries_exceeded"

    @pytest.mark.asyncio
    async def test_no_git_hash_marks_task_error(self, orchestrator, mock_task_client):
        """Test tasks without git_hash are marked as error."""
        task = TaskResponse(
            id=uuid.uuid4(),
            policy_id=uuid.uuid4(),
            sim_suite="test",
            status="unprocessed",
            assigned_at=None,
            assignee=None,
            created_at=datetime.now(timezone.utc),
            attributes={},
            policy_name="test",
            retries=0,
        )
        mock_task_client.tasks[task.id] = task

        await orchestrator.run_cycle()

        assert len(mock_task_client.update_calls) == 1
        update_call = mock_task_client.update_calls[0]
        assert task.id in update_call.updates
        assert update_call.updates[task.id].status == "error"
        assert update_call.updates[task.id].attributes["reason"] == "no_git_hash"

    @pytest.mark.asyncio
    async def test_assign_task_to_existing_worker(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory, sample_worker_factory
    ):
        """Test assigning task to existing idle worker."""
        task = sample_task_factory(git_hash="existing_hash")
        mock_task_client.tasks[task.id] = task

        worker = sample_worker_factory(
            git_hash="existing_hash",
            container_id="container_existing",
            container_name="worker_existing",
        )
        mock_container_manager.workers[worker.container_id] = worker

        await orchestrator.run_cycle()

        # Should use existing worker, not spawn new one
        assert len(mock_container_manager.start_worker_calls) == 0
        assert len(mock_task_client.claim_calls) == 1
        assert task.id in mock_task_client.claim_calls[0].tasks
        assert mock_task_client.claim_calls[0].assignee == "worker_existing"

    @pytest.mark.asyncio
    async def test_spawn_new_worker_when_none_exist(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory
    ):
        """Test spawning new worker when no workers exist for git hash."""
        task = sample_task_factory(git_hash="new_hash")
        mock_task_client.tasks[task.id] = task

        await orchestrator.run_cycle()

        assert len(mock_container_manager.start_worker_calls) == 1
        assert mock_container_manager.start_worker_calls[0][0] == "new_hash"
        assert len(mock_task_client.claim_calls) == 1
        assert task.id in mock_task_client.claim_calls[0].tasks

    @pytest.mark.asyncio
    async def test_all_workers_busy_skip_assignment(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory, sample_worker_factory
    ):
        """Test that tasks are skipped when all workers for git hash are busy."""
        # Create two unassigned tasks
        task1 = sample_task_factory(git_hash="busy_hash")
        task2 = sample_task_factory(git_hash="busy_hash")
        mock_task_client.tasks[task1.id] = task1
        mock_task_client.tasks[task2.id] = task2

        # Create a busy worker
        assigned_task = sample_task_factory(
            git_hash="busy_hash",
            assignee="worker_busy",
            assigned_at=datetime.now(timezone.utc),
        )
        mock_task_client.tasks[assigned_task.id] = assigned_task

        worker = sample_worker_factory(
            git_hash="busy_hash",
            container_id="container_busy",
            container_name="worker_busy",
        )
        mock_container_manager.workers[worker.container_id] = worker

        await orchestrator.run_cycle()

        # Should not spawn or claim since worker is busy
        assert len(mock_container_manager.start_worker_calls) == 0
        assert len(mock_task_client.claim_calls) == 0

    @pytest.mark.asyncio
    async def test_cleanup_idle_workers(
        self, orchestrator, mock_container_manager, mock_task_client, sample_worker_factory, sample_task_factory
    ):
        """Test cleanup of workers idle beyond timeout."""
        worker = sample_worker_factory(
            git_hash="idle_hash",
            container_id="container_idle",
            container_name="worker_idle",
        )
        mock_container_manager.workers[worker.container_id] = worker

        # Create old completed task
        old_task = sample_task_factory(
            git_hash="idle_hash",
            assignee="worker_idle",
            assigned_at=datetime.now(timezone.utc) - timedelta(hours=2),
            status="done",
        )
        mock_task_client.tasks[old_task.id] = old_task

        await orchestrator.run_cycle()

        assert len(mock_container_manager.cleanup_calls) == 1
        assert mock_container_manager.cleanup_calls[0] == "container_idle"

    @pytest.mark.asyncio
    async def test_multiple_git_hashes_concurrent_processing(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory
    ):
        """Test handling multiple git hashes simultaneously."""
        tasks = [
            sample_task_factory(git_hash="hash1"),
            sample_task_factory(git_hash="hash2"),
            sample_task_factory(git_hash="hash3"),
        ]
        for task in tasks:
            mock_task_client.tasks[task.id] = task

        await orchestrator.run_cycle()

        # One worker per unique git hash
        assert len(mock_container_manager.start_worker_calls) == 3
        spawned_hashes = {call[0] for call in mock_container_manager.start_worker_calls}
        assert spawned_hashes == {"hash1", "hash2", "hash3"}
        assert len(mock_task_client.claim_calls) == 3

    @pytest.mark.asyncio
    async def test_full_cycle_with_mixed_scenarios(
        self, orchestrator, mock_container_manager, mock_task_client, sample_task_factory, sample_worker_factory
    ):
        """End-to-end test with various scenarios in one cycle."""
        # Setup tasks
        timeout_task = sample_task_factory(
            git_hash="timeout_hash",
            assignee="container_timeout",
            assigned_at=datetime.now(timezone.utc) - timedelta(minutes=11),
            retries=1,
        )
        new_task = sample_task_factory(git_hash="existing_hash")
        spawn_task = sample_task_factory(git_hash="new_hash")
        no_hash_task = TaskResponse(
            id=uuid.uuid4(),
            policy_id=uuid.uuid4(),
            sim_suite="test",
            status="unprocessed",
            assigned_at=None,
            assignee=None,
            created_at=datetime.now(timezone.utc),
            attributes={},
            policy_name="test",
            retries=0,
        )

        for task in [timeout_task, new_task, spawn_task, no_hash_task]:
            mock_task_client.tasks[task.id] = task

        # Setup workers
        timeout_worker = sample_worker_factory(
            git_hash="timeout_hash",
            container_id="container_timeout_id",
            container_name="container_timeout",
        )
        existing_worker = sample_worker_factory(
            git_hash="existing_hash",
            container_id="container_existing",
        )
        mock_container_manager.workers = {
            timeout_worker.container_id: timeout_worker,
            existing_worker.container_id: existing_worker,
        }

        await orchestrator.run_cycle()

        # Verify results
        assert "container_timeout_id" in mock_container_manager.cleanup_calls
        assert len(mock_task_client.update_calls) == 2  # timeout task + no_hash task
        assert len(mock_container_manager.start_worker_calls) == 2  # timeout_hash + new_hash
        spawned_hashes = {call[0] for call in mock_container_manager.start_worker_calls}
        assert spawned_hashes == {"timeout_hash", "new_hash"}
        assert len(mock_task_client.claim_calls) == 3  # timeout + existing + new

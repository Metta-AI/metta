import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskClaimResponse,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
    TasksResponse,
    GitHashesResponse,
)


@pytest.fixture
def mock_container_manager():
    """Mock container manager for testing."""
    manager = Mock(spec=AbstractContainerManager)
    manager.discover_alive_workers = AsyncMock()
    manager.cleanup_container = Mock()
    manager.start_worker_container = Mock()
    return manager


@pytest.fixture
def mock_task_client():
    """Mock task client for testing."""
    client = Mock(spec=EvalTaskClient)
    client.get_available_tasks = AsyncMock()
    client.get_claimed_tasks = AsyncMock()
    client.claim_tasks = AsyncMock()
    client.get_git_hashes_for_workers = AsyncMock()
    client.update_task_status = AsyncMock()
    return client


@pytest.fixture
def orchestrator(mock_container_manager, mock_task_client):
    """Create orchestrator with mocked dependencies."""
    orch = EvalTaskOrchestrator(
        backend_url="http://test.backend",
        machine_token="test-token",
        docker_image="test-image",
        poll_interval=1.0,
        worker_idle_timeout=60.0,
        max_workers=3,
        container_manager=mock_container_manager,
    )
    orch._task_client = mock_task_client
    return orch


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return TaskResponse(
        id=uuid.uuid4(),
        policy_id=uuid.uuid4(),
        sim_suite="test_suite",
        status="unprocessed",
        assigned_at=None,
        assignee=None,
        created_at=datetime.now(timezone.utc),
        attributes={"git_hash": "abc123"},
        policy_name="test_policy",
        retries=0,
        user_id="test_user",
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_worker():
    """Create a sample worker for testing."""
    return WorkerInfo(
        git_hashes=["abc123", "def456"],
        container_id="container123",
        container_name="worker1",
        assigned_task=None,
    )


class TestEvalTaskOrchestrator:
    @pytest.mark.asyncio
    async def test_attempt_claim_task_success(self, orchestrator, sample_task, sample_worker):
        """Test successful task claiming."""
        orchestrator._task_client.claim_tasks.return_value = TaskClaimResponse(claimed=[sample_task.id])
        
        result = await orchestrator._attempt_claim_task(sample_task, sample_worker)
        
        assert result is True
        orchestrator._task_client.claim_tasks.assert_called_once_with(
            TaskClaimRequest(tasks=[sample_task.id], assignee=sample_worker.container_name)
        )

    @pytest.mark.asyncio
    async def test_attempt_claim_task_failure(self, orchestrator, sample_task, sample_worker):
        """Test failed task claiming."""
        orchestrator._task_client.claim_tasks.return_value = TaskClaimResponse(claimed=[])
        
        result = await orchestrator._attempt_claim_task(sample_task, sample_worker)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_available_workers(self, orchestrator, sample_worker, sample_task):
        """Test getting available workers with task assignments."""
        # Setup
        claimed_task = TaskResponse(**sample_task.model_dump())
        claimed_task.assignee = sample_worker.container_name
        
        orchestrator._container_manager.discover_alive_workers.return_value = [sample_worker]
        orchestrator._task_client.get_git_hashes_for_workers.return_value = GitHashesResponse(
            git_hashes={sample_worker.container_name: ["abc123", "def456"]}
        )
        
        # Execute
        result = await orchestrator._get_available_workers([claimed_task])
        
        # Verify
        assert len(result) == 1
        worker = result[sample_worker.container_name]
        assert worker.assigned_task == claimed_task
        assert worker.git_hashes == ["abc123", "def456"]

    @pytest.mark.asyncio
    async def test_kill_dead_workers_and_tasks_timeout(self, orchestrator, sample_task, sample_worker):
        """Test killing overdue tasks and workers."""
        # Setup overdue task
        overdue_task = TaskResponse(**sample_task.model_dump())
        overdue_task.assignee = sample_worker.container_name
        overdue_task.assigned_at = datetime.now(timezone.utc) - timedelta(minutes=15)
        overdue_task.retries = 1
        
        alive_workers = {sample_worker.container_name: sample_worker}
        
        # Execute
        await orchestrator._kill_dead_workers_and_tasks([overdue_task], alive_workers)
        
        # Verify task status update
        orchestrator._task_client.update_task_status.assert_called_once()
        call_args = orchestrator._task_client.update_task_status.call_args[0][0]
        assert call_args.updates[overdue_task.id].status == "unprocessed"
        assert call_args.updates[overdue_task.id].clear_assignee is True
        
        # Verify worker cleanup
        orchestrator._container_manager.cleanup_container.assert_called_once_with(sample_worker.container_id)
        assert sample_worker.container_name not in alive_workers

    @pytest.mark.asyncio
    async def test_kill_dead_workers_max_retries(self, orchestrator, sample_task, sample_worker):
        """Test task marked as error when max retries exceeded."""
        # Setup task with max retries
        overdue_task = TaskResponse(**sample_task.model_dump())
        overdue_task.assignee = sample_worker.container_name
        overdue_task.assigned_at = datetime.now(timezone.utc) - timedelta(minutes=15)
        overdue_task.retries = 3
        
        alive_workers = {sample_worker.container_name: sample_worker}
        
        # Execute
        await orchestrator._kill_dead_workers_and_tasks([overdue_task], alive_workers)
        
        # Verify task marked as error
        call_args = orchestrator._task_client.update_task_status.call_args[0][0]
        assert call_args.updates[overdue_task.id].status == "error"
        assert call_args.updates[overdue_task.id].attributes["reason"] == "max_retries_exceeded"

    @pytest.mark.asyncio
    async def test_assign_task_to_worker_matching_hash(self, orchestrator, sample_task, sample_worker):
        """Test task assignment prioritizing worker's git hashes."""
        available_tasks = {"abc123": [sample_task], "def456": []}
        
        with patch.object(orchestrator, '_attempt_claim_task', return_value=True) as mock_claim:
            await orchestrator._assign_task_to_worker(sample_worker, available_tasks)
            
            mock_claim.assert_called_once_with(sample_task, sample_worker)
            assert len(available_tasks["abc123"]) == 0

    @pytest.mark.asyncio
    async def test_assign_task_to_worker_fallback(self, orchestrator, sample_task, sample_worker):
        """Test task assignment falls back to any available task."""
        other_task = TaskResponse(**sample_task.model_dump())
        other_task.id = uuid.uuid4()
        available_tasks = {"xyz789": [other_task]}
        
        with patch.object(orchestrator, '_attempt_claim_task', return_value=True) as mock_claim:
            await orchestrator._assign_task_to_worker(sample_worker, available_tasks)
            
            mock_claim.assert_called_once_with(other_task, sample_worker)

    @pytest.mark.asyncio
    async def test_assign_tasks_to_workers(self, orchestrator, sample_worker, sample_task):
        """Test assigning tasks to available workers."""
        # Setup worker without assigned task
        sample_worker.assigned_task = None
        alive_workers = {sample_worker.container_name: sample_worker}
        
        orchestrator._task_client.get_available_tasks.return_value = TasksResponse(tasks=[sample_task])
        
        with patch.object(orchestrator, '_assign_task_to_worker') as mock_assign:
            await orchestrator._assign_tasks_to_workers(alive_workers)
            
            mock_assign.assert_called_once_with(sample_worker, {"abc123": [sample_task]})

    @pytest.mark.asyncio
    async def test_start_new_workers(self, orchestrator, sample_worker):
        """Test starting new workers when below max capacity."""
        alive_workers = {sample_worker.container_name: sample_worker}  # Only 1 worker
        
        await orchestrator._start_new_workers(alive_workers)
        
        # Should start 2 more workers (max_workers=3, current=1)
        assert orchestrator._container_manager.start_worker_container.call_count == 2
        
        # Verify call arguments
        expected_call_args = {
            "backend_url": "http://test.backend",
            "docker_image": "test-image",
            "machine_token": "test-token",
        }
        for call in orchestrator._container_manager.start_worker_container.call_args_list:
            assert call.kwargs == expected_call_args

    @pytest.mark.asyncio
    async def test_run_cycle_integration(self, orchestrator, sample_worker, sample_task):
        """Test complete run cycle integration."""
        # Setup mocks
        orchestrator._task_client.get_claimed_tasks.return_value = TasksResponse(tasks=[])
        orchestrator._container_manager.discover_alive_workers.return_value = [sample_worker]
        orchestrator._task_client.get_git_hashes_for_workers.return_value = GitHashesResponse(
            git_hashes={sample_worker.container_name: ["abc123"]}
        )
        orchestrator._task_client.get_available_tasks.return_value = TasksResponse(tasks=[sample_task])
        
        with patch.object(orchestrator, '_assign_task_to_worker') as mock_assign:
            await orchestrator.run_cycle()
            
            # Verify all main steps were called
            orchestrator._task_client.get_claimed_tasks.assert_called_once()
            orchestrator._container_manager.discover_alive_workers.assert_called_once()
            orchestrator._task_client.get_available_tasks.assert_called_once()
            mock_assign.assert_called_once()
            
            # Should start 2 more workers (max=3, current=1)
            assert orchestrator._container_manager.start_worker_container.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_exception_handling(self, orchestrator):
        """Test that run loop handles exceptions gracefully."""
        # Mock run_cycle to raise exception once, then work normally
        call_count = 0
        original_run_cycle = orchestrator.run_cycle
        
        async def failing_run_cycle():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test exception")
            # Stop the loop after second call
            raise KeyboardInterrupt()
        
        orchestrator.run_cycle = failing_run_cycle
        
        # Should not crash, but should handle exception
        with pytest.raises(KeyboardInterrupt):
            await orchestrator.run()
        
        assert call_count == 2  # Exception on first call, KeyboardInterrupt on second

    def test_init_defaults(self):
        """Test orchestrator initialization with defaults."""
        orch = EvalTaskOrchestrator(
            backend_url="http://test.backend",
            machine_token="test-token"
        )
        
        assert orch._backend_url == "http://test.backend"
        assert orch._machine_token == "test-token"
        assert orch._docker_image == "metta-policy-evaluator-local:latest"
        assert orch._poll_interval == 5.0
        assert orch._worker_idle_timeout == 1200.0
        assert orch._max_workers == 5
        assert orch._task_client is not None
        assert orch._container_manager is not None

    @pytest.mark.asyncio
    async def test_no_available_tasks(self, orchestrator, sample_worker):
        """Test behavior when no tasks are available."""
        alive_workers = {sample_worker.container_name: sample_worker}
        orchestrator._task_client.get_available_tasks.return_value = TasksResponse(tasks=[])
        
        with patch.object(orchestrator, '_assign_task_to_worker') as mock_assign:
            await orchestrator._assign_tasks_to_workers(alive_workers)
            
            # Should call assign_task_to_worker but with empty task dict
            mock_assign.assert_called_once_with(sample_worker, {})

    @pytest.mark.asyncio
    async def test_worker_already_has_task(self, orchestrator, sample_worker, sample_task):
        """Test that workers with assigned tasks don't get new tasks."""
        sample_worker.assigned_task = sample_task
        alive_workers = {sample_worker.container_name: sample_worker}
        
        orchestrator._task_client.get_available_tasks.return_value = TasksResponse(tasks=[sample_task])
        
        with patch.object(orchestrator, '_assign_task_to_worker') as mock_assign:
            await orchestrator._assign_tasks_to_workers(alive_workers)
            
            # Should not assign task to worker that already has one
            mock_assign.assert_not_called()
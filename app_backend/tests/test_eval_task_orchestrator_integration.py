import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.eval_task_worker import AbstractTaskExecutor, EvalTaskWorker, TaskResult
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import (
    TaskCreateRequest,
    TaskFilterParams,
)
from metta.app_backend.test_support.client_adapter import TestClientAdapter
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager
from metta.app_backend.worker_managers.worker import Worker


class SuccessTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: EvalTaskRow) -> TaskResult:
        return TaskResult(success=True)


class FailureTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: EvalTaskRow) -> TaskResult:
        raise Exception("Failed task")


class TestEvalTaskOrchestratorIntegration:
    """Integration tests for EvalTaskOrchestrator with one-worker-per-task model."""

    @pytest.fixture
    def eval_task_client(self, test_client: TestClient) -> EvalTaskClient:
        """Create an eval task client for testing."""
        client = EvalTaskClient(backend_url=str(test_client.base_url), machine_token=None)
        client._http_client = TestClientAdapter.with_softmax_user(test_client)
        return client

    def create_success_worker(self, worker_name: str, eval_task_client: EvalTaskClient) -> EvalTaskWorker:
        return EvalTaskWorker(
            client=eval_task_client,
            assignee=worker_name,
            task_executor=SuccessTaskExecutor(),
            poll_interval=0.5,
        )

    @pytest.fixture
    def success_thread_manager(self, eval_task_client: EvalTaskClient) -> ThreadWorkerManager:
        """Create a ThreadWorkerManager with success workers."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, eval_task_client)

        return ThreadWorkerManager(create_worker=create_worker)

    @pytest.fixture
    def failure_thread_manager(self, eval_task_client: EvalTaskClient) -> ThreadWorkerManager:
        """Create a ThreadWorkerManager with failure workers."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return EvalTaskWorker(
                client=eval_task_client,
                assignee=worker_name,
                task_executor=FailureTaskExecutor(),
                poll_interval=0.5,
            )

        return ThreadWorkerManager(create_worker=create_worker)

    @pytest.mark.asyncio
    async def test_spawns_worker_per_task(self, eval_task_client: EvalTaskClient):
        """Test that orchestrator spawns one worker per available task."""
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        worker_counter = [0]

        def mock_start_worker(num_cpus_request: int = 3, memory_request: int = 12):
            worker_counter[0] += 1
            return f"worker-{worker_counter[0]}"

        mock_worker_manager.start_worker = mock_start_worker
        mock_worker_manager.cleanup_worker = Mock()

        # Mock claim_tasks to return success
        original_claim = eval_task_client.claim_tasks
        eval_task_client.claim_tasks = Mock(side_effect=lambda req: Mock(claimed=req.tasks))

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
        )

        # Create 3 tasks
        test_hash = f"spawn_test_{uuid.uuid4().hex[:8]}"
        for i in range(3):
            eval_task_client.create_task(
                TaskCreateRequest(
                    command="metta evaluate test",
                    git_hash=f"{test_hash}_{i}",
                    attributes={"test": "spawn"},
                )
            )

        # Run one cycle
        await orchestrator.run_cycle()

        # Should have started 3 workers
        assert worker_counter[0] == 3

        # Restore original
        eval_task_client.claim_tasks = original_claim

    @pytest.mark.asyncio
    async def test_kills_worker_on_claim_failure(self, eval_task_client: EvalTaskClient):
        """Test that worker is killed if task claim fails."""
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock(return_value="worker-1")
        mock_worker_manager.cleanup_worker = Mock()

        # Mock claim_tasks to return failure (empty claimed list)
        original_claim = eval_task_client.claim_tasks
        eval_task_client.claim_tasks = Mock(return_value=Mock(claimed=[]))

        # Also mock get_available_tasks to return only our task
        original_get_available = eval_task_client.get_available_tasks
        test_task = Mock(id=999, attributes={"parallelism": 1})
        eval_task_client.get_available_tasks = Mock(return_value=Mock(tasks=[test_task]))

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
        )

        await orchestrator.run_cycle()

        # Worker should have been started then cleaned up
        mock_worker_manager.start_worker.assert_called_once()
        mock_worker_manager.cleanup_worker.assert_called_once_with("worker-1")

        eval_task_client.claim_tasks = original_claim
        eval_task_client.get_available_tasks = original_get_available

    @pytest.mark.asyncio
    async def test_cleanup_idle_workers(self, eval_task_client: EvalTaskClient):
        """Test that workers with no assigned task are killed."""
        # Create worker that appears alive but has no task
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(
            return_value=[Worker(name="idle-worker", status="Running")]
        )
        mock_worker_manager.start_worker = Mock(return_value="new-worker")
        mock_worker_manager.cleanup_worker = Mock()

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
        )

        # No tasks exist, so idle-worker should be cleaned up
        await orchestrator.run_cycle()

        mock_worker_manager.cleanup_worker.assert_called_once_with("idle-worker")

    @pytest.mark.asyncio
    async def test_fail_task_when_worker_dead(self, eval_task_client: EvalTaskClient):
        """Test that tasks are failed when their worker dies.

        Note: system_error triggers automatic retry (up to 3 attempts), so we verify
        that the attempt was recorded with the correct error and the task is ready for retry.
        """
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        # Return no alive workers - simulates worker dying
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock(return_value="dead-worker")
        mock_worker_manager.cleanup_worker = Mock()

        # Create the task first
        test_hash = f"dead_worker_{uuid.uuid4().hex[:8]}"
        task = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate test",
                git_hash=test_hash,
                attributes={"test": "dead_worker"},
            )
        )

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
        )

        # First cycle: spawn worker and claim task (uses real claim_tasks)
        await orchestrator.run_cycle()

        # Verify task was claimed
        filters = TaskFilterParams(git_hash=test_hash, limit=10)
        tasks = eval_task_client.get_all_tasks(filters=filters)
        claimed_task = next((t for t in tasks.tasks if t.id == task.id), None)
        assert claimed_task is not None
        assert claimed_task.assignee == "dead-worker"

        # Mock get_available_tasks to return empty so the failed task doesn't get re-claimed
        original_get_available = eval_task_client.get_available_tasks
        eval_task_client.get_available_tasks = Mock(return_value=Mock(tasks=[]))

        # Now the worker "dies" - discover still returns empty
        # Next cycle should fail the task because worker is dead
        await orchestrator.run_cycle()

        # Restore
        eval_task_client.get_available_tasks = original_get_available

        # Check that the attempt was recorded as system_error with worker_dead reason
        # Note: system_error on first attempt triggers a retry, so task goes back to unprocessed
        attempts = eval_task_client.get_task_attempts(task.id)
        assert len(attempts.attempts) >= 1
        failed_attempt = attempts.attempts[0]
        assert failed_attempt.status == "system_error"
        assert failed_attempt.status_details is not None
        assert failed_attempt.status_details.get("unassign_reason") == "worker_dead"

        # Task should be back to unprocessed (ready for retry)
        tasks = eval_task_client.get_all_tasks(filters=filters)
        retried_task = next((t for t in tasks.tasks if t.id == task.id), None)
        assert retried_task is not None
        assert retried_task.status == "unprocessed"

    @pytest.mark.asyncio
    async def test_successful_task_processing(
        self, eval_task_client: EvalTaskClient, success_thread_manager: ThreadWorkerManager
    ):
        """Test end-to-end successful task processing with real workers."""
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_thread_manager,
            poll_interval=0.5,
            task_timeout_minutes=5.0,
        )

        # Create a test task
        test_hash = f"success_test_{uuid.uuid4().hex[:8]}"
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash=test_hash,
                attributes={"test": "integration"},
            )
        )
        task_id = task_response.id

        try:
            # Run orchestrator cycle to spawn worker and assign task
            await orchestrator.run_cycle()

            # Poll for completion
            processed_task = None
            for _ in range(10):
                await asyncio.sleep(1)

                filters = TaskFilterParams(git_hash=test_hash, limit=10)
                all_tasks = eval_task_client.get_all_tasks(filters=filters)
                processed_task = next((t for t in all_tasks.tasks if t.id == task_id), None)

                if processed_task and processed_task.status == "done":
                    break

            assert processed_task is not None
            assert processed_task.status == "done"

        finally:
            success_thread_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_killed_after_task_completes(
        self, eval_task_client: EvalTaskClient, success_thread_manager: ThreadWorkerManager
    ):
        """Test that workers are killed after their task completes (idle worker cleanup)."""
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_thread_manager,
            poll_interval=0.5,
            task_timeout_minutes=5.0,
        )

        # Create a test task
        test_hash = f"worker_cleanup_{uuid.uuid4().hex[:8]}"
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate test",
                git_hash=test_hash,
                attributes={"test": "worker_cleanup"},
            )
        )
        task_id = task_response.id

        try:
            # Run orchestrator cycle to spawn worker and assign task
            await orchestrator.run_cycle()

            # Worker should be alive
            alive_workers = await success_thread_manager.discover_alive_workers()
            assert len(alive_workers) == 1

            # Wait for task to complete
            for _ in range(10):
                await asyncio.sleep(1)
                filters = TaskFilterParams(git_hash=test_hash, limit=10)
                all_tasks = eval_task_client.get_all_tasks(filters=filters)
                task = next((t for t in all_tasks.tasks if t.id == task_id), None)
                if task and task.status == "done":
                    break

            assert task is not None
            assert task.status == "done"

            # Run another orchestrator cycle - should clean up the idle worker
            await orchestrator.run_cycle()

            # Worker should be killed (no longer has an assigned task)
            alive_workers = await success_thread_manager.discover_alive_workers()
            assert len(alive_workers) == 0, f"Expected 0 workers, found {len(alive_workers)}"

        finally:
            success_thread_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_failed_task_processing(
        self, eval_task_client: EvalTaskClient, failure_thread_manager: ThreadWorkerManager
    ):
        """Test that failed tasks are marked as error."""
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=failure_thread_manager,
            poll_interval=0.5,
            task_timeout_minutes=5.0,
        )

        # Create a test task
        test_hash = f"failure_test_{uuid.uuid4().hex[:8]}"
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash=test_hash,
                attributes={"test": "failure"},
            )
        )
        task_id = task_response.id

        try:
            # Run orchestrator cycle
            await orchestrator.run_cycle()

            # Poll for failure
            failed_task = None
            for _ in range(5):
                await asyncio.sleep(1)

                filters = TaskFilterParams(git_hash=test_hash, limit=10)
                all_tasks = eval_task_client.get_all_tasks(filters=filters)
                failed_task = next((t for t in all_tasks.tasks if t.id == task_id), None)

                if failed_task and failed_task.status == "error":
                    break

            assert failed_task is not None
            assert failed_task.status == "error"

        finally:
            failure_thread_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_multiple_tasks_spawn_multiple_workers(
        self, eval_task_client: EvalTaskClient, success_thread_manager: ThreadWorkerManager
    ):
        """Test that multiple tasks spawn multiple workers concurrently."""
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_thread_manager,
            poll_interval=0.5,
            task_timeout_minutes=10.0,
        )

        # Create multiple tasks
        test_hash = f"multi_test_{uuid.uuid4().hex[:8]}"
        task_ids = []
        for i in range(3):
            task = eval_task_client.create_task(
                TaskCreateRequest(
                    command="metta evaluate test",
                    git_hash=f"{test_hash}_{i}",
                    attributes={"test": "multi"},
                )
            )
            task_ids.append(task.id)

        try:
            # Run orchestrator - should spawn 3 workers
            await orchestrator.run_cycle()

            # Check that workers were started
            alive_workers = await success_thread_manager.discover_alive_workers()
            assert len(alive_workers) == 3

            # Poll for completion
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < 15:
                await asyncio.sleep(1)

                filters = TaskFilterParams(limit=20)
                all_tasks = eval_task_client.get_all_tasks(filters=filters)
                done_count = sum(1 for t in all_tasks.tasks if t.id in task_ids and t.status == "done")

                if done_count == 3:
                    break

            # Verify all completed
            filters = TaskFilterParams(limit=20)
            all_tasks = eval_task_client.get_all_tasks(filters=filters)
            completed = [t for t in all_tasks.tasks if t.id in task_ids and t.status == "done"]
            assert len(completed) == 3

        finally:
            success_thread_manager.shutdown_all()

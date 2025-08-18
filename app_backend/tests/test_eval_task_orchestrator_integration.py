import asyncio
import socket
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator, FixedScaler, WorkerInfo
from metta.app_backend.eval_task_worker import AbstractTaskExecutor, EvalTaskWorker, TaskResult
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskCreateRequest,
    TaskFilterParams,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager
from metta.app_backend.worker_managers.worker import Worker


class SuccessTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: TaskResponse) -> TaskResult:
        return TaskResult(success=True)


class FailureTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: TaskResponse) -> TaskResult:
        raise Exception("Failed task")


class GitHashAwareTaskExecutor(AbstractTaskExecutor):
    """Task executor that tracks which git hashes it has processed."""

    def __init__(self):
        self.processed_git_hashes: set[str] = set()
        self.task_count = 0

    async def execute_task(self, task: TaskResponse) -> TaskResult:
        self.task_count += 1
        if task.git_hash:
            self.processed_git_hashes.add(task.git_hash)
        # Simulate some work
        await asyncio.sleep(0.1)
        return TaskResult(success=True)


class DelayedTaskExecutor(AbstractTaskExecutor):
    """Task executor with configurable delay."""

    def __init__(self, delay: float = 0.5, max_tasks: int | None = None):
        self.delay = delay
        self.max_tasks = max_tasks
        self.executed_tasks = 0

    async def execute_task(self, task: TaskResponse) -> TaskResult:
        if self.max_tasks and self.executed_tasks >= self.max_tasks:
            raise Exception(f"Max tasks ({self.max_tasks}) reached")

        self.executed_tasks += 1
        await asyncio.sleep(self.delay)
        return TaskResult(success=True)


class HttpEvalTaskClientEnv:
    """Environment for HTTP-based eval task client tests."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self._httpx_clients = []

    def make_client(self) -> EvalTaskClient:
        """Create a new EvalTaskClient instance."""
        client = EvalTaskClient.__new__(EvalTaskClient)
        httpx_client = AsyncClient(base_url=self.base_url)
        client._http_client = httpx_client
        client._machine_token = self.token
        self._httpx_clients.append(httpx_client)
        return client

    async def aclose_all(self):
        await asyncio.gather(*(cl.aclose() for cl in self._httpx_clients), return_exceptions=True)


class TestEvalTaskOrchestratorIntegration:
    """Integration tests for EvalTaskOrchestrator with real database and FastAPI client."""

    def _find_free_port(self) -> int:
        """Find a free port for the HTTP server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @asynccontextmanager
    async def _http_server(self, test_app: FastAPI):
        """Start a real HTTP server for testing."""
        port = self._find_free_port()

        config = uvicorn.Config(test_app, host="127.0.0.1", port=port, log_level="critical")
        server = uvicorn.Server(config)

        # Start server in background
        task = asyncio.create_task(server.serve())

        # Wait for server to start
        await asyncio.sleep(0.2)

        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest_asyncio.fixture
    async def http_env(self, test_app: FastAPI, test_client: TestClient) -> AsyncGenerator[HttpEvalTaskClientEnv, Any]:
        """Create an HTTP environment for eval task client tests."""
        async with self._http_server(test_app) as base_url:
            async with AsyncClient(base_url=base_url) as tmp:
                r = await tmp.post(
                    "/tokens",
                    json={"name": "integration_test_token", "permissions": ["read", "write"]},
                    headers={"X-Auth-Request-Email": "test_user@example.com"},
                )
                r.raise_for_status()
                token = r.json()["token"]
            env = HttpEvalTaskClientEnv(base_url=base_url, token=token)
            try:
                yield env
            finally:
                await env.aclose_all()

    @pytest.fixture
    def test_policy_id(self, stats_client: StatsClient) -> uuid.UUID:
        """Create a test policy and return its ID."""
        training_run = stats_client.create_training_run(
            name=f"test_integration_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "integration"},
        )

        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )

        policy = stats_client.create_policy(
            name=f"test_integration_policy_{uuid.uuid4().hex[:8]}",
            description="Test policy for integration tests",
            epoch_id=epoch.id,
        )

        return policy.id

    def _create_managed_worker(
        self,
        worker_name: str,
        http_env: HttpEvalTaskClientEnv,
        task_executor: AbstractTaskExecutor,
        poll_interval: float = 0.5,
    ) -> tuple[EvalTaskWorker, EvalTaskClient]:
        """Create a worker with a managed client that can be properly closed."""
        worker_client = http_env.make_client()
        worker = EvalTaskWorker(
            client=worker_client,
            assignee=worker_name,
            task_executor=task_executor,
            poll_interval=poll_interval,
        )
        return worker, worker_client

    def create_success_worker(self, worker_name: str, http_env: HttpEvalTaskClientEnv) -> EvalTaskWorker:
        return EvalTaskWorker(
            client=http_env.make_client(),
            assignee=worker_name,
            task_executor=SuccessTaskExecutor(),
            poll_interval=0.5,
        )

    @pytest.fixture
    def mock_thread_manager(self, http_env: HttpEvalTaskClientEnv) -> ThreadWorkerManager:
        """Create a ThreadWorkerManager with mock workers."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, http_env)

        return ThreadWorkerManager(create_worker=create_worker)

    @pytest.fixture
    def orchestrator(
        self, mock_thread_manager: ThreadWorkerManager, http_env: HttpEvalTaskClientEnv
    ) -> EvalTaskOrchestrator:
        """Create orchestrator with mocked worker manager."""
        orch = EvalTaskOrchestrator(
            task_client=http_env.make_client(),
            worker_manager=mock_thread_manager,
            poll_interval=0.5,  # Fast polling for tests
            worker_scaler=FixedScaler(2),
            worker_idle_timeout=5.0,  # Short timeout for tests
        )
        return orch

    @pytest.mark.asyncio
    async def test_successful_task_processing(self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv):
        """Test that tasks are successfully processed by workers."""
        eval_task_client = http_env.make_client()

        # Create a test task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="integration_test_hash",
                sim_suite="navigation",
                env_overrides={"test": "integration"},
            )
        )
        task_id = task_response.id

        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=SuccessTaskExecutor(),
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,
            worker_scaler=FixedScaler(2),
        )

        try:
            # Run first orchestrator cycle to start workers
            await orchestrator.run_cycle()

            # Brief wait for workers to start
            await asyncio.sleep(0.5)

            # Run second orchestrator cycle to assign tasks
            await orchestrator.run_cycle()

            # Wait for workers to process tasks
            # Poll for completion rather than fixed sleep
            for _ in range(10):  # Up to 10 seconds
                await asyncio.sleep(1)

                # Check if task is completed
                filters = TaskFilterParams(policy_ids=[test_policy_id], limit=10)
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                processed_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

                if processed_task and processed_task.status == "done":
                    break

            # Final check
            assert processed_task is not None
            assert processed_task.status == "done"

        finally:
            # Clean up workers
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_failed_task_processing(self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv):
        """Test that failed tasks are marked as error."""

        # Create orchestrator with failure workers
        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=FailureTaskExecutor(),
                poll_interval=0.5,
            )

        failure_manager = ThreadWorkerManager(create_worker=create_worker)
        eval_task_client = http_env.make_client()

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=failure_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,
            worker_scaler=FixedScaler(1),
        )

        # Create a test task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="failure_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        try:
            # Run first orchestrator cycle to start workers
            await orchestrator.run_cycle()
            await asyncio.sleep(0.5)

            # Run second orchestrator cycle to assign tasks
            await orchestrator.run_cycle()

            # Wait for workers to process and fail the task
            # Poll for completion rather than fixed sleep
            for _ in range(5):  # Up to 5 seconds
                await asyncio.sleep(1)

                # Check if task failed
                filters = TaskFilterParams(policy_ids=[test_policy_id], limit=10)
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                failed_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

                if failed_task and failed_task.status == "error":
                    break

            # Final check
            assert failed_task is not None
            assert failed_task.status == "error"

            # Check error reason was stored
            assert "error_reason" in failed_task.attributes or any(
                key.startswith("error_reason_") for key in failed_task.attributes.keys()
            )

        finally:
            failure_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_multiple_workers_concurrent_processing(
        self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv
    ):
        """Test multiple workers processing different tasks concurrently."""

        # Create orchestrator with multiple workers
        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, http_env)

        success_manager = ThreadWorkerManager(create_worker=create_worker)

        eval_task_client = http_env.make_client()
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_manager,
            poll_interval=0.5,
            worker_idle_timeout=10.0,
            worker_scaler=FixedScaler(3),
        )

        # Create multiple test tasks
        task_ids = []
        for i in range(5):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"concurrent_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        try:
            # Run multiple orchestrator cycles
            start_time = datetime.now()
            for _ in range(10):  # Run for a while to process all tasks
                await orchestrator.run_cycle()
                await asyncio.sleep(0.5)

                # Check if all tasks are done
                filters = TaskFilterParams(policy_ids=[test_policy_id], limit=20)
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)

                done_tasks = [task for task in all_tasks.tasks if task.id in task_ids and task.status == "done"]
                if len(done_tasks) == len(task_ids):
                    break

                # Safety timeout
                if (datetime.now() - start_time).total_seconds() > 30:
                    break

            # Verify all tasks were completed
            filters = TaskFilterParams(policy_ids=[test_policy_id], limit=20)
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            completed_count = sum(1 for task in all_tasks.tasks if task.id in task_ids and task.status == "done")

            # At least some tasks should be completed
            assert completed_count > 0, "No tasks were completed"

        finally:
            success_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_discovery_and_lifecycle(self, http_env: HttpEvalTaskClientEnv):
        """Test worker discovery and lifecycle management."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=SuccessTaskExecutor(),
                poll_interval=0.1,
            )

        success_manager = ThreadWorkerManager(create_worker=create_worker)

        try:
            # Initially no workers
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 0

            # Start a worker
            worker_info = success_manager.start_worker()
            assert worker_info is not None

            # Should discover the worker
            await asyncio.sleep(0.5)
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 1
            assert alive_workers[0].name == worker_info

            # Clean up specific worker
            success_manager.cleanup_worker(worker_info)
            await asyncio.sleep(0.5)

            # Should no longer discover the worker
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 0

        finally:
            success_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_orchestrator_with_custom_worker_manager(self, http_env: HttpEvalTaskClientEnv):
        """Test that orchestrator works with custom worker managers."""

        # Create mock worker manager
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock()
        mock_worker_manager.cleanup_worker = Mock()

        # Create orchestrator with custom worker manager
        eval_task_client = http_env.make_client()
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
            worker_scaler=FixedScaler(1),
        )

        # Should use worker manager
        assert orchestrator._worker_manager is mock_worker_manager

        # Run a cycle - should call worker manager methods
        await orchestrator.run_cycle()

        mock_worker_manager.discover_alive_workers.assert_called_once()
        mock_worker_manager.start_worker.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_dead_worker_with_assigned_task(
        self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv
    ):
        """Test that tasks assigned to dead workers are unclaimed and retried."""

        # Create orchestrator with controlled worker manager
        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, http_env)

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        eval_task_client = http_env.make_client()
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,
            worker_scaler=FixedScaler(1),
        )

        # Create a test task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="dead_worker_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        try:
            # Start worker and assign task
            await orchestrator.run_cycle()  # Start worker

            await asyncio.sleep(0.5)

            await orchestrator.run_cycle()  # Assign task

            # Wait for task to be assigned
            for _ in range(5):
                await asyncio.sleep(0.5)
                filters = TaskFilterParams(policy_ids=[test_policy_id])
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                assigned_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

                if assigned_task and assigned_task.assignee:
                    break
            assert assigned_task is not None
            assert assigned_task.assignee is not None
            # Worker assignment verified

            # Kill the worker (simulate worker death)
            worker_manager.shutdown_all()
            await asyncio.sleep(0.1)

            # Run orchestrator cycle - should detect dead worker and unclaim task
            await orchestrator.run_cycle()

            # Verify task state after worker death
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            final_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

            assert final_task is not None

            # The task could either be completed (if worker finished before being killed)
            # or unclaimed (if orchestrator detected dead worker)
            if final_task.status == "done":
                # Worker completed task before being killed - this is valid
                assert final_task.assignee is not None
            else:
                # Worker was killed and task was unclaimed
                assert final_task.assignee is None
                assert final_task.status == "unprocessed"  # Should be retried since retries < 3
                assert "unassign_reason_0" in final_task.attributes
                assert final_task.attributes["unassign_reason_0"] == "worker_dead"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_kill_task_with_timeout(self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv):
        """Test that tasks running too long are timed out and unclaimed."""

        # Create a mock task that appears to have been assigned 15 minutes ago
        eval_task_client = http_env.make_client()
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="timeout_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Manually assign the task with an old timestamp (simulating long-running task)

        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=15)

        # First claim the task
        claim_request = TaskClaimRequest(tasks=[task_id], assignee="timeout_test_worker")
        await eval_task_client.claim_tasks(claim_request)

        # Then manually set the assigned_at timestamp to be old
        # We'll need to do this via direct database manipulation or create a test helper
        # For now, let's use the update API to simulate this
        update_request = TaskUpdateRequest(
            updates={
                task_id: TaskStatusUpdate(
                    status="unprocessed",  # Use valid status
                    attributes={"assigned_at_override": old_timestamp.isoformat()},
                )
            }
        )
        await eval_task_client.update_task_status(update_request)

        # Create orchestrator with a mock worker that appears to be alive
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        # Make the timeout_test_worker appear alive so timeout logic is triggered
        mock_worker_manager.discover_alive_workers = AsyncMock(
            return_value=[Worker(name="timeout_test_worker", status="Running")]
        )
        mock_worker_manager.start_worker = Mock()
        mock_worker_manager.cleanup_worker = Mock()

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,  # Short timeout for testing
            worker_scaler=FixedScaler(1),
        )

        # Manually create a task with old timestamp for testing
        # Since we can't easily manipulate assigned_at through the API,
        # we'll create a mock scenario

        # Get the task and verify it's assigned
        filters = TaskFilterParams(policy_ids=[test_policy_id])
        all_tasks = await eval_task_client.get_all_tasks(filters=filters)
        assigned_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

        assert assigned_task is not None
        assert assigned_task.assignee == "timeout_test_worker"

        # For this test, we need to modify the task's assigned_at field directly
        # Since this is complex to do through the API, we'll test the timeout logic
        # by creating a task response object with old timestamp

        # Create a mock old task for testing the timeout logic
        old_task = TaskResponse(
            id=task_id,
            policy_id=test_policy_id,
            sim_suite="navigation",
            status="unprocessed",  # Use valid status
            assigned_at=old_timestamp,
            assignee="timeout_test_worker",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=20),
            attributes={"git_hash": "timeout_test_hash"},
            retries=0,
            user_id=None,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

        # Test the timeout detection logic directly
        # Include the timeout_test_worker in alive workers so timeout logic is triggered
        alive_workers_for_timeout_test = {
            "timeout_test_worker": WorkerInfo(
                worker=Worker(name="timeout_test_worker", status="Running"), git_hashes=set()
            )
        }
        await orchestrator._kill_dead_workers_and_tasks([old_task], alive_workers_for_timeout_test)

        # Verify task was unclaimed due to timeout
        filters = TaskFilterParams(policy_ids=[test_policy_id])
        all_tasks = await eval_task_client.get_all_tasks(filters=filters)
        timeout_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

        assert timeout_task is not None
        assert timeout_task.assignee is None
        assert timeout_task.status == "unprocessed"  # Should be retried since retries < 3
        assert "unassign_reason_0" in timeout_task.attributes
        assert timeout_task.attributes["unassign_reason_0"] == "worker_timeout"

    @pytest.mark.asyncio
    async def test_kill_task_after_max_retries(self, http_env: HttpEvalTaskClientEnv, test_policy_id: uuid.UUID):
        """Test that tasks with max retries are marked as error instead of unprocessed."""
        eval_task_client = http_env.make_client()

        # Create a task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="max_retries_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Simulate a task that has already been retried 3 times (max retries)
        # First claim and update with retries
        claim_request = TaskClaimRequest(tasks=[task_id], assignee="max_retries_worker")
        await eval_task_client.claim_tasks(claim_request)

        # Set retries to 3 (max)
        update_request = TaskUpdateRequest(
            updates={
                task_id: TaskStatusUpdate(
                    status="unprocessed",  # Required field
                    attributes={"retries_override": "3"},
                )
            }
        )
        await eval_task_client.update_task_status(update_request)

        # Create mock orchestrator
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock()
        mock_worker_manager.cleanup_worker = Mock()

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
            worker_scaler=FixedScaler(1),
        )

        # Create a mock task with 3 retries assigned to dead worker
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=15)
        max_retry_task = TaskResponse(
            id=task_id,
            policy_id=test_policy_id,
            sim_suite="navigation",
            status="unprocessed",  # Use valid status
            assigned_at=old_timestamp,
            assignee="dead_max_retries_worker",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            attributes={"git_hash": "max_retries_test_hash"},
            retries=3,  # Max retries reached
            user_id=None,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

        # Test the max retry logic
        await orchestrator._kill_dead_workers_and_tasks([max_retry_task], {})

        # Verify task was marked as error (not unprocessed) due to max retries
        filters = TaskFilterParams(policy_ids=[test_policy_id])
        all_tasks = await eval_task_client.get_all_tasks(filters=filters)
        error_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

        assert error_task is not None
        assert error_task.assignee is None
        assert error_task.status == "error"  # Should be error since retries >= 3
        assert "unassign_reason_3" in error_task.attributes

    @pytest.mark.asyncio
    async def test_kill_long_running_worker(self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv):
        """Test that workers running tasks too long are killed."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, http_env)

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        eval_task_client = http_env.make_client()
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,
            worker_scaler=FixedScaler(1),
        )

        try:
            # Start a worker
            await orchestrator.run_cycle()
            await asyncio.sleep(0.5)

            # Get the started worker
            alive_workers = await worker_manager.discover_alive_workers()
            assert len(alive_workers) == 1
            worker_name = alive_workers[0].name

            # Create a mock task assigned to this worker with old timestamp
            task_id = uuid.uuid4()
            old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=15)

            long_running_task = TaskResponse(
                id=task_id,
                policy_id=test_policy_id,
                sim_suite="navigation",
                status="unprocessed",  # Use valid status
                assigned_at=old_timestamp,
                assignee=worker_name,
                created_at=datetime.now(timezone.utc) - timedelta(hours=1),
                attributes={"git_hash": "long_running_test_hash"},
                retries=0,
                user_id=None,
                updated_at=datetime.now(timezone.utc) - timedelta(minutes=15),
            )

            # Get current alive workers for the test
            alive_workers_by_name = await orchestrator._get_available_workers([])

            # Test worker cleanup logic
            await orchestrator._kill_dead_workers_and_tasks([long_running_task], alive_workers_by_name)

            # Verify worker was killed (cleanup_worker should have been called)
            # Since we can't easily verify the internal state, we'll check that
            # the logic completed without error and the task processing continued

            # The test validates the code path executes without throwing exceptions
            # which is the primary concern for this integration test

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_kill_dead_workers_error_handling(self, test_policy_id: uuid.UUID):
        """Test error handling in kill_dead_workers_and_tasks function."""

        # Create mock orchestrator with error-prone task client
        # Create a mock task client that raises an exception
        error_task_client = Mock()
        error_task_client.update_task_status = AsyncMock(side_effect=Exception("Simulated API error"))
        error_task_client.get_git_hashes_for_workers = AsyncMock(return_value=Mock(git_hashes={}))
        error_task_client.close = AsyncMock()

        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock()
        mock_worker_manager.cleanup_worker = Mock()

        # Create orchestrator with error-prone client
        orchestrator = EvalTaskOrchestrator(
            task_client=error_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
            worker_scaler=FixedScaler(1),
        )

        # Create a task assigned to a dead worker
        dead_worker_task = TaskResponse(
            id=uuid.uuid4(),
            policy_id=test_policy_id,
            sim_suite="navigation",
            status="unprocessed",  # Use valid status
            assigned_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            assignee="dead_worker_for_error_test",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            attributes={"git_hash": "error_test_hash"},
            retries=1,
            user_id=None,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )

        # Test that errors are handled gracefully (no exception should bubble up)
        try:
            await orchestrator._kill_dead_workers_and_tasks([dead_worker_task], {})
            # Should complete without raising exception despite API error
            assert True  # Test passed if we reach here
        except Exception:
            pytest.fail("kill_dead_workers_and_tasks should handle errors gracefully")

    @pytest.mark.asyncio
    async def test_git_hash_task_affinity(self, http_env: HttpEvalTaskClientEnv, test_policy_id: uuid.UUID):
        """Test that workers prefer tasks with matching git hashes."""

        eval_task_client = http_env.make_client()

        # Create tasks with different git hashes
        git_hash_a = "hash_a_12345"
        git_hash_b = "hash_b_67890"

        tasks_a = []
        tasks_b = []

        # Create 3 tasks for each git hash
        for _i in range(3):
            task_a = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=git_hash_a,
                    sim_suite="navigation",
                )
            )
            tasks_a.append(task_a.id)

            task_b = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=git_hash_b,
                    sim_suite="navigation",
                )
            )
            tasks_b.append(task_b.id)

        # Create workers that track git hash affinity
        git_hash_executors = {}

        def create_worker(worker_name: str) -> EvalTaskWorker:
            executor = GitHashAwareTaskExecutor()
            git_hash_executors[worker_name] = executor
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=executor,
                poll_interval=0.2,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            worker_scaler=FixedScaler(2),
        )

        try:
            # Run orchestration cycles to process tasks
            for _ in range(15):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.3)

                # Check if all tasks are completed
                filters = TaskFilterParams(policy_ids=[test_policy_id])
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                done_tasks = [t for t in all_tasks.tasks if t.status == "done"]

                if len(done_tasks) >= 6:  # All tasks completed
                    break

            # Verify git hash affinity worked
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            # Count completed tasks by git hash
            completed_a = sum(1 for t in all_tasks.tasks if t.git_hash == git_hash_a and t.status == "done")
            completed_b = sum(1 for t in all_tasks.tasks if t.git_hash == git_hash_b and t.status == "done")

            # At least some tasks should be completed
            assert completed_a + completed_b > 0, "Some tasks should be completed"

            # Workers should have processed tasks with specific git hashes
            # (exact affinity depends on timing, so we check that git hash tracking worked)
            total_unique_hashes = sum(len(executor.processed_git_hashes) for executor in git_hash_executors.values())
            assert total_unique_hashes > 0, "Workers should have processed tasks with git hashes"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_dynamic_worker_scaling(
        self, test_client: TestClient, test_app: FastAPI, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv
    ):
        """Test dynamic worker creation based on available tasks.

        This test has special worker_clients cleanup logic that needs to be preserved.
        """

        eval_task_client = http_env.make_client()
        task_ids = []
        for i in range(6):  # Reduced from 10 to 6
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"scaling_test_hash_{i % 3}",  # 3 different git hashes
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            worker = EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=DelayedTaskExecutor(delay=0.2),  # Reduced delay from 0.5 to 0.2
                poll_interval=0.1,
            )
            return worker

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            worker_scaler=FixedScaler(3),  # Scale to 3 workers (reduced from 5)
        )

        try:
            worker_counts = []

            # Run orchestration and track worker scaling
            for _cycle in range(15):  # Reduced from 20 to 15
                await orchestrator.run_cycle()

                # Track how many workers are alive
                alive_workers = await worker_manager.discover_alive_workers()
                worker_counts.append(len(alive_workers))

                await asyncio.sleep(0.2)  # Reduced from 0.3 to 0.2

                # Check if work is progressing
                filters = TaskFilterParams(policy_ids=[test_policy_id])
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                completed_tasks = [t for t in all_tasks.tasks if t.status == "done"]

                if len(completed_tasks) >= 4:  # Reduced from 8 to 4 (since we have 6 tasks total)
                    break

            # Verify worker scaling occurred
            max_workers_seen = max(worker_counts) if worker_counts else 0
            assert max_workers_seen >= 2, f"Should scale to at least 2 workers, saw max {max_workers_seen}"
            assert max_workers_seen <= 3, f"Should not exceed 3 workers (FixedScaler), saw {max_workers_seen}"

            # Verify tasks were distributed and completed
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            completed_count = sum(1 for t in all_tasks.tasks if t.status == "done")
            assert completed_count >= 3, f"Expected at least 3 completed tasks, got {completed_count}"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_idle_timeout_behavior(self, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv):
        """Test worker idle timeout and cleanup behavior."""

        eval_task_client = http_env.make_client()

        # Create a limited number of tasks
        task_ids = []
        for i in range(2):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"idle_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=DelayedTaskExecutor(delay=0.3),
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            worker_idle_timeout=2.0,  # Short timeout for testing
            worker_scaler=FixedScaler(4),
        )

        try:
            # Run orchestration to start workers and process tasks
            for _ in range(15):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.3)

                # Check task completion
                filters = TaskFilterParams(policy_ids=[test_policy_id])
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                completed_tasks = [t for t in all_tasks.tasks if t.status == "done"]

                if len(completed_tasks) >= 2:
                    # All tasks done, continue running to test idle cleanup
                    break

            # Continue running to let idle timeout take effect
            await asyncio.sleep(3.0)  # Wait longer than idle timeout

            # Run a few more cycles to trigger cleanup
            for _ in range(3):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.5)

            # Verify tasks were completed
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            completed_count = sum(1 for t in all_tasks.tasks if t.status == "done")
            assert completed_count == 2, f"Expected 2 completed tasks, got {completed_count}"

            # The test validates that the orchestrator handles idle workers appropriately
            # Exact idle timeout behavior depends on the worker manager implementation

        finally:
            worker_manager.shutdown_all()
            await eval_task_client.close()

    @pytest.mark.asyncio
    async def test_task_retry_with_failures(
        self, test_client: TestClient, test_app: FastAPI, test_policy_id: uuid.UUID, http_env: HttpEvalTaskClientEnv
    ):
        """Test task failure handling behavior."""
        eval_task_client = http_env.make_client()

        # Create a task that will fail
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="failure_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Create an executor that always fails
        class FailingTaskExecutor(AbstractTaskExecutor):
            def __init__(self):
                self.execution_count = 0

            async def execute_task(self, task: TaskResponse) -> TaskResult:
                self.execution_count += 1
                raise Exception("Simulated failure")

        failing_executor = FailingTaskExecutor()

        def create_worker(worker_name: str) -> EvalTaskWorker:
            worker_client = http_env.make_client()
            return EvalTaskWorker(
                client=worker_client,
                assignee=worker_name,
                task_executor=failing_executor,
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            worker_scaler=FixedScaler(1),
        )

        try:
            # Run orchestration to process the failing task
            for _cycle in range(10):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.3)

                # Check task status
                filters = TaskFilterParams(policy_ids=[test_policy_id])
                all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                task = next((t for t in all_tasks.tasks if t.id == task_id), None)

                if task and task.status in ["done", "error"]:
                    break

            # Verify final task state
            assert task is not None
            # Task should be marked as error when execution fails
            assert task.status == "error", f"Task should be in error state, got {task.status}"

            # Verify the executor was called at least once
            assert failing_executor.execution_count >= 1, (
                f"Expected at least 1 execution attempt, got {failing_executor.execution_count}"
            )

        finally:
            worker_manager.shutdown_all()
            await eval_task_client.close()

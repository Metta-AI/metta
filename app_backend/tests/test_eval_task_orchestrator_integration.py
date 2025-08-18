import asyncio
import uuid
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.eval_task_worker import AbstractTaskExecutor, EvalTaskWorker
from metta.app_backend.routes.eval_task_routes import (
    TaskCreateRequest,
    TaskFilterParams,
    TaskResponse,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager


class SuccessTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: TaskResponse) -> None:
        pass


class FailureTaskExecutor(AbstractTaskExecutor):
    def __init__(self):
        pass

    async def execute_task(self, task: TaskResponse) -> None:
        raise Exception("Failed task")


class TestEvalTaskOrchestratorIntegration:
    """Integration tests for EvalTaskOrchestrator with real database and FastAPI client."""

    @pytest.fixture
    def eval_task_client(self, test_client: TestClient, test_app: FastAPI) -> EvalTaskClient:
        """Create an eval task client for testing."""
        token_response = test_client.post(
            "/tokens",
            json={"name": "integration_test_token", "permissions": ["read", "write"]},
            headers={"X-Auth-Request-Email": "test_user@example.com"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]

        client = EvalTaskClient.__new__(EvalTaskClient)
        client._http_client = AsyncClient(transport=ASGITransport(app=test_app), base_url=test_client.base_url)
        client._machine_token = token

        return client

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

    def create_success_worker(self, worker_name: str, eval_task_client: EvalTaskClient) -> EvalTaskWorker:
        return EvalTaskWorker(
            client=eval_task_client,
            assignee=worker_name,
            task_executor=SuccessTaskExecutor(),
            poll_interval=0.5,
        )

    @pytest.fixture
    def mock_thread_manager(self, eval_task_client: EvalTaskClient, test_client: TestClient) -> ThreadWorkerManager:
        """Create a ThreadWorkerManager with mock workers."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, eval_task_client)

        return ThreadWorkerManager(create_worker=create_worker)

    @pytest.fixture
    def orchestrator(
        self, eval_task_client: EvalTaskClient, mock_thread_manager: ThreadWorkerManager
    ) -> EvalTaskOrchestrator:
        """Create orchestrator with mocked worker manager."""
        orch = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_thread_manager,
            poll_interval=0.5,  # Fast polling for tests
            worker_idle_timeout=5.0,  # Short timeout for tests
            max_workers=2,
        )
        return orch

    @pytest.mark.asyncio
    async def test_successful_task_processing(
        self, orchestrator: EvalTaskOrchestrator, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test that tasks are successfully processed by workers."""
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
            if isinstance(orchestrator._worker_manager, ThreadWorkerManager):
                orchestrator._worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_failed_task_processing(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test that failed tasks are marked as error."""

        # Create orchestrator with failure workers
        def create_worker(worker_name: str) -> EvalTaskWorker:
            return EvalTaskWorker(
                client=eval_task_client,
                assignee=worker_name,
                task_executor=FailureTaskExecutor(),
                poll_interval=0.5,
            )

        failure_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=failure_manager,
            poll_interval=0.5,
            worker_idle_timeout=5.0,
            max_workers=1,
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
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test multiple workers processing different tasks concurrently."""

        # Create orchestrator with multiple workers
        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, eval_task_client)

        success_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_manager,
            poll_interval=0.5,
            worker_idle_timeout=10.0,
            max_workers=3,
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
    async def test_worker_discovery_and_lifecycle(self, eval_task_client: EvalTaskClient):
        """Test worker discovery and lifecycle management."""

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_success_worker(worker_name, eval_task_client)

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
    async def test_orchestrator_with_custom_worker_manager(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test that orchestrator works with custom worker managers."""
        from unittest.mock import AsyncMock, Mock

        # Create mock worker manager
        mock_worker_manager = Mock(spec=AbstractWorkerManager)
        mock_worker_manager.discover_alive_workers = AsyncMock(return_value=[])
        mock_worker_manager.start_worker = Mock()
        mock_worker_manager.cleanup_worker = Mock()

        # Create orchestrator with custom worker manager
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mock_worker_manager,
            poll_interval=0.5,
            max_workers=1,
        )

        # Should use worker manager
        assert orchestrator._worker_manager is mock_worker_manager

        # Run a cycle - should call worker manager methods
        await orchestrator.run_cycle()

        mock_worker_manager.discover_alive_workers.assert_called_once()
        mock_worker_manager.start_worker.assert_called_once()

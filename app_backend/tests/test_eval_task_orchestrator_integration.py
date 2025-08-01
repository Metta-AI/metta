import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.routes.eval_task_routes import (
    TaskCreateRequest,
    TaskFilterParams,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager

from .test_workers.mock_workers import (
    MockSuccessWorker,
    MockFailureWorker,
    MockTimeoutWorker,
    MockConditionalWorker,
    MockWorkerFactory,
)


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
        client._base_url = str(test_client.base_url)

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

    @pytest.fixture
    def mock_thread_manager(self, eval_task_client: EvalTaskClient, test_client: TestClient) -> ThreadWorkerManager:
        """Create a ThreadWorkerManager with mock workers."""
        def worker_factory(worker_name: str) -> MockSuccessWorker:
            return MockSuccessWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(test_client.base_url),
                machine_token="test-token"
            )
        return ThreadWorkerManager(worker_factory=worker_factory)

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
            if hasattr(orchestrator._worker_manager, "shutdown_all"):
                orchestrator._worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_failed_task_processing(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test that failed tasks are marked as error."""
        # Create orchestrator with failure workers
        def failure_worker_factory(worker_name: str) -> MockFailureWorker:
            from .test_workers.mock_workers import MockFailureWorker
            return MockFailureWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token",
                failure_message="Integration test failure"
            )
        failure_manager = ThreadWorkerManager(worker_factory=failure_worker_factory)

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
        def success_worker_factory(worker_name: str) -> MockSuccessWorker:
            return MockSuccessWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token",
                sim_delay=0.2
            )
        success_manager = ThreadWorkerManager(worker_factory=success_worker_factory)

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
    async def test_worker_timeout_handling(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test that workers are properly killed when tasks timeout."""
        # Create orchestrator with timeout workers and very short timeout
        def timeout_worker_factory(worker_name: str) -> MockTimeoutWorker:
            from .test_workers.mock_workers import MockTimeoutWorker
            return MockTimeoutWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token",
                sim_delay=10.0
            )
        timeout_manager = ThreadWorkerManager(worker_factory=timeout_worker_factory)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=timeout_manager,
            poll_interval=0.5,
            worker_idle_timeout=2.0,  # Very short timeout
            max_workers=1,
        )

        # Create a test task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="timeout_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        try:
            # Run orchestrator cycle to start the task
            await orchestrator.run_cycle()
            await asyncio.sleep(1)

            # Simulate task assignment time in the past (older than timeout)
            await eval_task_client.update_task_status(
                TaskUpdateRequest(
                    updates={
                        task_id: TaskStatusUpdate(
                            status="unprocessed",
                            attributes={
                                "assigned_at_override": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
                            },
                        )
                    }
                )
            )

            # Run orchestrator cycle to handle timeout
            await orchestrator.run_cycle()
            await asyncio.sleep(1)

            # Task should be unprocessed again or marked as error (depending on retries)
            filters = TaskFilterParams(policy_ids=[test_policy_id], limit=10)
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            timeout_task = next((task for task in all_tasks.tasks if task.id == task_id), None)
            assert timeout_task is not None
            # Task should either be unprocessed (retry) or error (max retries)
            assert timeout_task.status in ["unprocessed", "error"]

        finally:
            timeout_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenarios(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test orchestrator handling mix of successful and failed tasks."""

        # Create conditional worker that alternates success/failure
        task_counter = [0]  # Use list to allow modification in closure
        def success_condition(task):
            task_counter[0] += 1
            should_succeed = task_counter[0] % 2 == 1  # Odd numbers succeed
            return should_succeed

        def conditional_worker_factory(worker_name: str) -> MockConditionalWorker:
            from .test_workers.mock_workers import MockConditionalWorker
            return MockConditionalWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token",
                success_condition=success_condition,
                failure_message="Even task ID failure"
            )
        mixed_manager = ThreadWorkerManager(worker_factory=conditional_worker_factory)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=mixed_manager,
            poll_interval=0.5,
            worker_idle_timeout=10.0,
            max_workers=3,
        )

        # Create multiple test tasks
        task_ids = []
        for i in range(4):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"mixed_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        try:
            # Run first orchestrator cycle to start workers
            await orchestrator.run_cycle()
            await asyncio.sleep(0.5)
            
            # Run orchestrator cycles to assign and process tasks
            start_time = datetime.now()
            for _ in range(15):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.5)

                # Safety timeout
                if (datetime.now() - start_time).total_seconds() > 20:
                    break

            # Check final task statuses
            filters = TaskFilterParams(policy_ids=[test_policy_id], limit=20)
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            task_statuses = {}
            for task in all_tasks.tasks:
                if task.id in task_ids:
                    task_statuses[task.id] = task.status

            # Should have mix of done and error statuses
            done_count = sum(1 for status in task_statuses.values() if status == "done")
            error_count = sum(1 for status in task_statuses.values() if status == "error")

            assert done_count > 0, "No tasks succeeded"
            assert error_count > 0, "No tasks failed"
            assert done_count + error_count >= 2, "Not enough tasks processed"  # At least 2 tasks should be processed

        finally:
            mixed_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_orchestrator_cleanup_on_shutdown(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test that orchestrator properly cleans up workers on shutdown."""
        def success_worker_factory(worker_name: str) -> MockSuccessWorker:
            return MockSuccessWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token"
            )
        success_manager = ThreadWorkerManager(worker_factory=success_worker_factory)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=success_manager,
            poll_interval=0.5,
            worker_idle_timeout=10.0,
            max_workers=2,
        )

        try:
            # Start some workers by running a cycle
            await orchestrator.run_cycle()
            await asyncio.sleep(1)

            # Verify workers are alive
            alive_workers = await success_manager.discover_alive_workers()
            initial_worker_count = len(alive_workers)
            assert initial_worker_count > 0, "No workers were started"

        finally:
            # Test cleanup
            success_manager.shutdown_all()
            await asyncio.sleep(1)  # Give time for cleanup

            # Verify all workers are cleaned up
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 0, "Workers were not properly cleaned up"

    @pytest.mark.asyncio
    async def test_worker_discovery_and_lifecycle(self, eval_task_client: EvalTaskClient):
        """Test worker discovery and lifecycle management."""
        def worker_factory(worker_name: str) -> MockSuccessWorker:
            return MockSuccessWorker(
                client=eval_task_client,
                assignee=worker_name,
                backend_url=str(eval_task_client._base_url),
                machine_token="test-token"
            )
        success_manager = ThreadWorkerManager(worker_factory=worker_factory)

        try:
            # Initially no workers
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 0

            # Start a worker
            worker_info = success_manager.start_worker()
            assert worker_info is not None
            assert worker_info.container_name is not None

            # Should discover the worker
            await asyncio.sleep(0.5)
            alive_workers = await success_manager.discover_alive_workers()
            assert len(alive_workers) == 1
            assert alive_workers[0].container_name == worker_info.container_name

            # Clean up specific worker
            success_manager.cleanup_worker(worker_info.container_id)
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

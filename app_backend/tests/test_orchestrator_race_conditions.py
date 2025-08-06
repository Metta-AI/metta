"""
Race condition tests for EvalTaskOrchestrator.

These tests focus on concurrent operations that could lead to data corruption,
task loss, or inconsistent state if not properly handled.
"""

import asyncio
import socket
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator
from metta.app_backend.eval_task_worker import AbstractTaskExecutor, EvalTaskWorker
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskCreateRequest,
    TaskFilterParams,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager


class SlowTaskExecutor(AbstractTaskExecutor):
    """Task executor that takes a specified amount of time to complete."""

    def __init__(self, delay: float = 1.0, should_fail: bool = False):
        self.delay = delay
        self.should_fail = should_fail
        self.executions = 0

    async def execute_task(self, task: TaskResponse) -> None:
        self.executions += 1
        await asyncio.sleep(self.delay)
        if self.should_fail:
            raise Exception(f"Simulated failure for task {task.id}")


class TestOrchestratorRaceConditions:
    """Test race conditions in EvalTaskOrchestrator."""

    def _find_free_port(self) -> int:
        """Find a free port for the HTTP server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @asynccontextmanager
    async def _http_server(self, app: FastAPI):
        """Start a real HTTP server for testing."""
        port = self._find_free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
        server = uvicorn.Server(config)

        # Start server in background
        task = asyncio.create_task(server.serve())

        # Wait for server to start
        await asyncio.sleep(0.1)

        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            # Shutdown server
            server.should_exit = True
            await task

    @pytest.fixture
    async def eval_task_client(self, test_client: TestClient, test_app: FastAPI):
        """Create an eval task client for testing."""
        token_response = test_client.post(
            "/tokens",
            json={"name": "race_test_token", "permissions": ["read", "write"]},
            headers={"X-Auth-Request-Email": "test_user@example.com"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]

        async with self._http_server(test_app) as base_url:
            client = EvalTaskClient.__new__(EvalTaskClient)
            from httpx import AsyncClient

            client._http_client = AsyncClient(base_url=base_url)
            client._machine_token = token
            yield client
            await client._http_client.aclose()

    @pytest.fixture
    def test_policy_id(self, stats_client: StatsClient) -> uuid.UUID:
        """Create a test policy and return its ID."""
        training_run = stats_client.create_training_run(
            name=f"test_race_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "race_conditions"},
        )

        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )

        policy = stats_client.create_policy(
            name=f"test_race_policy_{uuid.uuid4().hex[:8]}",
            description="Test policy for race condition tests",
            epoch_id=epoch.id,
        )

        return policy.id

    def create_test_worker(self, worker_name: str, http_env, delay: float = 0.5) -> EvalTaskWorker:
        """Create a test worker with configurable delay."""
        return EvalTaskWorker(
            client=http_env.make_client(),
            assignee=worker_name,
            task_executor=SlowTaskExecutor(delay=delay),
            poll_interval=0.1,  # Fast polling for race tests
        )

    @pytest.mark.asyncio
    async def test_concurrent_task_claim_race(self, http_env, orchestrator_test_policy_id: uuid.UUID):
        """Test that multiple workers trying to claim the same task handle race conditions correctly."""
        eval_task_client = http_env.make_client()

        # Create a single task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=orchestrator_test_policy_id,
                git_hash="race_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Create multiple workers that will compete for the task
        def create_worker(worker_name: str) -> EvalTaskWorker:
            return EvalTaskWorker(
                client=http_env.make_client(),
                assignee=worker_name,
                task_executor=SlowTaskExecutor(delay=2.0),
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        # Create multiple orchestrators to simulate concurrent claim attempts
        orchestrators = []
        for _i in range(3):
            orchestrator = EvalTaskOrchestrator(
                task_client=http_env.make_client(),
                worker_manager=worker_manager,
                poll_interval=0.1,
                max_workers=1,
            )
            orchestrators.append(orchestrator)

        try:
            # Start workers from all orchestrators simultaneously
            await asyncio.gather(*[orch.run_cycle() for orch in orchestrators])
            await asyncio.sleep(0.1)  # Let workers start

            # Run assignment cycles concurrently
            await asyncio.gather(*[orch.run_cycle() for orch in orchestrators])

            # Wait for task processing
            await asyncio.sleep(3.0)

            # Verify only one worker got the task and it was completed
            filters = TaskFilterParams(policy_ids=[orchestrator_test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            completed_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

            assert completed_task is not None
            assert completed_task.status == "done"

            # Verify no task duplication occurred
            task_count = sum(1 for task in all_tasks.tasks if task.id == task_id)
            assert task_count == 1, "Task should not be duplicated"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_death_during_assignment_race(self, http_env, orchestrator_test_policy_id: uuid.UUID):
        """Test that orchestrator handles worker death gracefully by starting new workers and reassigning tasks."""
        eval_task_client = http_env.make_client()

        # Create a task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=orchestrator_test_policy_id,
                git_hash="worker_death_test_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return EvalTaskWorker(
                client=http_env.make_client(),
                assignee=worker_name,
                task_executor=SlowTaskExecutor(delay=2.0),  # Long delay to prevent actual execution
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.1,
            max_workers=1,
        )

        try:
            # Run a few cycles to start worker and assign task
            for _ in range(3):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.1)

            # Verify task is assigned to original worker
            filters = TaskFilterParams(policy_ids=[orchestrator_test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            task = next((task for task in all_tasks.tasks if task.id == task_id), None)

            assert task is not None
            assert task.assignee is not None  # Task should be assigned
            original_worker_name = task.assignee
            original_retries = task.retries

            # Kill the worker to simulate failure
            worker_manager.cleanup_worker(original_worker_name)
            await asyncio.sleep(0.1)

            # Run cycles to detect dead worker, start new worker, and reassign task
            for _ in range(5):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.1)

            # Verify fault tolerance: task should be reassigned to a new worker
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)
            task = next((task for task in all_tasks.tasks if task.id == task_id), None)

            assert task is not None
            assert task.status == "unprocessed"  # Status should be unprocessed (ready for work)
            assert task.assignee is not None  # Task should be reassigned to new worker
            assert task.assignee != original_worker_name  # Should be a different worker
            assert task.retries > original_retries  # Retries should have increased
            assert "unassign_reason" in str(task.attributes)  # Should have unassign reason recorded

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_concurrent_status_update_race(self, http_env, orchestrator_test_policy_id: uuid.UUID):
        """Test concurrent status updates with require_assignee conflicts."""
        eval_task_client = http_env.make_client()

        # Create and claim a task
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=orchestrator_test_policy_id,
                git_hash="status_update_race_hash",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Claim the task
        claim_request = TaskClaimRequest(tasks=[task_id], assignee="worker_1")
        await eval_task_client.claim_tasks(claim_request)

        # Simulate race condition: orchestrator tries to unassign while worker updates status
        async def orchestrator_unassign():
            """Orchestrator trying to unassign due to timeout."""
            await asyncio.sleep(0.1)  # Small delay
            update_request = TaskUpdateRequest(
                updates={
                    task_id: TaskStatusUpdate(
                        status="unprocessed", clear_assignee=True, attributes={"unassign_reason_0": "worker_timeout"}
                    )
                }
            )
            try:
                await eval_task_client.update_task_status(update_request)
            except Exception:
                # Race condition may cause this to fail, which is acceptable
                pass

        async def worker_complete_task():
            """Worker trying to mark task as done."""
            update_request = TaskUpdateRequest(
                require_assignee="worker_1",
                updates={
                    task_id: TaskStatusUpdate(
                        status="done", attributes={"completion_time": datetime.now(timezone.utc).isoformat()}
                    )
                },
            )
            try:
                await eval_task_client.update_task_status(update_request)
            except Exception:
                # Race condition may cause this to fail, which is acceptable
                pass

        # Run both operations concurrently
        await asyncio.gather(orchestrator_unassign(), worker_complete_task(), return_exceptions=True)

        # Verify final state is consistent
        filters = TaskFilterParams(policy_ids=[orchestrator_test_policy_id])
        all_tasks = await eval_task_client.get_all_tasks(filters=filters)
        final_task = next((task for task in all_tasks.tasks if task.id == task_id), None)

        assert final_task is not None
        # Either the worker succeeded (done) or orchestrator succeeded (unprocessed)
        assert final_task.status in ["done", "unprocessed"]

        # If done, should still have assignee; if unprocessed, should not
        if final_task.status == "done":
            assert final_task.assignee == "worker_1"
        else:
            assert final_task.assignee is None

    @pytest.mark.asyncio
    async def test_multiple_orchestrator_competition(self, http_env, orchestrator_test_policy_id: uuid.UUID):
        """Test multiple orchestrator instances competing for resources."""
        eval_task_client = http_env.make_client()

        # Create multiple tasks
        task_ids = []
        for i in range(5):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=orchestrator_test_policy_id,
                    git_hash=f"multi_orch_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        # Create multiple orchestrators with their own worker managers
        orchestrators = []
        worker_managers = []

        for _i in range(3):

            def create_worker(worker_name: str) -> EvalTaskWorker:
                return self.create_test_worker(worker_name, http_env, delay=1.0)

            worker_manager = ThreadWorkerManager(create_worker=create_worker)
            worker_managers.append(worker_manager)

            orchestrator = EvalTaskOrchestrator(
                task_client=http_env.make_client(),
                worker_manager=worker_manager,
                poll_interval=0.1,
                max_workers=2,
            )
            orchestrators.append(orchestrator)

        try:
            # Run multiple orchestration cycles concurrently
            for _cycle in range(5):
                await asyncio.gather(*[orch.run_cycle() for orch in orchestrators])
                await asyncio.sleep(0.2)  # Brief pause between cycles

            # Wait for tasks to complete
            await asyncio.sleep(3.0)

            # Verify all tasks were completed without duplication
            filters = TaskFilterParams(policy_ids=[orchestrator_test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            completed_count = 0
            for task_id in task_ids:
                task_count = sum(1 for task in all_tasks.tasks if task.id == task_id)
                assert task_count == 1, f"Task {task_id} should appear exactly once"

                task = next((task for task in all_tasks.tasks if task.id == task_id), None)
                if task and task.status == "done":
                    completed_count += 1

            # At least some tasks should be completed
            assert completed_count > 0, "Some tasks should be completed"

        finally:
            for worker_manager in worker_managers:
                worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_lifecycle_race_conditions(self, http_env, orchestrator_test_policy_id: uuid.UUID):
        """Test race conditions during worker startup and shutdown."""
        eval_task_client = http_env.make_client()

        # Create tasks
        task_ids = []
        for _i in range(3):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=orchestrator_test_policy_id,
                    git_hash=f"lifecycle_race_hash_{_i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_test_worker(worker_name, http_env, delay=0.5)

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.1,
            max_workers=3,
        )

        try:
            # Simulate chaotic worker lifecycle
            for cycle in range(10):
                # Start workers and assign tasks
                await orchestrator.run_cycle()

                if cycle % 3 == 0:
                    # Occasionally kill random workers
                    alive_workers = await worker_manager.discover_alive_workers()
                    if alive_workers:
                        worker_to_kill = alive_workers[0]
                        worker_manager.cleanup_worker(worker_to_kill.name)

                await asyncio.sleep(0.1)

            # Let remaining work complete
            await asyncio.sleep(2.0)

            # Verify system maintained consistency
            filters = TaskFilterParams(policy_ids=[orchestrator_test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            for task_id in task_ids:
                matching_tasks = [task for task in all_tasks.tasks if task.id == task_id]
                assert len(matching_tasks) == 1, f"Task {task_id} should appear exactly once"

                task = matching_tasks[0]
                # Task should be in a valid final state
                assert task.status in ["unprocessed", "done", "error"]

        finally:
            worker_manager.shutdown_all()

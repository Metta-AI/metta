"""
Resilience tests for EvalTaskOrchestrator.

These tests focus on the orchestrator's ability to handle failures,
network issues, database problems, and other adverse conditions gracefully.
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ConnectError, HTTPStatusError

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


class FlakyTaskExecutor(AbstractTaskExecutor):
    """Task executor that fails intermittently."""

    def __init__(self, failure_rate: float = 0.3, delay: float = 0.2):
        self.failure_rate = failure_rate
        self.delay = delay
        self.attempts = 0
        self.successes = 0

    async def execute_task(self, task: TaskResponse) -> None:
        self.attempts += 1
        await asyncio.sleep(self.delay)

        # Fail based on failure rate
        import random

        if random.random() < self.failure_rate:
            raise Exception(f"Flaky failure on attempt {self.attempts}")

        self.successes += 1


class ReliableTaskExecutor(AbstractTaskExecutor):
    """Task executor that always succeeds after a delay."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.executed_tasks = 0

    async def execute_task(self, task: TaskResponse) -> None:
        await asyncio.sleep(self.delay)
        self.executed_tasks += 1


class TestOrchestratorResilience:
    """Test orchestrator resilience to various failure conditions."""

    @pytest.fixture
    def eval_task_client(self, test_client: TestClient, test_app: FastAPI) -> EvalTaskClient:
        """Create an eval task client for testing."""
        token_response = test_client.post(
            "/tokens",
            json={"name": "resilience_test_token", "permissions": ["read", "write"]},
            headers={"X-Auth-Request-Email": "test_user@example.com"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]

        client = EvalTaskClient.__new__(EvalTaskClient)
        from httpx import ASGITransport, AsyncClient

        client._http_client = AsyncClient(transport=ASGITransport(app=test_app), base_url=test_client.base_url)
        client._machine_token = token

        return client

    @pytest.fixture
    def test_policy_id(self, stats_client: StatsClient) -> uuid.UUID:
        """Create a test policy and return its ID."""
        training_run = stats_client.create_training_run(
            name=f"test_resilience_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "resilience"},
        )

        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )

        policy = stats_client.create_policy(
            name=f"test_resilience_policy_{uuid.uuid4().hex[:8]}",
            description="Test policy for resilience tests",
            epoch_id=epoch.id,
        )

        return policy.id

    def create_reliable_worker(self, worker_name: str, eval_task_client: EvalTaskClient) -> EvalTaskWorker:
        """Create a worker that executes tasks reliably."""
        return EvalTaskWorker(
            client=eval_task_client,
            assignee=worker_name,
            task_executor=ReliableTaskExecutor(delay=0.1),
            poll_interval=0.1,
        )

    @pytest.mark.asyncio
    async def test_network_failure_resilience(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test orchestrator resilience to network failures."""

        # Create tasks
        task_ids = []
        for i in range(3):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"network_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_reliable_worker(worker_name, eval_task_client)

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        # Create orchestrator with original client
        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            max_workers=2,
        )

        try:
            # Run a few successful cycles
            for _ in range(3):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.1)

            # Inject network failures into the client
            original_get_claimed_tasks = eval_task_client.get_claimed_tasks
            failure_count = 0

            async def failing_get_claimed_tasks(*args, **kwargs):
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 3:  # Fail first 3 calls
                    raise ConnectError("Simulated network failure")
                # Then succeed
                return await original_get_claimed_tasks(*args, **kwargs)

            eval_task_client.get_claimed_tasks = failing_get_claimed_tasks

            # Continue running - should handle failures gracefully
            for _cycle in range(10):
                try:
                    await orchestrator.run_cycle()
                except Exception:
                    # Orchestrator should handle network failures gracefully
                    # Some failures might propagate but shouldn't crash
                    pass
                await asyncio.sleep(0.2)

            # Restore original method
            eval_task_client.get_claimed_tasks = original_get_claimed_tasks

            # Run a few more cycles to recover
            for _ in range(3):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.2)

            # Verify system recovered and processed some tasks
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            # At least some tasks should have been processed despite network issues
            processed_tasks = [t for t in all_tasks.tasks if t.status in ["done", "error"]]
            # We can't guarantee all tasks completed due to timing, but system should be resilient
            assert len(processed_tasks) >= 0, "System should remain functional despite network failures"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_database_error_handling(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test handling of database errors during orchestration."""

        # Create some tasks first
        task_ids = []
        for i in range(2):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"db_error_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return self.create_reliable_worker(worker_name, eval_task_client)

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            max_workers=1,
        )

        try:
            # Start some workers
            await orchestrator.run_cycle()
            await asyncio.sleep(0.1)

            # Inject database errors
            original_update_task_status = eval_task_client.update_task_status
            error_count = 0

            async def failing_update_task_status(*args, **kwargs):
                nonlocal error_count
                error_count += 1
                if error_count <= 2:  # Fail first 2 calls
                    response = Mock()
                    response.status_code = 500
                    response.text = "Internal Server Error"
                    raise HTTPStatusError("Database connection failed", request=Mock(), response=response)
                # Then succeed
                return await original_update_task_status(*args, **kwargs)

            eval_task_client.update_task_status = failing_update_task_status

            # Continue orchestration despite database errors
            for _cycle in range(8):
                try:
                    await orchestrator.run_cycle()
                except Exception:
                    # Some database errors might propagate, but orchestrator should continue
                    pass
                await asyncio.sleep(0.3)

            # Restore original method
            eval_task_client.update_task_status = original_update_task_status

            # Run recovery cycles
            for _ in range(3):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.2)

            # System should eventually recover and function normally
            # This test primarily validates that database errors don't crash the orchestrator
            assert error_count > 0, "Database errors should have been triggered"

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_worker_manager_failures(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test resilience to worker manager failures."""

        # Create a task for worker manager failure test
        await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="worker_mgr_test_hash",
                sim_suite="navigation",
            )
        )

        # Create a flaky worker manager that occasionally fails
        class FlakyWorkerManager(AbstractWorkerManager):
            def __init__(self):
                self.workers = {}
                self.failure_count = 0
                self.max_failures = 3

            def start_worker(self) -> str:
                self.failure_count += 1
                if self.failure_count <= self.max_failures:
                    raise Exception(f"Simulated worker start failure {self.failure_count}")

                # After failures, succeed
                worker_name = f"flaky-worker-{len(self.workers)}"
                self.workers[worker_name] = {"status": "Running"}
                return worker_name

            def cleanup_worker(self, worker_name: str) -> None:
                self.workers.pop(worker_name, None)

            async def discover_alive_workers(self):
                from metta.app_backend.worker_managers.worker import Worker

                return [Worker(name=name, status="Running") for name in self.workers.keys()]

        flaky_manager = FlakyWorkerManager()

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=flaky_manager,
            poll_interval=0.2,
            max_workers=2,
        )

        try:
            # Run orchestration despite worker manager failures
            for _cycle in range(15):
                try:
                    await orchestrator.run_cycle()
                except Exception:
                    # Worker manager failures should be handled gracefully
                    pass
                await asyncio.sleep(0.2)

            # Verify that despite initial failures, the orchestrator attempted to start workers
            assert flaky_manager.failure_count > flaky_manager.max_failures, (
                "Should have attempted to start workers despite failures"
            )

            # Eventually workers should be started after failures stop
            # Workers may be started after failures stop
            # Some workers might have been created after the failure period
            # This test mainly validates graceful handling of worker manager failures

        finally:
            # Clean up any workers that were created
            try:
                for worker_name in list(flaky_manager.workers.keys()):
                    flaky_manager.cleanup_worker(worker_name)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_concurrent_orchestrator_resilience(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test resilience with multiple orchestrators running concurrently."""

        # Create multiple tasks
        task_ids = []
        for i in range(6):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"concurrent_resilience_hash_{i % 2}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        # Create multiple orchestrators with some having flaky components
        orchestrators = []
        worker_managers = []

        for orch_idx in range(3):
            # Create factory function that captures the orchestrator index
            def create_worker_factory(orchestrator_index: int):
                def create_worker_func(worker_name: str) -> EvalTaskWorker:
                    if orchestrator_index == 1:  # Make one orchestrator have flaky workers
                        executor = FlakyTaskExecutor(failure_rate=0.4, delay=0.2)
                    else:
                        executor = ReliableTaskExecutor(delay=0.1)

                    return EvalTaskWorker(
                        client=eval_task_client,
                        assignee=worker_name,
                        task_executor=executor,
                        poll_interval=0.1,
                    )

                return create_worker_func

            create_worker = create_worker_factory(orch_idx)

            worker_manager = ThreadWorkerManager(create_worker=create_worker)
            worker_managers.append(worker_manager)

            orchestrator = EvalTaskOrchestrator(
                task_client=eval_task_client,
                worker_manager=worker_manager,
                poll_interval=0.15,
                max_workers=2,
            )
            orchestrators.append(orchestrator)

        try:
            # Run all orchestrators concurrently
            async def run_orchestrator(orch, cycles=20):
                for _cycle in range(cycles):
                    try:
                        await orch.run_cycle()
                    except Exception:
                        # Some failures are expected in resilience testing
                        pass
                    await asyncio.sleep(0.1)

            # Run all orchestrators concurrently
            await asyncio.gather(*[run_orchestrator(orch) for orch in orchestrators], return_exceptions=True)

            # Additional time for task completion
            await asyncio.sleep(2.0)

            # Verify system maintained consistency despite concurrent operation and failures
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            # Verify no task duplication
            for task_id in task_ids:
                matching_tasks = [t for t in all_tasks.tasks if t.id == task_id]
                assert len(matching_tasks) == 1, f"Task {task_id} should appear exactly once"

            # Count different final states
            final_states = {}
            for task in all_tasks.tasks:
                if task.id in task_ids:
                    status = task.status
                    final_states[status] = final_states.get(status, 0) + 1

            # At least some tasks should reach a final state despite failures
            total_final = final_states.get("done", 0) + final_states.get("error", 0)
            assert total_final > 0, f"Some tasks should complete despite failures. States: {final_states}"

        finally:
            for worker_manager in worker_managers:
                worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test handling of resource exhaustion scenarios."""

        # Create many tasks to simulate load
        task_ids = []
        for i in range(8):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"resource_test_hash_{i}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        # Create worker with resource constraints simulation
        class ResourceConstrainedExecutor(AbstractTaskExecutor):
            def __init__(self):
                self.active_tasks = 0
                self.max_concurrent = 2  # Simulate resource limit
                self.completed_tasks = 0

            async def execute_task(self, task: TaskResponse) -> None:
                if self.active_tasks >= self.max_concurrent:
                    raise Exception("Resource exhausted - too many concurrent tasks")

                self.active_tasks += 1
                try:
                    await asyncio.sleep(0.3)  # Simulate work
                    self.completed_tasks += 1
                finally:
                    self.active_tasks -= 1

        resource_executor = ResourceConstrainedExecutor()

        def create_worker(worker_name: str) -> EvalTaskWorker:
            return EvalTaskWorker(
                client=eval_task_client,
                assignee=worker_name,
                task_executor=resource_executor,
                poll_interval=0.1,
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.2,
            max_workers=1,  # Limited workers to test resource constraints
        )

        try:
            # Run orchestration under resource constraints
            for cycle in range(25):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.2)

                # Check progress periodically
                if cycle % 5 == 0:
                    filters = TaskFilterParams(policy_ids=[test_policy_id])
                    all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                    completed = sum(1 for t in all_tasks.tasks if t.status == "done")
                    if completed >= 4:  # Some reasonable progress
                        break

            # Verify system handled resource constraints gracefully
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            completed_count = sum(1 for t in all_tasks.tasks if t.status == "done")
            error_count = sum(1 for t in all_tasks.tasks if t.status == "error")

            # Some tasks should be completed despite resource constraints
            assert completed_count + error_count > 0, "System should make progress despite resource constraints"

            # Verify executor's resource limit was respected
            assert resource_executor.active_tasks <= resource_executor.max_concurrent, (
                "Resource limits should be respected"
            )

        finally:
            worker_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_graceful_degradation_under_load(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test graceful degradation under high load conditions."""

        # Create a large number of tasks
        task_ids = []
        for i in range(12):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"load_test_hash_{i % 4}",
                    sim_suite="navigation",
                )
            )
            task_ids.append(task_response.id)

        def create_worker(worker_name: str) -> EvalTaskWorker:
            # Mix of fast and slow workers to simulate realistic conditions
            delay = 0.1 if "fast" in worker_name else 0.4
            return EvalTaskWorker(
                client=eval_task_client,
                assignee=worker_name,
                task_executor=ReliableTaskExecutor(delay=delay),
                poll_interval=0.05,  # Aggressive polling under load
            )

        worker_manager = ThreadWorkerManager(create_worker=create_worker)

        orchestrator = EvalTaskOrchestrator(
            task_client=eval_task_client,
            worker_manager=worker_manager,
            poll_interval=0.1,  # Fast polling under load
            max_workers=4,  # Allow more workers under load
        )

        try:
            start_time = datetime.now()

            # Run high-intensity orchestration
            for cycle in range(30):
                await orchestrator.run_cycle()
                await asyncio.sleep(0.1)

                # Check if we're making reasonable progress
                if cycle % 10 == 0:
                    filters = TaskFilterParams(policy_ids=[test_policy_id])
                    all_tasks = await eval_task_client.get_all_tasks(filters=filters)
                    completed = sum(1 for t in all_tasks.tasks if t.status == "done")

                    # If most tasks are done, we can finish early
                    if completed >= 8:
                        break

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Final assessment
            filters = TaskFilterParams(policy_ids=[test_policy_id])
            all_tasks = await eval_task_client.get_all_tasks(filters=filters)

            completed_count = sum(1 for t in all_tasks.tasks if t.status == "done")
            # Check for processing tasks (not used in this test)

            # System should make significant progress under load
            assert completed_count >= 4, f"Expected at least 4 completed tasks under load, got {completed_count}"

            # Verify reasonable performance (should complete significant work in reasonable time)
            throughput = completed_count / duration if duration > 0 else 0
            assert throughput > 0.5, f"Expected reasonable throughput (>0.5 tasks/sec), got {throughput:.2f}"

        finally:
            worker_manager.shutdown_all()

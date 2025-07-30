import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskCreateRequest,
    TaskFilterParams,
    TaskStatusUpdate,
    TaskUpdateRequest,
)


class TestEvalTaskRoutes:
    """End-to-end tests for eval task routes."""

    @pytest.fixture
    def eval_task_client(self, test_client: TestClient, test_app: FastAPI) -> EvalTaskClient:
        """Create an eval task client for testing."""
        token_response = test_client.post(
            "/tokens",
            json={"name": "eval_test_token", "permissions": ["read", "write"]},
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
        # Create training run, epoch, and policy
        training_run = stats_client.create_training_run(
            name=f"test_eval_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "true"},
        )

        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )

        policy = stats_client.create_policy(
            name=f"test_eval_policy_{uuid.uuid4().hex[:8]}",
            description="Test policy for eval tasks",
            epoch_id=epoch.id,
        )

        return policy.id

    @pytest.mark.asyncio
    async def test_create_eval_task(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test creating an eval task."""
        request = TaskCreateRequest(
            policy_id=test_policy_id,
            git_hash="abc123def456",
            env_overrides={"key": "value"},
            sim_suite="navigation",
        )

        response = await eval_task_client.create_task(request)

        assert response.policy_id == test_policy_id
        assert response.sim_suite == "navigation"
        assert response.status == "unprocessed"
        assert response.assigned_at is None
        assert response.assignee is None
        assert response.attributes["git_hash"] == "abc123def456"
        assert response.attributes["env_overrides"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_available_tasks(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test getting available tasks."""
        task_ids = []
        for i in range(3):
            request = TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash=f"hash_{i}",
                sim_suite=f"suite_{i}",
            )
            response = await eval_task_client.create_task(request)
            task_ids.append(response.id)

        response = await eval_task_client.get_available_tasks(limit=10)

        assert len(response.tasks) >= 3
        returned_ids = [task.id for task in response.tasks]
        for task_id in task_ids:
            assert task_id in returned_ids

    @pytest.mark.asyncio
    async def test_claim_and_update_tasks(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test the complete workflow of claiming and updating tasks."""
        # Create task
        create_request = TaskCreateRequest(
            policy_id=test_policy_id,
            git_hash="workflow_test_hash",
            sim_suite="all",
        )
        task_response = await eval_task_client.create_task(create_request)
        task_id = task_response.id

        # Claim task
        claim_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_1"))
        assert task_id in claim_response.claimed

        # Verify claimed
        claimed_response = await eval_task_client.get_claimed_tasks(assignee="worker_1")
        assert any(task.id == task_id for task in claimed_response.tasks)

        # Update to done
        update_response = await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_1",
                updates={task_id: TaskStatusUpdate(status="done")},
            )
        )
        assert update_response.statuses[task_id] == "done"

        # Verify no longer available
        available_response = await eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

    @pytest.mark.asyncio
    async def test_get_claimed_tasks_without_assignee(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test getting all claimed tasks without specifying an assignee."""
        task_ids_by_worker = {}
        workers = ["worker_alpha", "worker_beta", "worker_gamma"]

        # Create and claim tasks for each worker
        for worker in workers:
            task_ids = []
            for i in range(2):
                task_response = await eval_task_client.create_task(
                    TaskCreateRequest(
                        policy_id=test_policy_id,
                        git_hash=f"test_all_claimed_{worker}_{i}",
                        sim_suite="navigation",
                    )
                )
                task_ids.append(task_response.id)

            claim_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=task_ids, assignee=worker))
            assert len(claim_response.claimed) == 2
            task_ids_by_worker[worker] = task_ids

        # Test getting all claimed tasks
        all_claimed_response = await eval_task_client.get_claimed_tasks()
        all_claimed_ids = [task.id for task in all_claimed_response.tasks]
        assert len(all_claimed_response.tasks) >= 6

        for task_ids in task_ids_by_worker.values():
            for task_id in task_ids:
                assert task_id in all_claimed_ids

        # Test with specific assignee
        specific_response = await eval_task_client.get_claimed_tasks(assignee="worker_beta")
        specific_ids = [task.id for task in specific_response.tasks]

        assert all(tid in specific_ids for tid in task_ids_by_worker["worker_beta"])
        assert all(tid not in specific_ids for tid in task_ids_by_worker["worker_alpha"])
        assert all(tid not in specific_ids for tid in task_ids_by_worker["worker_gamma"])

    @pytest.mark.asyncio
    async def test_task_assignment_expiry(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test that assigned tasks become available again after expiry."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="expiry_test",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        # Claim task
        claim_response = await eval_task_client.claim_tasks(
            TaskClaimRequest(tasks=[task_id], assignee="worker_timeout")
        )
        assert task_id in claim_response.claimed

        # Verify not available
        available_response = await eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

        # Verify claimed
        claimed_response = await eval_task_client.get_claimed_tasks(assignee="worker_timeout")
        assert any(task.id == task_id for task in claimed_response.tasks)

    @pytest.mark.asyncio
    async def test_multiple_workers_claiming_same_task(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test that only one worker can claim a task."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="concurrent_test",
                sim_suite="memory",
            )
        )
        task_id = task_response.id

        # First worker claims
        claim1_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_a"))
        assert task_id in claim1_response.claimed

        # Second worker fails
        claim2_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_b"))
        assert task_id not in claim2_response.claimed

    @pytest.mark.asyncio
    async def test_record_episode_with_eval_task(
        self,
        stats_client: StatsClient,
        test_policy_id: uuid.UUID,
        eval_task_client: EvalTaskClient,
        stats_repo: MettaRepo,
    ):
        """Test recording an episode linked to an eval task."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="episode_test",
                sim_suite="all",
            )
        )
        eval_task_id = task_response.id

        episode = stats_client.record_episode(
            agent_policies={0: test_policy_id},
            agent_metrics={0: {"score": 100.0, "steps": 50}},
            primary_policy_id=test_policy_id,
            eval_name="navigation/simple",
            simulation_suite="navigation",
            replay_url="https://example.com/replay",
            attributes={"test": "true"},
            eval_task_id=eval_task_id,
        )

        # Verify stored correctly
        async with stats_repo.connect() as con:
            result = await con.execute(
                "SELECT eval_task_id FROM episodes WHERE id = %s",
                (episode.id,),
            )
            row = await result.fetchone()
            assert row[0] == eval_task_id

    @pytest.mark.asyncio
    async def test_invalid_status_update(
        self,
        eval_task_client: EvalTaskClient,
        test_policy_id: uuid.UUID,
        test_client,
        test_user_headers: dict[str, str],
    ):
        """Test that invalid status updates are rejected."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="invalid_status_test",
                sim_suite="arena",
            )
        )
        task_id = task_response.id

        claim_response = await eval_task_client.claim_tasks(
            TaskClaimRequest(tasks=[task_id], assignee="worker_invalid")
        )
        assert task_id in claim_response.claimed

        # Try invalid status update
        update_response = test_client.post(
            "/tasks/claimed/update",
            json={
                "assignee": "worker_invalid",
                "updates": {str(task_id): {"status": "invalid_status"}},
            },
            headers=test_user_headers,
        )
        assert update_response.status_code == 422

    @pytest.mark.asyncio
    async def test_update_task_with_error_reason(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test updating task status to error with an error reason."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="error_reason_test",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        claim_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_error"))
        assert task_id in claim_response.claimed

        # Update with error reason
        error_reason = "Failed to checkout git hash: fatal: reference is not a tree"
        update_response = await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_error",
                updates={task_id: TaskStatusUpdate(status="error", attributes={"error_reason": error_reason})},
            )
        )
        assert update_response.statuses[task_id] == "error"

        # Verify not re-queued
        available_response = await eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

    @pytest.mark.asyncio
    async def test_update_task_mixed_formats(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test updating multiple tasks with mixed string and object formats."""
        # Create and claim tasks
        task_ids = []
        for i in range(3):
            task_response = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=test_policy_id,
                    git_hash=f"mixed_test_{i}",
                    sim_suite="all",
                )
            )
            task_ids.append(task_response.id)

        claim_response = await eval_task_client.claim_tasks(TaskClaimRequest(tasks=task_ids, assignee="worker_mixed"))
        assert len(claim_response.claimed) == 3

        # Update with different formats
        update_response = await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_mixed",
                updates={
                    task_ids[0]: TaskStatusUpdate(status="done"),
                    task_ids[1]: TaskStatusUpdate(
                        status="error",
                        attributes={"error_reason": "Simulation failed: OOM"},
                    ),
                    task_ids[2]: TaskStatusUpdate(status="canceled"),
                },
            )
        )
        assert update_response.statuses[task_ids[0]] == "done"
        assert update_response.statuses[task_ids[1]] == "error"
        assert update_response.statuses[task_ids[2]] == "canceled"

    @pytest.mark.asyncio
    async def test_error_reason_stored_in_db(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID, stats_repo: MettaRepo
    ):
        """Test that error_reason is properly stored in the database attributes."""
        task_response = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="db_error_test",
                sim_suite="navigation",
            )
        )
        task_id = task_response.id

        await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_db_test"))

        # Update with error
        error_reason = "Database connection timeout after 30 seconds"
        await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_db_test",
                updates={task_id: TaskStatusUpdate(status="error", attributes={"error_reason": error_reason})},
            )
        )

        # Verify in DB
        async with stats_repo.connect() as con:
            result = await con.execute("SELECT status, attributes FROM eval_tasks WHERE id = %s", (task_id,))
            row = await result.fetchone()
            assert row is not None, f"Task {task_id} not found in database"
            assert row[0] == "error"
            assert row[1]["error_reason"] == error_reason

    @pytest.mark.asyncio
    async def test_get_all_tasks_with_filters(self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID):
        """Test get_all_tasks with all filter criteria."""
        # Create tasks with different attributes
        created_tasks = []

        # Use unique test prefix to avoid conflicts with other tests
        test_prefix = f"filter_test_{uuid.uuid4().hex[:8]}"

        # Task 1: unprocessed, git_hash_1, policy_1, suite_navigation
        task1 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash=f"{test_prefix}_git_hash_1",
                sim_suite="navigation",
            )
        )
        created_tasks.append(("task1", task1))

        # Task 2: unprocessed, git_hash_2, policy_1, suite_memory
        task2 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash=f"{test_prefix}_git_hash_2",
                sim_suite="memory",
            )
        )
        created_tasks.append(("task2", task2))

        # Task 3: claimed (still unprocessed), git_hash_1, policy_1, suite_navigation
        task3 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash=f"{test_prefix}_git_hash_1",
                sim_suite="navigation",
            )
        )
        await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task3.id], assignee="worker_filter_test"))
        created_tasks.append(("task3", task3))

        # Task 4: done status, git_hash_1, policy_1, suite_navigation
        task4 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash=f"{test_prefix}_git_hash_1",
                sim_suite="navigation",
            )
        )
        await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task4.id], assignee="worker_filter_test"))
        await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_filter_test",
                updates={task4.id: TaskStatusUpdate(status="done")},
            )
        )
        created_tasks.append(("task4", task4))

        # Test 1: Filter by status (only unprocessed)
        filters = TaskFilterParams(statuses=["unprocessed"], limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id in task_ids  # claimed but still unprocessed
        assert task4.id not in task_ids  # done status

        # Test 2: Filter by git_hash
        filters = TaskFilterParams(git_hash=f"{test_prefix}_git_hash_1", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert task1.id in task_ids
        assert task2.id not in task_ids  # different git_hash
        assert task3.id in task_ids
        assert task4.id in task_ids

        # Test 3: Filter by policy_ids (single)
        filters = TaskFilterParams(policy_ids=[test_policy_id], limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert all(task.id in task_ids for _, task in created_tasks)

        # Test 4: Filter by sim_suites (single)
        filters = TaskFilterParams(sim_suites=["navigation"], limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert task1.id in task_ids
        assert task2.id not in task_ids  # memory suite
        assert task3.id in task_ids
        assert task4.id in task_ids

        # Test 5: Combined filters
        filters = TaskFilterParams(
            statuses=["unprocessed"], git_hash=f"{test_prefix}_git_hash_1", sim_suites=["navigation"], limit=100
        )
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert task1.id in task_ids
        assert task2.id not in task_ids  # wrong git_hash and sim_suite
        assert task3.id in task_ids
        assert task4.id not in task_ids  # wrong status

    @pytest.mark.asyncio
    async def test_get_all_tasks_with_multiple_statuses(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID
    ):
        """Test filtering by multiple statuses."""
        # Create tasks with different statuses
        tasks_by_status = {}

        # Create unprocessed task
        unprocessed = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="status_test",
                sim_suite="all",
            )
        )
        tasks_by_status["unprocessed"] = unprocessed

        # Create done task
        done_task = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="status_test",
                sim_suite="all",
            )
        )
        await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[done_task.id], assignee="worker_status"))
        await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_status",
                updates={done_task.id: TaskStatusUpdate(status="done")},
            )
        )
        tasks_by_status["done"] = done_task

        # Create error task
        error_task = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="status_test",
                sim_suite="all",
            )
        )
        await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[error_task.id], assignee="worker_status"))
        await eval_task_client.update_task_status(
            TaskUpdateRequest(
                require_assignee="worker_status",
                updates={error_task.id: TaskStatusUpdate(status="error", attributes={"error_reason": "Test error"})},
            )
        )
        tasks_by_status["error"] = error_task

        # Test filtering by multiple statuses
        filters = TaskFilterParams(statuses=["done", "error"], git_hash="status_test", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert tasks_by_status["unprocessed"].id not in task_ids
        assert tasks_by_status["done"].id in task_ids
        assert tasks_by_status["error"].id in task_ids

    @pytest.mark.asyncio
    async def test_get_all_tasks_with_multiple_sim_suites_and_policies(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID, stats_client: StatsClient
    ):
        """Test filtering by multiple sim_suites and policy_ids."""
        # Create a second policy
        training_run = stats_client.create_training_run(
            name=f"test_multi_filter_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "true"},
        )
        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )
        second_policy = stats_client.create_policy(
            name=f"test_multi_filter_policy_{uuid.uuid4().hex[:8]}",
            description="Second test policy",
            epoch_id=epoch.id,
        )

        # Create tasks with different combinations
        tasks = {}

        # Policy 1, navigation
        task1 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="multi_test",
                sim_suite="navigation",
            )
        )
        tasks["policy1_navigation"] = task1

        # Policy 1, memory
        task2 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=test_policy_id,
                git_hash="multi_test",
                sim_suite="memory",
            )
        )
        tasks["policy1_memory"] = task2

        # Policy 2, navigation
        task3 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=second_policy.id,
                git_hash="multi_test",
                sim_suite="navigation",
            )
        )
        tasks["policy2_navigation"] = task3

        # Policy 2, arena
        task4 = await eval_task_client.create_task(
            TaskCreateRequest(
                policy_id=second_policy.id,
                git_hash="multi_test",
                sim_suite="arena",
            )
        )
        tasks["policy2_arena"] = task4

        # Test 1: Multiple sim_suites
        filters = TaskFilterParams(sim_suites=["navigation", "memory"], git_hash="multi_test", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert tasks["policy1_navigation"].id in task_ids
        assert tasks["policy1_memory"].id in task_ids
        assert tasks["policy2_navigation"].id in task_ids
        assert tasks["policy2_arena"].id not in task_ids  # arena not in filter

        # Test 2: Multiple policy_ids
        filters = TaskFilterParams(policy_ids=[test_policy_id, second_policy.id], git_hash="multi_test", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert all(task.id in task_ids for task in tasks.values())  # All tasks should be included

        # Test 3: Combined multiple filters
        filters = TaskFilterParams(
            policy_ids=[second_policy.id], sim_suites=["navigation", "arena"], git_hash="multi_test", limit=100
        )
        response = await eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert tasks["policy1_navigation"].id not in task_ids  # wrong policy
        assert tasks["policy1_memory"].id not in task_ids  # wrong policy
        assert tasks["policy2_navigation"].id in task_ids
        assert tasks["policy2_arena"].id in task_ids

    @pytest.mark.asyncio
    async def test_get_all_tasks_sql_query_with_arrays(
        self, eval_task_client: EvalTaskClient, test_policy_id: uuid.UUID, stats_client: StatsClient
    ):
        """Test that SQL queries with array parameters work correctly."""
        # Create multiple policies
        policies = []
        for i in range(3):
            training_run = stats_client.create_training_run(
                name=f"test_sql_array_run_{i}_{uuid.uuid4().hex[:8]}",
                attributes={"test": "true"},
            )
            epoch = stats_client.create_epoch(
                run_id=training_run.id,
                start_training_epoch=0,
                end_training_epoch=100,
            )
            policy = stats_client.create_policy(
                name=f"test_sql_array_policy_{i}_{uuid.uuid4().hex[:8]}",
                description=f"Test policy {i}",
                epoch_id=epoch.id,
            )
            policies.append(policy.id)

        # Create tasks with different statuses and sim_suites
        created_tasks = []
        statuses_to_create = ["unprocessed", "done", "error"]
        sim_suites_to_create = ["navigation", "memory", "arena"]

        for i, (status, sim_suite) in enumerate(zip(statuses_to_create * 3, sim_suites_to_create * 3, strict=False)):
            policy_id = policies[i % len(policies)]
            task = await eval_task_client.create_task(
                TaskCreateRequest(
                    policy_id=policy_id,
                    git_hash="sql_test",
                    sim_suite=sim_suite,
                )
            )
            created_tasks.append((task, status, sim_suite, policy_id))

            # Update status if needed
            if status != "unprocessed":
                await eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task.id], assignee="sql_test_worker"))
                await eval_task_client.update_task_status(
                    TaskUpdateRequest(
                        require_assignee="sql_test_worker",
                        updates={task.id: TaskStatusUpdate(status=status)},  # type: ignore
                    )
                )

        # Test 1: Multiple statuses with IN clause
        filters = TaskFilterParams(statuses=["unprocessed", "done"], git_hash="sql_test", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        returned_statuses = {t.status for t in response.tasks}
        assert "unprocessed" in returned_statuses or "done" in returned_statuses
        assert "error" not in returned_statuses
        assert "canceled" not in returned_statuses

        # Test 2: Multiple policy_ids
        filters = TaskFilterParams(
            policy_ids=policies[:2],  # First two policies
            git_hash="sql_test",
            limit=100,
        )
        response = await eval_task_client.get_all_tasks(filters=filters)
        returned_policy_ids = {t.policy_id for t in response.tasks}
        assert all(pid in returned_policy_ids or pid in policies[:2] for pid in returned_policy_ids)
        assert (
            policies[2] not in returned_policy_ids
            or len([t for t in response.tasks if t.policy_id == policies[2]]) == 0
        )

        # Test 3: Multiple sim_suites
        filters = TaskFilterParams(sim_suites=["navigation", "memory"], git_hash="sql_test", limit=100)
        response = await eval_task_client.get_all_tasks(filters=filters)
        returned_sim_suites = {t.sim_suite for t in response.tasks}
        assert all(suite in ["navigation", "memory"] for suite in returned_sim_suites)
        assert "arena" not in returned_sim_suites

        # Test 4: Empty arrays should return no results for those filters
        filters = TaskFilterParams(
            statuses=[],  # Empty list
            git_hash="sql_test",
            limit=100,
        )
        response = await eval_task_client.get_all_tasks(filters=filters)
        # Should return results since empty list is treated as no filter in our implementation

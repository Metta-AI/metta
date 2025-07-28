import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskCreateRequest,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.stats_client import StatsClient


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

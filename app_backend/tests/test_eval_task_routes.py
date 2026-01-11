import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskCreateRequest,
    TaskFilterParams,
    TaskFinishRequest,
)
from metta.app_backend.test_support.client_adapter import TestClientAdapter


class TestEvalTaskRoutes:
    """End-to-end tests for eval task routes."""

    @pytest.fixture
    def eval_task_client(self, test_client: TestClient) -> EvalTaskClient:
        """Create an eval task client for testing."""
        # Create client without a real token - auth will use X-Auth-Request-Email header
        client = EvalTaskClient(backend_url=str(test_client.base_url), machine_token=None)
        client._http_client = TestClientAdapter.with_softmax_user(test_client)

        return client

    def test_create_eval_task(self, eval_task_client: EvalTaskClient):
        """Test creating an eval task."""
        request = TaskCreateRequest(
            command="metta evaluate navigation",
            git_hash="abc123def456",
            attributes={"sim_suite": "navigation", "key": "value"},
        )

        response = eval_task_client.create_task(request)

        # Basic assertions
        assert response.command == "metta evaluate navigation"
        assert response.git_hash == "abc123def456"
        assert response.status == "unprocessed"
        assert response.assigned_at is None
        assert response.assignee is None
        assert response.attributes is not None
        assert response.attributes.get("sim_suite") == "navigation"
        assert response.attributes.get("key") == "value"

    def test_get_available_tasks(self, eval_task_client: EvalTaskClient):
        """Test getting available tasks."""
        task_ids = []
        for i in range(3):
            request = TaskCreateRequest(
                command=f"metta evaluate suite_{i}",
                git_hash=f"hash_{i}",
                attributes={"sim_suite": f"suite_{i}"},
            )
            response = eval_task_client.create_task(request)
            task_ids.append(response.id)

        response = eval_task_client.get_available_tasks(limit=10)

        assert len(response.tasks) >= 3
        returned_ids = [task.id for task in response.tasks]
        for task_id in task_ids:
            assert task_id in returned_ids

    def test_claim_and_update_tasks(self, eval_task_client: EvalTaskClient):
        """Test the complete workflow of claiming and updating tasks."""
        # Create task
        create_request = TaskCreateRequest(
            command="metta evaluate all",
            git_hash="workflow_test_hash",
            attributes={"sim_suite": "all"},
        )
        task_response = eval_task_client.create_task(create_request)
        task_id = task_response.id

        # Claim task
        claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_1"))
        assert task_id in claim_response.claimed

        # Verify claimed
        claimed_response = eval_task_client.get_claimed_tasks(assignee="worker_1")
        assert any(task.id == task_id for task in claimed_response.tasks)

        # Update to done
        eval_task_client.finish_task(task_id, TaskFinishRequest(task_id=task_id, status="done"))

        # Verify status updated
        claimed_tasks = eval_task_client.get_claimed_tasks(assignee="worker_1")
        updated_task = next((t for t in claimed_tasks.tasks if t.id == task_id), None)
        assert updated_task is None or updated_task.status == "done"

        # Verify no longer available
        available_response = eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

    def test_get_claimed_tasks_without_assignee(self, eval_task_client: EvalTaskClient):
        """Test getting all claimed tasks without specifying an assignee."""
        task_ids_by_worker = {}
        workers = ["worker_alpha", "worker_beta", "worker_gamma"]

        # Create and claim tasks for each worker
        for worker in workers:
            task_ids = []
            for i in range(2):
                task_response = eval_task_client.create_task(
                    TaskCreateRequest(
                        command="metta evaluate navigation",
                        git_hash=f"test_all_claimed_{worker}_{i}",
                        attributes={"sim_suite": "navigation"},
                    )
                )
                task_ids.append(task_response.id)

            claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=task_ids, assignee=worker))
            assert len(claim_response.claimed) == 2
            task_ids_by_worker[worker] = task_ids

        # Test getting all claimed tasks
        all_claimed_response = eval_task_client.get_claimed_tasks()
        all_claimed_ids = [task.id for task in all_claimed_response.tasks]
        assert len(all_claimed_response.tasks) >= 6

        for task_ids in task_ids_by_worker.values():
            for task_id in task_ids:
                assert task_id in all_claimed_ids

        # Test with specific assignee
        specific_response = eval_task_client.get_claimed_tasks(assignee="worker_beta")
        specific_ids = [task.id for task in specific_response.tasks]

        assert all(tid in specific_ids for tid in task_ids_by_worker["worker_beta"])
        assert all(tid not in specific_ids for tid in task_ids_by_worker["worker_alpha"])
        assert all(tid not in specific_ids for tid in task_ids_by_worker["worker_gamma"])

    def test_task_assignment_expiry(self, eval_task_client: EvalTaskClient):
        """Test that assigned tasks become available again after expiry."""
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash="expiry_test",
                attributes={"sim_suite": "navigation"},
            )
        )
        task_id = task_response.id

        # Claim task
        claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_timeout"))
        assert task_id in claim_response.claimed

        # Verify not available
        available_response = eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

        # Verify claimed
        claimed_response = eval_task_client.get_claimed_tasks(assignee="worker_timeout")
        assert any(task.id == task_id for task in claimed_response.tasks)

    @pytest.mark.slow
    def test_multiple_workers_claiming_same_task(self, eval_task_client: EvalTaskClient):
        """Test that only one worker can claim a task."""
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate memory",
                git_hash="concurrent_test",
                attributes={"sim_suite": "memory"},
            )
        )
        task_id = task_response.id

        # First worker claims
        claim1_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_a"))
        assert task_id in claim1_response.claimed

        # Second worker fails
        claim2_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_b"))
        assert task_id not in claim2_response.claimed

    def test_invalid_status_update(
        self,
        eval_task_client: EvalTaskClient,
        test_client: TestClient,
        auth_headers: dict[str, str],
    ):
        """Test that invalid status updates are rejected."""
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate arena",
                git_hash="invalid_status_test",
                attributes={"sim_suite": "arena"},
            )
        )
        task_id = task_response.id

        claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_invalid"))
        assert task_id in claim_response.claimed

        # Try invalid status update
        update_response = test_client.post(
            f"/tasks/{task_id}/finish",
            json={"task_id": task_id, "status": "invalid_status"},
            headers=auth_headers,
        )
        assert update_response.status_code == 422

    def test_update_task_with_error_reason(self, eval_task_client: EvalTaskClient):
        """Test updating task status to error with an error reason."""
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash="error_reason_test",
                attributes={"sim_suite": "navigation"},
            )
        )
        task_id = task_response.id

        claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_error"))
        assert task_id in claim_response.claimed

        # Update with error reason
        error_reason = "Failed to checkout git hash: fatal: reference is not a tree"
        eval_task_client.finish_task(
            task_id, TaskFinishRequest(task_id=task_id, status="error", status_details={"error_reason": error_reason})
        )

        # Verify not re-queued
        available_response = eval_task_client.get_available_tasks()
        assert task_id not in [task.id for task in available_response.tasks]

    def test_update_task_mixed_formats(self, eval_task_client: EvalTaskClient):
        """Test updating multiple tasks with mixed string and object formats."""
        # Create and claim tasks
        task_ids = []
        for i in range(3):
            task_response = eval_task_client.create_task(
                TaskCreateRequest(
                    command="metta evaluate all",
                    git_hash=f"mixed_test_{i}",
                    attributes={"sim_suite": "all"},
                )
            )
            task_ids.append(task_response.id)

        claim_response = eval_task_client.claim_tasks(TaskClaimRequest(tasks=task_ids, assignee="worker_mixed"))
        assert len(claim_response.claimed) == 3

        # Update with different formats
        eval_task_client.finish_task(task_ids[0], TaskFinishRequest(task_id=task_ids[0], status="done"))
        eval_task_client.finish_task(
            task_ids[1],
            TaskFinishRequest(
                task_id=task_ids[1], status="error", status_details={"error_reason": "Simulation failed: OOM"}
            ),
        )
        eval_task_client.finish_task(task_ids[2], TaskFinishRequest(task_id=task_ids[2], status="canceled"))

        # Verify all updated
        all_tasks = eval_task_client.get_all_tasks(TaskFilterParams(limit=100))
        task_map = {t.id: t for t in all_tasks.tasks if t.id in task_ids}
        assert task_map[task_ids[0]].status == "done"
        assert task_map[task_ids[1]].status == "error"
        assert task_map[task_ids[2]].status == "canceled"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_error_reason_stored_in_db(self, eval_task_client: EvalTaskClient, stats_repo: MettaRepo):
        """Test that error_reason is properly stored in the database attributes."""
        task_response = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash="db_error_test",
                attributes={"sim_suite": "navigation"},
            )
        )
        task_id = task_response.id

        eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task_id], assignee="worker_db_test"))

        # Update with error
        error_reason = "Database connection timeout after 30 seconds"
        eval_task_client.finish_task(
            task_id, TaskFinishRequest(task_id=task_id, status="error", status_details={"error_reason": error_reason})
        )

        # Verify in DB - status is now in task_attempts table
        async with stats_repo.connect() as con:
            result = await con.execute(
                """
                SELECT a.status, a.status_details
                FROM eval_tasks t
                JOIN task_attempts a ON t.latest_attempt_id = a.id
                WHERE t.id = %s
                """,
                (task_id,),
            )
            row = await result.fetchone()
            assert row is not None, f"Task {task_id} not found in database"
            assert row[0] == "error"
            assert row[1]["error_reason"] == error_reason

    @pytest.mark.slow
    def test_get_all_tasks_with_filters(self, eval_task_client: EvalTaskClient):
        """Test get_all_tasks with status and git_hash filters."""
        # Create tasks with different attributes
        created_tasks = []

        # Use unique test prefix to avoid conflicts with other tests
        test_prefix = f"filter_test_{uuid.uuid4().hex[:8]}"

        # Task 1: unprocessed, git_hash_1, suite_navigation
        task1 = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash=f"{test_prefix}_git_hash_1",
                attributes={"sim_suite": "navigation"},
            )
        )
        created_tasks.append(("task1", task1))

        # Task 2: unprocessed, git_hash_2, suite_memory
        task2 = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate memory",
                git_hash=f"{test_prefix}_git_hash_2",
                attributes={"sim_suite": "memory"},
            )
        )
        created_tasks.append(("task2", task2))

        # Task 3: claimed (still unprocessed), git_hash_1, suite_navigation
        task3 = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash=f"{test_prefix}_git_hash_1",
                attributes={"sim_suite": "navigation"},
            )
        )
        eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task3.id], assignee="worker_filter_test"))
        created_tasks.append(("task3", task3))

        # Task 4: done status, git_hash_1, suite_navigation
        task4 = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate navigation",
                git_hash=f"{test_prefix}_git_hash_1",
                attributes={"sim_suite": "navigation"},
            )
        )
        eval_task_client.claim_tasks(TaskClaimRequest(tasks=[task4.id], assignee="worker_filter_test"))
        eval_task_client.finish_task(task4.id, TaskFinishRequest(task_id=task4.id, status="done"))
        created_tasks.append(("task4", task4))

        # Test 1: Filter by status (only unprocessed)
        filters = TaskFilterParams(statuses=["unprocessed"], limit=100)
        response = eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id in task_ids  # claimed but still unprocessed
        assert task4.id not in task_ids  # done status

        # Test 2: Filter by git_hash
        filters = TaskFilterParams(git_hash=f"{test_prefix}_git_hash_1", limit=100)
        response = eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert task1.id in task_ids
        assert task2.id not in task_ids  # different git_hash
        assert task3.id in task_ids
        assert task4.id in task_ids

        # Test 3: Combined filters
        filters = TaskFilterParams(statuses=["unprocessed"], git_hash=f"{test_prefix}_git_hash_1", limit=100)
        response = eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]
        assert task1.id in task_ids
        assert task2.id not in task_ids  # wrong git_hash
        assert task3.id in task_ids
        assert task4.id not in task_ids  # wrong status

    @pytest.mark.slow
    def test_get_all_tasks_with_multiple_statuses(self, eval_task_client: EvalTaskClient):
        """Test filtering by multiple statuses."""
        # Create tasks with different statuses
        tasks_by_status = {}

        # Create unprocessed task
        unprocessed = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate all",
                git_hash="status_test",
                attributes={"sim_suite": "all"},
            )
        )
        tasks_by_status["unprocessed"] = unprocessed

        # Create done task
        done_task = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate all",
                git_hash="status_test",
                attributes={"sim_suite": "all"},
            )
        )
        eval_task_client.claim_tasks(TaskClaimRequest(tasks=[done_task.id], assignee="worker_status"))
        eval_task_client.finish_task(done_task.id, TaskFinishRequest(task_id=done_task.id, status="done"))
        tasks_by_status["done"] = done_task

        # Create error task
        error_task = eval_task_client.create_task(
            TaskCreateRequest(
                command="metta evaluate all",
                git_hash="status_test",
                attributes={"sim_suite": "all"},
            )
        )
        eval_task_client.claim_tasks(TaskClaimRequest(tasks=[error_task.id], assignee="worker_status"))
        eval_task_client.finish_task(
            error_task.id,
            TaskFinishRequest(task_id=error_task.id, status="error", status_details={"error_reason": "Test error"}),
        )
        tasks_by_status["error"] = error_task

        # Test filtering by multiple statuses
        filters = TaskFilterParams(statuses=["done", "error"], git_hash="status_test", limit=100)
        response = eval_task_client.get_all_tasks(filters=filters)
        task_ids = [t.id for t in response.tasks]

        assert tasks_by_status["unprocessed"].id not in task_ids
        assert tasks_by_status["done"].id in task_ids
        assert tasks_by_status["error"].id in task_ids

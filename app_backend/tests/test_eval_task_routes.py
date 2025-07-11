import uuid
from typing import Dict

import pytest

from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.stats_client import StatsClient


class TestEvalTaskRoutes:
    """End-to-end tests for eval task routes."""

    @pytest.fixture
    def test_policy_id(self, stats_client: StatsClient) -> str:
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

        return str(policy.id)

    def test_create_eval_task(self, test_client, test_user_headers: Dict[str, str], test_policy_id: str):
        """Test creating an eval task."""
        response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "abc123def456",
                "env_overrides": {"key": "value"},
                "sim_suite": "navigation",
            },
            headers=test_user_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["policy_id"] == test_policy_id
        assert data["sim_suite"] == "navigation"
        assert data["status"] == "unprocessed"
        assert data["assigned_at"] is None
        assert data["assignee"] is None
        assert data["attributes"]["git_hash"] == "abc123def456"
        assert data["attributes"]["env_overrides"] == {"key": "value"}

    def test_get_available_tasks(self, test_client, test_user_headers: Dict[str, str], test_policy_id: str):
        """Test getting available tasks."""
        # Create some tasks
        task_ids = []
        for i in range(3):
            response = test_client.post(
                "/tasks",
                json={
                    "policy_id": test_policy_id,
                    "git_hash": f"hash_{i}",
                    "sim_suite": f"suite_{i}",
                },
                headers=test_user_headers,
            )
            assert response.status_code == 200
            task_ids.append(response.json()["id"])

        # Get available tasks
        response = test_client.get("/tasks/available?limit=10", headers=test_user_headers)
        assert response.status_code == 200

        tasks = response.json()["tasks"]
        # Should include at least the 3 we just created
        assert len(tasks) >= 3

        # Check that our tasks are in the results
        returned_ids = [task["id"] for task in tasks]
        for task_id in task_ids:
            assert task_id in returned_ids

    def test_claim_and_update_tasks(self, test_client, test_user_headers: Dict[str, str], test_policy_id: str):
        """Test the complete workflow of claiming and updating tasks."""
        # Create tasks
        create_response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "workflow_test_hash",
                "sim_suite": "all",
            },
            headers=test_user_headers,
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["id"]

        # Claim the task
        claim_response = test_client.post(
            "/tasks/claim",
            json={
                "eval_task_ids": [task_id],
                "assignee": "worker_1",
            },
            headers=test_user_headers,
        )
        assert claim_response.status_code == 200
        claimed_ids = claim_response.json()
        assert task_id in claimed_ids

        # Get claimed tasks for the assignee
        claimed_response = test_client.get(
            "/tasks/claimed?assignee=worker_1",
            headers=test_user_headers,
        )
        assert claimed_response.status_code == 200
        claimed_tasks = claimed_response.json()["tasks"]
        assert len(claimed_tasks) >= 1
        assert any(task["id"] == task_id for task in claimed_tasks)

        # Update task status to done
        update_response = test_client.post(
            "/tasks/claimed/update",
            json={
                "assignee": "worker_1",
                "statuses": {task_id: "done"},
            },
            headers=test_user_headers,
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated[task_id] == "done"

        # Verify the task is no longer available
        available_response = test_client.get("/tasks/available", headers=test_user_headers)
        assert available_response.status_code == 200
        available_tasks = available_response.json()["tasks"]
        available_ids = [task["id"] for task in available_tasks]
        assert task_id not in available_ids

    def test_task_assignment_expiry(
        self, test_client, test_user_headers: Dict[str, str], test_policy_id: str, stats_repo: MettaRepo
    ):
        """Test that assigned tasks become available again after expiry."""
        # Create a task
        create_response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "expiry_test",
                "sim_suite": "navigation",
            },
            headers=test_user_headers,
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["id"]

        # Claim it
        claim_response = test_client.post(
            "/tasks/claim",
            json={
                "eval_task_ids": [task_id],
                "assignee": "worker_timeout",
            },
            headers=test_user_headers,
        )
        assert claim_response.status_code == 200
        assert task_id in claim_response.json()

        # Verify it's not available immediately
        available_response = test_client.get("/tasks/available", headers=test_user_headers)
        available_ids = [task["id"] for task in available_response.json()["tasks"]]
        assert task_id not in available_ids

        # TODO: In a real test, we would need to mock time or update the database directly
        # to simulate assignment expiry. For now, we'll just verify the claimed task shows up
        # in the claimed tasks list
        claimed_response = test_client.get(
            "/tasks/claimed?assignee=worker_timeout",
            headers=test_user_headers,
        )
        assert claimed_response.status_code == 200
        claimed_tasks = claimed_response.json()["tasks"]
        assert any(task["id"] == task_id for task in claimed_tasks)

    def test_multiple_workers_claiming_same_task(
        self, test_client, test_user_headers: Dict[str, str], test_policy_id: str
    ):
        """Test that only one worker can claim a task."""
        # Create a task
        create_response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "concurrent_test",
                "sim_suite": "memory",
            },
            headers=test_user_headers,
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["id"]

        # First worker claims it
        claim1_response = test_client.post(
            "/tasks/claim",
            json={
                "eval_task_ids": [task_id],
                "assignee": "worker_a",
            },
            headers=test_user_headers,
        )
        assert claim1_response.status_code == 200
        assert task_id in claim1_response.json()

        # Second worker tries to claim it
        claim2_response = test_client.post(
            "/tasks/claim",
            json={
                "eval_task_ids": [task_id],
                "assignee": "worker_b",
            },
            headers=test_user_headers,
        )
        assert claim2_response.status_code == 200
        # Should return empty list since task is already claimed
        assert task_id not in claim2_response.json()

    @pytest.mark.asyncio
    async def test_record_episode_with_eval_task(
        self,
        stats_client: StatsClient,
        test_policy_id: str,
        test_client,
        test_user_headers: Dict[str, str],
        stats_repo: MettaRepo,
    ):
        """Test recording an episode linked to an eval task."""
        # Create an eval task
        create_response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "episode_test",
                "sim_suite": "all",
            },
            headers=test_user_headers,
        )
        assert create_response.status_code == 200
        eval_task_id = create_response.json()["id"]

        # Record an episode with the eval_task_id
        policy_uuid = uuid.UUID(test_policy_id)
        eval_task_uuid = uuid.UUID(eval_task_id)
        episode = stats_client.record_episode(
            agent_policies={0: policy_uuid},
            agent_metrics={0: {"score": 100.0, "steps": 50}},
            primary_policy_id=policy_uuid,
            eval_name="navigation/simple",
            simulation_suite="navigation",
            replay_url="https://example.com/replay",
            attributes={"test": "true"},
            eval_task_id=eval_task_uuid,
        )

        assert episode.id is not None

        # Verify the eval_task_id was set in the episodes table
        async with stats_repo.connect() as con:
            result = await con.execute(
                "SELECT eval_task_id FROM episodes WHERE id = %s",
                (episode.id,),
            )
            episode_row = await result.fetchone()
            assert episode_row is not None
            assert episode_row[0] == eval_task_uuid

    def test_invalid_status_update(self, test_client, test_user_headers: Dict[str, str], test_policy_id: str):
        """Test that invalid status updates are rejected."""
        # Create and claim a task
        create_response = test_client.post(
            "/tasks",
            json={
                "policy_id": test_policy_id,
                "git_hash": "invalid_status_test",
                "sim_suite": "arena",
            },
            headers=test_user_headers,
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["id"]

        claim_response = test_client.post(
            "/tasks/claim",
            json={
                "eval_task_ids": [task_id],
                "assignee": "worker_invalid",
            },
            headers=test_user_headers,
        )
        assert claim_response.status_code == 200

        # Try to update with invalid status
        update_response = test_client.post(
            "/tasks/claimed/update",
            json={
                "assignee": "worker_invalid",
                "statuses": {task_id: "invalid_status"},
            },
            headers=test_user_headers,
        )
        assert update_response.status_code == 400
        assert "Invalid status" in update_response.json()["detail"]

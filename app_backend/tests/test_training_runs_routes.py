import uuid
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient


class TestTrainingRunsRoutes:
    """Tests for the training runs API routes."""

    @pytest.fixture(scope="class")
    def test_training_runs(self, stats_client: StatsClient) -> Dict:
        """Create test training runs with comprehensive data."""
        # Create multiple training runs
        run1 = stats_client.create_training_run(
            name="training_run_1", attributes={"algorithm": "PPO", "env": "test_env"}, url="https://wandb.ai/test/run1"
        )
        run2 = stats_client.create_training_run(
            name="training_run_2", attributes={"algorithm": "SAC", "env": "test_env"}, url=None
        )

        # Create epochs for each run with different training epochs
        epoch1_early = stats_client.create_epoch(
            run_id=run1.id, start_training_epoch=0, end_training_epoch=50, attributes={"lr": "1e-4"}
        )
        epoch1_late = stats_client.create_epoch(
            run_id=run1.id, start_training_epoch=51, end_training_epoch=100, attributes={"lr": "5e-5"}
        )

        epoch2_early = stats_client.create_epoch(
            run_id=run2.id, start_training_epoch=0, end_training_epoch=25, attributes={"lr": "3e-4"}
        )
        epoch2_mid = stats_client.create_epoch(
            run_id=run2.id, start_training_epoch=26, end_training_epoch=75, attributes={"lr": "1e-4"}
        )
        epoch2_late = stats_client.create_epoch(
            run_id=run2.id, start_training_epoch=76, end_training_epoch=120, attributes={"lr": "5e-5"}
        )

        # Create policies for each epoch
        policy1_early = stats_client.create_policy(
            name="run1_policy_epoch50", description="Early policy for run 1", epoch_id=epoch1_early.id
        )
        policy1_late = stats_client.create_policy(
            name="run1_policy_epoch100", description="Late policy for run 1", epoch_id=epoch1_late.id
        )

        policy2_early = stats_client.create_policy(
            name="run2_policy_epoch25", description="Early policy for run 2", epoch_id=epoch2_early.id
        )
        policy2_mid = stats_client.create_policy(
            name="run2_policy_epoch75", description="Mid policy for run 2", epoch_id=epoch2_mid.id
        )
        policy2_late = stats_client.create_policy(
            name="run2_policy_epoch120", description="Late policy for run 2", epoch_id=epoch2_late.id
        )

        # Create episodes with performance data for multiple suites and metrics
        # Test suite data
        test_episodes = [
            # Run 1 episodes
            (
                policy1_early,
                epoch1_early,
                "test_suite/nav_task",
                {"reward": 80.0, "success": 0.8},
                {"agent_groups": {"0": 1}},
            ),
            (
                policy1_early,
                epoch1_early,
                "test_suite/manipulation_task",
                {"reward": 70.0, "success": 0.7},
                {"agent_groups": {"0": 1}},
            ),
            (
                policy1_late,
                epoch1_late,
                "test_suite/nav_task",
                {"reward": 90.0, "success": 0.9},
                {"agent_groups": {"0": 1}},
            ),
            (
                policy1_late,
                epoch1_late,
                "test_suite/manipulation_task",
                {"reward": 85.0, "success": 0.85},
                {"agent_groups": {"0": 1}},
            ),
            # Run 2 episodes
            (
                policy2_early,
                epoch2_early,
                "test_suite/nav_task",
                {"reward": 60.0, "success": 0.6},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_early,
                epoch2_early,
                "test_suite/manipulation_task",
                {"reward": 55.0, "success": 0.55},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_mid,
                epoch2_mid,
                "test_suite/nav_task",
                {"reward": 75.0, "success": 0.75},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_mid,
                epoch2_mid,
                "test_suite/manipulation_task",
                {"reward": 80.0, "success": 0.8},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_late,
                epoch2_late,
                "test_suite/nav_task",
                {"reward": 95.0, "success": 0.95},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_late,
                epoch2_late,
                "test_suite/manipulation_task",
                {"reward": 92.0, "success": 0.92},
                {"agent_groups": {"0": 2}},
            ),
        ]

        # Additional suite data
        object_episodes = [
            (
                policy1_early,
                epoch1_early,
                "object_suite/pickup_task",
                {"reward": 65.0, "success": 0.65},
                {"agent_groups": {"0": 1}},
            ),
            (
                policy1_late,
                epoch1_late,
                "object_suite/pickup_task",
                {"reward": 88.0, "success": 0.88},
                {"agent_groups": {"0": 1}},
            ),
            (
                policy2_mid,
                epoch2_mid,
                "object_suite/pickup_task",
                {"reward": 72.0, "success": 0.72},
                {"agent_groups": {"0": 2}},
            ),
            (
                policy2_late,
                epoch2_late,
                "object_suite/pickup_task",
                {"reward": 91.0, "success": 0.91},
                {"agent_groups": {"0": 2}},
            ),
        ]

        all_episodes = test_episodes + object_episodes

        # Create a mapping of policy objects to their names for episode creation
        policy_names = {
            policy1_early.id: "run1_policy_epoch50",
            policy1_late.id: "run1_policy_epoch100",
            policy2_early.id: "run2_policy_epoch25",
            policy2_mid.id: "run2_policy_epoch75",
            policy2_late.id: "run2_policy_epoch120",
        }

        episode_ids = []
        for policy, epoch, eval_name, metrics, attributes in all_episodes:
            policy_name = policy_names[policy.id]
            episode = stats_client.record_episode(
                agent_policies={0: policy.id},
                agent_metrics={0: metrics},
                primary_policy_id=policy.id,
                stats_epoch=epoch.id,
                sim_name=eval_name,
                env_label="test_env",
                replay_url=f"https://replay.example.com/{policy_name}/{eval_name.replace('/', '_')}",
                attributes=attributes,
            )
            episode_ids.append(episode.id)

        return {
            "runs": [run1, run2],
            "epochs": [epoch1_early, epoch1_late, epoch2_early, epoch2_mid, epoch2_late],
            "policies": {
                "run1_early": policy1_early,
                "run1_late": policy1_late,
                "run2_early": policy2_early,
                "run2_mid": policy2_mid,
                "run2_late": policy2_late,
            },
            "episode_ids": episode_ids,
        }

    def test_get_training_runs_empty(
        self,
        isolated_test_client: TestClient,
        auth_headers: Dict[str, str],
    ) -> None:
        """Test getting training runs when none exist."""
        response = isolated_test_client.get("/training-runs", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "training_runs" in data
        assert isinstance(data["training_runs"], list)

    def test_get_training_runs_list(
        self, test_client: TestClient, test_training_runs: Dict, auth_headers: Dict[str, str]
    ) -> None:
        """Test listing all training runs."""
        response = test_client.get("/training-runs", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert "training_runs" in data
        runs = data["training_runs"]
        assert len(runs) >= 2

        # Find our test runs
        run1 = next((r for r in runs if r["name"] == "training_run_1"), None)
        run2 = next((r for r in runs if r["name"] == "training_run_2"), None)

        assert run1 is not None
        assert run2 is not None

        # Verify run1 structure and data
        assert "id" in run1
        assert run1["name"] == "training_run_1"
        assert run1["status"] == "running"  # Default status
        assert run1["user_id"] == "test_user@example.com"
        assert "created_at" in run1
        assert run1["url"] == "https://wandb.ai/test/run1"

        # Verify run2 structure and data
        assert "id" in run2
        assert run2["name"] == "training_run_2"
        assert run2["status"] == "running"
        assert run2["user_id"] == "test_user@example.com"
        assert "created_at" in run2
        assert run2["url"] is None

    def test_get_specific_training_run(
        self, test_client: TestClient, test_training_runs: Dict, auth_headers: Dict[str, str]
    ) -> None:
        """Test getting a specific training run by ID."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/training-runs/{run1.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == str(run1.id)
        assert data["name"] == "training_run_1"
        assert data["status"] == "running"
        assert data["user_id"] == "test_user@example.com"
        assert "created_at" in data
        assert data["url"] == "https://wandb.ai/test/run1"

    def test_get_training_run_not_found(self, test_client: TestClient, auth_headers: Dict[str, str]) -> None:
        """Test getting a non-existent training run."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = test_client.get(f"/training-runs/{fake_id}", headers=auth_headers)
        assert response.status_code == 404
        assert "Training run not found" in response.json()["detail"]

    def test_get_training_run_invalid_uuid(self, test_client: TestClient, auth_headers: Dict[str, str]) -> None:
        """Test getting a training run with invalid UUID."""
        response = test_client.get("/training-runs/invalid-uuid", headers=auth_headers)
        assert response.status_code == 404

    def test_update_training_run_status(
        self, stats_client: StatsClient, test_client: TestClient, auth_headers: Dict[str, str]
    ) -> None:
        """Test updating training run status."""
        # Create a training run
        training_run = stats_client.create_training_run(name="test_status_update", attributes={"test": "status_update"})

        # Test updating to completed
        response = test_client.patch(
            f"/stats/training-runs/{training_run.id}/status", json={"status": "completed"}, headers=auth_headers
        )
        assert response.status_code == 204

        # Verify the status was updated
        response = test_client.get(f"/training-runs/{training_run.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["finished_at"] is not None  # Should be set when status changes from running

        # Test updating to failed
        response = test_client.patch(
            f"/stats/training-runs/{training_run.id}/status", json={"status": "failed"}, headers=auth_headers
        )
        assert response.status_code == 204

        # Verify the status was updated
        response = test_client.get(f"/training-runs/{training_run.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"

    def test_update_training_run_status_validation(
        self, stats_client: StatsClient, test_client: TestClient, auth_headers: Dict[str, str]
    ) -> None:
        """Test status update validation."""
        # Create a training run
        training_run = stats_client.create_training_run(
            name="test_status_validation", attributes={"test": "status_validation"}
        )

        # Test invalid status value
        response = test_client.patch(
            f"/stats/training-runs/{training_run.id}/status", json={"status": "invalid_status"}, headers=auth_headers
        )
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]

        # Test missing status field
        response = test_client.patch(f"/stats/training-runs/{training_run.id}/status", json={}, headers=auth_headers)
        assert response.status_code == 400
        assert "Missing 'status' field" in response.json()["detail"]

        # Test invalid UUID
        response = test_client.patch(
            "/stats/training-runs/invalid-uuid/status", json={"status": "completed"}, headers=auth_headers
        )
        assert response.status_code == 400
        assert "Invalid UUID format" in response.json()["detail"]

        # Test non-existent training run (the exact error message may vary)

        fake_id = str(uuid.uuid4())  # Generate a random UUID that definitely won't exist
        response = test_client.patch(
            f"/stats/training-runs/{fake_id}/status", json={"status": "completed"}, headers=auth_headers
        )
        assert response.status_code == 404  # Should return 404 for non-existent resource
        assert "not found" in response.json()["detail"].lower()

    def test_update_training_run_status_not_found_error(
        self, test_client: TestClient, auth_headers: Dict[str, str]
    ) -> None:
        """Test that non-existent training run returns proper 'not found' error, not 'Invalid UUID format'."""

        fake_id = str(uuid.uuid4())  # Valid UUID format but non-existent
        response = test_client.patch(
            f"/stats/training-runs/{fake_id}/status", json={"status": "completed"}, headers=auth_headers
        )
        assert response.status_code == 404  # Should be 404 not 400
        assert "not found" in response.json()["detail"].lower()
        assert "Invalid UUID format" not in response.json()["detail"]

    def test_training_failure_updates_status(
        self, stats_client: StatsClient, test_client: TestClient, auth_headers: Dict[str, str]
    ) -> None:
        """Test that training failures can be handled and status updated properly."""
        # Create a training run
        training_run = stats_client.create_training_run(name="test_failure_handling", attributes={"test": "failure"})

        # Get the training run to verify initial status is 'running'
        response = test_client.get(f"/training-runs/{training_run.id}", headers=auth_headers)
        assert response.status_code == 200
        initial_data = response.json()
        assert initial_data["status"] == "running"

        # Simulate what our training failure handling should do:
        # When training fails, it should update the status to 'failed'
        stats_client.update_training_run_status(training_run.id, "failed")

        # Verify the status was updated to 'failed'
        # This demonstrates that the status update mechanism works
        # The actual automatic failure detection is implemented in the train() wrapper
        response = test_client.get(f"/training-runs/{training_run.id}", headers=auth_headers)
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["status"] == "failed"

        # Also test that we can update back to other statuses
        stats_client.update_training_run_status(training_run.id, "completed")
        response = test_client.get(f"/training-runs/{training_run.id}", headers=auth_headers)
        assert response.status_code == 200
        final_data = response.json()
        assert final_data["status"] == "completed"

    def test_stats_client_update_training_run_status(self, stats_client: StatsClient) -> None:
        """Test the StatsClient update_training_run_status method."""
        # Create a training run
        training_run = stats_client.create_training_run(
            name="test_client_status_update", attributes={"test": "client_update"}
        )

        # Test updating status via client (should not raise any exception)
        stats_client.update_training_run_status(training_run.id, "completed")
        stats_client.update_training_run_status(training_run.id, "failed")
        stats_client.update_training_run_status(training_run.id, "running")

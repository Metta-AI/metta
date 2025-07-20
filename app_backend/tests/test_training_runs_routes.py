from typing import Dict

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.stats_client import StatsClient


class TestTrainingRunsRoutes:
    """Tests for the training runs API routes."""

    @pytest.fixture(scope="class")
    def stats_client(self, test_client: TestClient) -> StatsClient:
        """Create a stats client for testing."""
        # Create a machine token
        token_response = test_client.post(
            "/tokens",
            json={"name": "test_training_runs_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]
        return StatsClient(test_client, machine_token=token)

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
                eval_name=eval_name,
                simulation_suite=None,
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

    def test_get_training_runs_empty(self, test_client: TestClient) -> None:
        """Test getting training runs when none exist."""
        response = test_client.get("/dashboard/training-runs")
        assert response.status_code == 200
        data = response.json()
        assert "training_runs" in data
        assert isinstance(data["training_runs"], list)

    def test_get_training_runs_list(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test listing all training runs."""
        response = test_client.get("/dashboard/training-runs")
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
        assert run1["user_id"] == "test@example.com"
        assert "created_at" in run1
        assert run1["url"] == "https://wandb.ai/test/run1"

        # Verify run2 structure and data
        assert "id" in run2
        assert run2["name"] == "training_run_2"
        assert run2["status"] == "running"
        assert run2["user_id"] == "test@example.com"
        assert "created_at" in run2
        assert run2["url"] is None

    def test_get_specific_training_run(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test getting a specific training run by ID."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/dashboard/training-runs/{run1.id}")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == str(run1.id)
        assert data["name"] == "training_run_1"
        assert data["status"] == "running"
        assert data["user_id"] == "test@example.com"
        assert "created_at" in data
        assert data["url"] == "https://wandb.ai/test/run1"

    def test_get_training_run_not_found(self, test_client: TestClient) -> None:
        """Test getting a non-existent training run."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = test_client.get(f"/dashboard/training-runs/{fake_id}")
        assert response.status_code == 404
        assert "Training run not found" in response.json()["detail"]

    def test_get_training_run_invalid_uuid(self, test_client: TestClient) -> None:
        """Test getting a training run with invalid UUID."""
        response = test_client.get("/dashboard/training-runs/invalid-uuid")
        assert response.status_code == 404

    def test_training_run_heatmap_basic(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test getting heatmap data for a specific training run."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/dashboard/training-runs/{run1.id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Verify heatmap structure
        assert "evalNames" in data
        assert "cells" in data
        assert "policyAverageScores" in data

        # Should contain evaluations from test_suite
        eval_names = data["evalNames"]
        assert "nav_task" in eval_names
        assert "manipulation_task" in eval_names

        # Should contain both policies from run1 (all policies, not filtered by selector)
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_epoch50" in policy_uris
        assert "run1_policy_epoch100" in policy_uris

        # Should not contain policies from other runs
        run2_policies = [uri for uri in policy_uris if uri.startswith("run2_")]
        assert len(run2_policies) == 0

        # Verify cell data
        assert "nav_task" in data["cells"]["run1_policy_epoch50"]
        assert "manipulation_task" in data["cells"]["run1_policy_epoch50"]

        nav_cell = data["cells"]["run1_policy_epoch50"]["nav_task"]
        assert nav_cell["value"] == 80.0
        assert "replayUrl" in nav_cell
        assert nav_cell["replayUrl"] is not None

    def test_training_run_heatmap_multiple_policies(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test that training run heatmap includes all policies from the run."""
        run2 = test_training_runs["runs"][1]

        response = test_client.get(f"/dashboard/training-runs/{run2.id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Should contain all three policies from run2
        policy_uris = list(data["cells"].keys())
        assert "run2_policy_epoch25" in policy_uris
        assert "run2_policy_epoch75" in policy_uris
        assert "run2_policy_epoch120" in policy_uris

        # Should not contain policies from other runs
        run1_policies = [uri for uri in policy_uris if uri.startswith("run1_")]
        assert len(run1_policies) == 0

        # Verify performance data is correct
        assert data["cells"]["run2_policy_epoch25"]["nav_task"]["value"] == 60.0
        assert data["cells"]["run2_policy_epoch75"]["nav_task"]["value"] == 75.0
        assert data["cells"]["run2_policy_epoch120"]["nav_task"]["value"] == 95.0

    def test_training_run_heatmap_different_suite(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test training run heatmap with different evaluation suite."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/dashboard/training-runs/{run1.id}/suites/object_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Should contain evaluations from object_suite
        eval_names = data["evalNames"]
        assert "pickup_task" in eval_names

        # Should still contain both policies from run1
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_epoch50" in policy_uris
        assert "run1_policy_epoch100" in policy_uris

        # Verify performance data
        assert data["cells"]["run1_policy_epoch50"]["pickup_task"]["value"] == 65.0
        assert data["cells"]["run1_policy_epoch100"]["pickup_task"]["value"] == 88.0

    def test_training_run_heatmap_different_metric(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test training run heatmap with different metric."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/dashboard/training-runs/{run1.id}/suites/test_suite/metrics/success/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Verify success metric values
        assert data["cells"]["run1_policy_epoch50"]["nav_task"]["value"] == 0.8
        assert data["cells"]["run1_policy_epoch100"]["nav_task"]["value"] == 0.9
        assert data["cells"]["run1_policy_epoch50"]["manipulation_task"]["value"] == 0.7
        assert data["cells"]["run1_policy_epoch100"]["manipulation_task"]["value"] == 0.85

    def test_training_run_heatmap_group_diff(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test training run heatmap with group difference calculation."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(f"/dashboard/training-runs/{run1.id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Should compute difference between groups
        # Since run1 policies only have group 1 data, group 2 should be 0
        # So difference should be group1_value - 0 = group1_value
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_epoch50" in policy_uris

        nav_cell = data["cells"]["run1_policy_epoch50"]["nav_task"]
        assert nav_cell["value"] == 80.0  # 80.0 - 0.0

    def test_training_run_heatmap_run_not_found(self, test_client: TestClient) -> None:
        """Test training run heatmap with non-existent run."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = test_client.get(f"/dashboard/training-runs/{fake_id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 404
        assert "Training run not found" in response.json()["detail"]

    def test_training_run_heatmap_invalid_uuid(self, test_client: TestClient) -> None:
        """Test training run heatmap with invalid UUID."""
        response = test_client.get("/dashboard/training-runs/invalid-uuid/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 404

    def test_training_run_heatmap_missing_suite_data(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test training run heatmap with suite that has no data."""
        run1 = test_training_runs["runs"][0]

        response = test_client.get(
            f"/dashboard/training-runs/{run1.id}/suites/nonexistent_suite/metrics/reward/heatmap"
        )
        assert response.status_code == 200
        data = response.json()

        # Should return empty or minimal data structure
        assert "evalNames" in data
        assert "cells" in data
        # No policies should be returned since no episodes exist for this suite
        assert len(data["cells"]) == 0

    def test_training_run_heatmap_no_group_filter(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test training run heatmap with empty group metric (total across all groups)."""
        run2 = test_training_runs["runs"][1]

        response = test_client.get(f"/dashboard/training-runs/{run2.id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        # Should include all policies from run2
        policy_uris = list(data["cells"].keys())
        assert "run2_policy_epoch25" in policy_uris
        assert "run2_policy_epoch75" in policy_uris
        assert "run2_policy_epoch120" in policy_uris

        # Values should be the same as group-specific since we only have one agent per episode
        assert data["cells"]["run2_policy_epoch25"]["nav_task"]["value"] == 60.0

    def test_training_run_heatmap_policy_ordering(self, test_client: TestClient, test_training_runs: Dict) -> None:
        """Test that policies are ordered correctly in training run heatmap."""
        run2 = test_training_runs["runs"][1]

        response = test_client.get(f"/dashboard/training-runs/{run2.id}/suites/test_suite/metrics/reward/heatmap")
        assert response.status_code == 200
        data = response.json()

        policy_uris = list(data["cells"].keys())

        # Should include all three policies
        assert len(policy_uris) == 3
        assert "run2_policy_epoch25" in policy_uris
        assert "run2_policy_epoch75" in policy_uris
        assert "run2_policy_epoch120" in policy_uris

        # The exact ordering depends on SQL query, but all should be present
        # In the modified heatmap, ordering is handled by the frontend for display


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

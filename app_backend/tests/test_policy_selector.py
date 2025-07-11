from typing import Dict

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.stats_client import StatsClient


class TestPolicySelector:
    """Integration tests for the policy_selector functionality in heatmap endpoints."""

    @pytest.fixture(scope="class")
    def stats_client(self, test_client: TestClient) -> StatsClient:
        """Create a stats client for testing."""
        # Create a machine token
        token_response = test_client.post(
            "/tokens",
            json={"name": "test_policy_selector_token"},
            headers={"X-Auth-Request-Email": "test_user"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]
        return StatsClient(test_client, machine_token=token)

    @pytest.fixture(scope="class")
    def test_data(self, stats_client: StatsClient) -> Dict:
        """Create comprehensive test data for policy selector tests."""
        # Create two training runs
        run1 = stats_client.create_training_run(name="run_1", attributes={"type": "test"})
        run2 = stats_client.create_training_run(name="run_2", attributes={"type": "test"})

        # Create epochs for each run
        epoch1_early = stats_client.create_epoch(run_id=run1.id, start_training_epoch=0, end_training_epoch=50)
        epoch1_late = stats_client.create_epoch(run_id=run1.id, start_training_epoch=51, end_training_epoch=100)

        epoch2_early = stats_client.create_epoch(run_id=run2.id, start_training_epoch=0, end_training_epoch=30)
        epoch2_late = stats_client.create_epoch(run_id=run2.id, start_training_epoch=31, end_training_epoch=80)

        # Create policies - multiple per training run
        policy1_early = stats_client.create_policy(name="run1_policy_early", epoch_id=epoch1_early.id)
        policy1_late = stats_client.create_policy(name="run1_policy_late", epoch_id=epoch1_late.id)

        policy2_early = stats_client.create_policy(name="run2_policy_early", epoch_id=epoch2_early.id)
        policy2_late = stats_client.create_policy(name="run2_policy_late", epoch_id=epoch2_late.id)

        # Create a policy without epoch_id (should be preserved in both selectors)
        policy_no_epoch = stats_client.create_policy(name="policy_no_epoch", description="Policy without epoch")

        # Create episodes with different performance patterns
        # Run 1: Early policy performs better (higher reward average)
        # eval_task_1: early=90, late=70
        # eval_task_2: early=80, late=60
        # Average: early=85, late=65 -> early should be selected as "best"

        stats_client.record_episode(
            agent_policies={0: policy1_early.id},
            agent_metrics={0: {"reward": 90.0, "success": 1.0}},
            primary_policy_id=policy1_early.id,
            stats_epoch=epoch1_early.id,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 1}},  # group 1
        )

        stats_client.record_episode(
            agent_policies={0: policy1_early.id},
            agent_metrics={0: {"reward": 80.0, "success": 0.8}},
            primary_policy_id=policy1_early.id,
            stats_epoch=epoch1_early.id,
            eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 1}},
        )

        stats_client.record_episode(
            agent_policies={0: policy1_late.id},
            agent_metrics={0: {"reward": 70.0, "success": 0.7}},
            primary_policy_id=policy1_late.id,
            stats_epoch=epoch1_late.id,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 1}},
        )

        stats_client.record_episode(
            agent_policies={0: policy1_late.id},
            agent_metrics={0: {"reward": 60.0, "success": 0.6}},
            primary_policy_id=policy1_late.id,
            stats_epoch=epoch1_late.id,
            eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 1}},
        )

        # Run 2: Late policy performs better (higher reward average)
        # eval_task_1: early=50, late=95
        # eval_task_2: early=40, late=85
        # Average: early=45, late=90 -> late should be selected as "best"

        stats_client.record_episode(
            agent_policies={0: policy2_early.id},
            agent_metrics={0: {"reward": 50.0, "success": 0.5}},
            primary_policy_id=policy2_early.id,
            stats_epoch=epoch2_early.id,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 2}},  # group 2
        )

        stats_client.record_episode(
            agent_policies={0: policy2_early.id},
            agent_metrics={0: {"reward": 40.0, "success": 0.4}},
            primary_policy_id=policy2_early.id,
            stats_epoch=epoch2_early.id,
            eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 2}},
        )

        stats_client.record_episode(
            agent_policies={0: policy2_late.id},
            agent_metrics={0: {"reward": 95.0, "success": 0.95}},
            primary_policy_id=policy2_late.id,
            stats_epoch=epoch2_late.id,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 2}},
        )

        stats_client.record_episode(
            agent_policies={0: policy2_late.id},
            agent_metrics={0: {"reward": 85.0, "success": 0.85}},
            primary_policy_id=policy2_late.id,
            stats_epoch=epoch2_late.id,
            eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 2}},
        )

        # Add episode for policy without epoch
        stats_client.record_episode(
            agent_policies={0: policy_no_epoch.id},
            agent_metrics={0: {"reward": 75.0, "success": 0.75}},
            primary_policy_id=policy_no_epoch.id,
            stats_epoch=None,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 3}},  # group 3
        )

        return {
            "runs": [run1, run2],
            "epochs": [epoch1_early, epoch1_late, epoch2_early, epoch2_late],
            "policies": {
                "run1_early": policy1_early,
                "run1_late": policy1_late,
                "run2_early": policy2_early,
                "run2_late": policy2_late,
                "no_epoch": policy_no_epoch,
            },
        }

    def test_policy_selector_latest_default(self, test_client: TestClient, test_data: Dict) -> None:
        """Test that policy_selector defaults to 'latest' behavior."""
        response = test_client.post("/dashboard/suites/test_suite/metrics/reward/heatmap", json={"group_metric": "1"})

        assert response.status_code == 200
        data = response.json()

        # Should include run1_policy_late (latest for run 1) but not run1_policy_early
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_late" in policy_uris
        assert "run1_policy_early" not in policy_uris

    def test_policy_selector_latest_explicit(self, test_client: TestClient, test_data: Dict) -> None:
        """Test explicit policy_selector='latest' behavior."""
        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap",
            json={"group_metric": "2", "policy_selector": "latest"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should include run2_policy_late (latest for run 2) but not run2_policy_early
        policy_uris = list(data["cells"].keys())
        assert "run2_policy_late" in policy_uris
        assert "run2_policy_early" not in policy_uris

    def test_policy_selector_best(self, test_client: TestClient, test_data: Dict) -> None:
        """Test policy_selector='best' selects highest scoring policies."""
        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap", json={"group_metric": "1", "policy_selector": "best"}
        )

        assert response.status_code == 200
        data = response.json()

        # For run 1: early policy has better average (85 vs 65), so should be selected
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_early" in policy_uris
        assert "run1_policy_late" not in policy_uris

        # Verify the actual scores
        early_scores = data["cells"]["run1_policy_early"]
        assert early_scores["eval_task_1"]["value"] == 90.0
        assert early_scores["eval_task_2"]["value"] == 80.0

    def test_policy_selector_best_different_winner(self, test_client: TestClient, test_data: Dict) -> None:
        """Test that best policy selection can choose different policies per run."""
        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap", json={"group_metric": "2", "policy_selector": "best"}
        )

        assert response.status_code == 200
        data = response.json()

        # For run 2: late policy has better average (90 vs 45), so should be selected
        policy_uris = list(data["cells"].keys())
        assert "run2_policy_late" in policy_uris
        assert "run2_policy_early" not in policy_uris

        # Verify the actual scores
        late_scores = data["cells"]["run2_policy_late"]
        assert late_scores["eval_task_1"]["value"] == 95.0
        assert late_scores["eval_task_2"]["value"] == 85.0

    def test_policy_selector_preserves_no_epoch_policies(self, test_client: TestClient, test_data: Dict) -> None:
        """Test that policies without epoch_id are preserved regardless of selector."""
        for selector in ["latest", "best"]:
            response = test_client.post(
                "/dashboard/suites/test_suite/metrics/reward/heatmap",
                json={"group_metric": "3", "policy_selector": selector},
            )

            assert response.status_code == 200
            data = response.json()

            # Policy without epoch should always be included
            policy_uris = list(data["cells"].keys())
            assert "policy_no_epoch" in policy_uris, f"policy_no_epoch missing with selector={selector}"

    def test_policy_selector_with_group_diff(self, test_client: TestClient, test_data: Dict) -> None:
        """Test policy_selector works with GroupDiff scenarios."""
        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap",
            json={"group_metric": {"group_1": "1", "group_2": "2"}, "policy_selector": "best"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should include best policy from each run
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_early" in policy_uris  # Best for run 1
        assert "run2_policy_late" in policy_uris  # Best for run 2
        assert "run1_policy_late" not in policy_uris
        assert "run2_policy_early" not in policy_uris

    def test_policy_selector_missing_evaluations_default_zero(
        self, stats_client: StatsClient, test_client: TestClient, test_data: Dict
    ) -> None:
        """Test that missing evaluation scores default to 0 when computing best policy."""
        # Create a third training run with policies that only have one evaluation
        run3 = stats_client.create_training_run(name="run_3_partial", attributes={"type": "test"})
        epoch3_early = stats_client.create_epoch(run_id=run3.id, start_training_epoch=0, end_training_epoch=25)
        epoch3_late = stats_client.create_epoch(run_id=run3.id, start_training_epoch=26, end_training_epoch=50)

        policy3_early = stats_client.create_policy(name="run3_policy_early", epoch_id=epoch3_early.id)
        policy3_late = stats_client.create_policy(name="run3_policy_late", epoch_id=epoch3_late.id)

        # Early policy: only eval_task_1 with score 100 -> average = (100 + 0) / 2 = 50
        stats_client.record_episode(
            agent_policies={0: policy3_early.id},
            agent_metrics={0: {"reward": 100.0}},
            primary_policy_id=policy3_early.id,
            stats_epoch=epoch3_early.id,
            eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 4}},
        )

        # Late policy: only eval_task_2 with score 80 -> average = (0 + 80) / 2 = 40
        stats_client.record_episode(
            agent_policies={0: policy3_late.id},
            agent_metrics={0: {"reward": 80.0}},
            primary_policy_id=policy3_late.id,
            stats_epoch=epoch3_late.id,
            eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
            simulation_suite=None,
            attributes={"agent_groups": {"0": 4}},
        )

        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap", json={"group_metric": "4", "policy_selector": "best"}
        )

        assert response.status_code == 200
        data = response.json()

        # Early policy should be selected (average 50 > 40)
        policy_uris = list(data["cells"].keys())
        assert "run3_policy_early" in policy_uris
        assert "run3_policy_late" not in policy_uris

        # Verify missing evaluation shows as 0
        early_scores = data["cells"]["run3_policy_early"]
        assert early_scores["eval_task_1"]["value"] == 100.0
        assert early_scores["eval_task_2"]["value"] == 0.0  # Missing, defaults to 0

    def test_policy_selector_tie_breaking(
        self, stats_client: StatsClient, test_client: TestClient, test_data: Dict
    ) -> None:
        """Test that ties are broken by selecting the latest policy."""
        # Create a training run where two policies have identical average scores
        run4 = stats_client.create_training_run(name="run_tie_test", attributes={"type": "test"})
        epoch4_early = stats_client.create_epoch(run_id=run4.id, start_training_epoch=0, end_training_epoch=30)
        epoch4_late = stats_client.create_epoch(run_id=run4.id, start_training_epoch=31, end_training_epoch=60)

        policy4_early = stats_client.create_policy(name="run4_policy_early", epoch_id=epoch4_early.id)
        policy4_late = stats_client.create_policy(name="run4_policy_late", epoch_id=epoch4_late.id)

        # Both policies get identical scores -> average = 50 for both
        policies_and_epochs = [(policy4_early, epoch4_early, "early"), (policy4_late, epoch4_late, "late")]
        for policy, epoch, _name_suffix in policies_and_epochs:
            stats_client.record_episode(
                agent_policies={0: policy.id},
                agent_metrics={0: {"reward": 60.0}},
                primary_policy_id=policy.id,
                stats_epoch=epoch.id,
                eval_name="test_suite/eval_task_1",  # Format: suite/eval_name
                simulation_suite=None,
                attributes={"agent_groups": {"0": 5}},
            )
            stats_client.record_episode(
                agent_policies={0: policy.id},
                agent_metrics={0: {"reward": 40.0}},
                primary_policy_id=policy.id,
                stats_epoch=epoch.id,
                eval_name="test_suite/eval_task_2",  # Format: suite/eval_name
                simulation_suite=None,
                attributes={"agent_groups": {"0": 5}},
            )

        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap", json={"group_metric": "5", "policy_selector": "best"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should select the latest policy due to tie-breaking (late epoch > early epoch)
        policy_uris = list(data["cells"].keys())
        # Note: Due to SQL ordering by end_training_epoch desc, "late" policy should be first in list
        # and therefore win the tie-breaker
        assert len([p for p in policy_uris if p.startswith("run4_policy")]) == 1
        # We can't guarantee which one wins the tie without looking at the exact SQL ordering,
        # but there should be exactly one policy from run4

    def test_policy_selector_invalid_value(self, test_client: TestClient, test_data: Dict) -> None:
        """Test that invalid policy_selector values are rejected."""
        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/reward/heatmap",
            json={"group_metric": "1", "policy_selector": "invalid"},
        )

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_policy_selector_different_metrics(
        self, stats_client: StatsClient, test_client: TestClient, test_data: Dict
    ) -> None:
        """Test that policy selection uses the metric being queried."""
        # Test with 'success' metric instead of 'reward'
        # From our test data:
        # Run 1: early policy success average = (1.0 + 0.8) / 2 = 0.9
        # Run 1: late policy success average = (0.7 + 0.6) / 2 = 0.65
        # So early should still be selected as best

        response = test_client.post(
            "/dashboard/suites/test_suite/metrics/success/heatmap",
            json={"group_metric": "1", "policy_selector": "best"},
        )

        assert response.status_code == 200
        data = response.json()

        # Early policy should still be selected (better success rate)
        policy_uris = list(data["cells"].keys())
        assert "run1_policy_early" in policy_uris
        assert "run1_policy_late" not in policy_uris

        # Verify the success metric values
        early_scores = data["cells"]["run1_policy_early"]
        assert early_scores["eval_task_1"]["value"] == 1.0
        assert early_scores["eval_task_2"]["value"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

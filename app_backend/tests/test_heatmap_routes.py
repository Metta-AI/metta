from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
from metta.app_backend.stats_client import StatsClient


class TestHeatmapRoutes:
    """Integration tests for heatmap routes with cache testing."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Create a PostgreSQL container for testing."""
        try:
            container = PostgresContainer(
                image="postgres:17",
                username="test_user",
                password="test_password",
                dbname="test_db",
                driver=None,
            )
            container.start()
            yield container
            container.stop()
        except Exception as e:
            pytest.skip(f"Failed to start PostgreSQL container: {e}")

    @pytest.fixture(scope="class")
    def db_uri(self, postgres_container: PostgresContainer) -> str:
        """Get the database URI for the test container."""
        return postgres_container.get_connection_url()

    @pytest.fixture(scope="class")
    def stats_repo(self, db_uri: str) -> MettaRepo:
        """Create a MettaRepo instance with the test database."""
        return MettaRepo(db_uri)

    @pytest.fixture(scope="class")
    def test_app(self, stats_repo: MettaRepo) -> FastAPI:
        """Create a test FastAPI app with dependency injection."""
        return create_app(stats_repo)

    @pytest.fixture(scope="class")
    def test_client(self, test_app: FastAPI) -> TestClient:
        """Create a test client."""
        return TestClient(test_app)

    @pytest.fixture(scope="class")
    def stats_client(self, test_client: TestClient) -> StatsClient:
        """Create a stats client for testing."""
        # First create a machine token
        token_response = test_client.post(
            "/tokens",
            json={"name": "test_heatmap_client_token"},
            headers={"X-Auth-Request-Email": "test_user"},
        )
        assert token_response.status_code == 200
        token = token_response.json()["token"]

        return StatsClient(test_client, machine_token=token)

    def _create_test_data(self, stats_client: StatsClient, run_name: str, num_policies: int = 2) -> dict:
        """Create test data for heatmap testing."""
        # Create a training run
        training_run = stats_client.create_training_run(
            name=run_name,
            attributes={"environment": "test_env", "algorithm": "test_alg"},
            url="https://example.com/run",
        )

        # Create epochs with different training epochs
        epoch1 = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
            attributes={"learning_rate": "0.001"},
        )
        
        epoch2 = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=100,
            end_training_epoch=200,
            attributes={"learning_rate": "0.0005"},
        )

        # Create policies
        policies = []
        policy_names = []
        for i in range(num_policies):
            epoch = epoch1 if i == 0 else epoch2
            policy_name = f"policy_{run_name}_{i}"
            policy = stats_client.create_policy(
                name=policy_name,
                description=f"Test policy {i} for {run_name}",
                epoch_id=epoch.id,
            )
            policies.append(policy)
            policy_names.append(policy_name)

        return {
            "training_run": training_run,
            "epochs": [epoch1, epoch2],
            "policies": policies,
            "policy_names": policy_names,
        }

    def _record_episodes(
        self, stats_client: StatsClient, test_data: dict, eval_names: List[str], metric_values: dict
    ) -> None:
        """Record episodes for the test data."""
        for i, policy in enumerate(test_data["policies"]):
            policy_name = test_data["policy_names"][i]
            for eval_name in eval_names:
                # Get the metric value for this policy/eval combination
                # Extract just the environment name from the full eval_name
                env_name = eval_name.split("/")[-1] if "/" in eval_name else eval_name
                metric_key = f"policy_{i}_{env_name}"
                metric_value = metric_values.get(metric_key, 50.0)
                
                
                stats_client.record_episode(
                    agent_policies={0: policy.id},
                    agent_metrics={0: {"reward": metric_value}},
                    primary_policy_id=policy.id,
                    stats_epoch=test_data["epochs"][i % len(test_data["epochs"])].id,
                    eval_name=eval_name,
                    simulation_suite=eval_name.split("/")[0],  # Use the suite name from eval_name
                    replay_url=f"https://example.com/replay/{policy_name}/{eval_name}",
                )

    def test_heatmap_cache_latest_selector(self, test_client: TestClient, stats_client: StatsClient) -> None:
        """Test that the cache works correctly with the 'latest' selector."""
        # Create initial test data
        test_data = self._create_test_data(stats_client, "cache_test_latest")
        
        # Record initial episodes
        eval_names = ["cache_latest_suite/eval_1", "cache_latest_suite/eval_2"]
        initial_metrics = {
            "policy_0_eval_1": 80.0,
            "policy_0_eval_2": 70.0,
            "policy_1_eval_1": 90.0,
            "policy_1_eval_2": 85.0,
        }
        self._record_episodes(stats_client, test_data, eval_names, initial_metrics)

        # First heatmap call - should populate cache
        response1 = test_client.post(
            "/dashboard/suites/cache_latest_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response1.status_code == 200
        heatmap1 = response1.json()

        # Verify initial data structure
        expected_env_names = ["eval_1", "eval_2"]  # Just the environment names, not the full eval_names
        assert "cells" in heatmap1
        assert "policyAverageScores" in heatmap1
        assert "evalNames" in heatmap1
        assert set(heatmap1["evalNames"]) == set(expected_env_names)

        # Record additional episodes (should update cache)
        additional_metrics = {
            "policy_0_eval_1": 85.0,  # Updated value
            "policy_0_eval_2": 75.0,  # Updated value
            "policy_1_eval_1": 95.0,  # Updated value
            "policy_1_eval_2": 90.0,  # Updated value
        }
        self._record_episodes(stats_client, test_data, eval_names, additional_metrics)

        # Second heatmap call - should use cached data with updates
        response2 = test_client.post(
            "/dashboard/suites/cache_latest_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response2.status_code == 200
        heatmap2 = response2.json()

        # Clear cache
        clear_response = test_client.post("/dashboard/clear_heatmap_cache")
        assert clear_response.status_code == 200
        assert clear_response.json()["message"] == "Heatmap cache cleared successfully"

        # Third heatmap call - should rebuild from database
        response3 = test_client.post(
            "/dashboard/suites/cache_latest_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response3.status_code == 200
        heatmap3 = response3.json()

        # Verify that heatmap2 and heatmap3 are identical (new data should replace old data)
        assert heatmap2["cells"] == heatmap3["cells"]
        assert heatmap2["policyAverageScores"] == heatmap3["policyAverageScores"]
        assert heatmap2["evalAverageScores"] == heatmap3["evalAverageScores"]

        # Verify that values were aggregated from initial and additional episodes
        # Since we're using "latest" selector, should use policy from epoch2 (latest)
        policy_1_name = test_data["policy_names"][1]
        # Initial: 90.0, Additional: 95.0 -> Average: (90.0 + 95.0) / 2 = 92.5
        assert heatmap3["cells"][policy_1_name]["eval_1"]["value"] == 92.5
        # Initial: 85.0, Additional: 90.0 -> Average: (85.0 + 90.0) / 2 = 87.5
        assert heatmap3["cells"][policy_1_name]["eval_2"]["value"] == 87.5

    def test_heatmap_cache_best_selector(self, test_client: TestClient, stats_client: StatsClient) -> None:
        """Test that the cache works correctly with the 'best' selector."""
        # Create initial test data
        test_data = self._create_test_data(stats_client, "cache_test_best")
        
        # Record initial episodes where policy_0 is better
        eval_names = ["cache_best_suite/eval_1", "cache_best_suite/eval_2"]
        initial_metrics = {
            "policy_0_eval_1": 95.0,  # Policy 0 is better initially
            "policy_0_eval_2": 90.0,
            "policy_1_eval_1": 80.0,
            "policy_1_eval_2": 75.0,
        }
        self._record_episodes(stats_client, test_data, eval_names, initial_metrics)

        # First heatmap call - should populate cache
        response1 = test_client.post(
            "/dashboard/suites/cache_best_suite/metrics/reward/heatmap",
            json={"policy_selector": "best"}
        )
        assert response1.status_code == 200
        heatmap1 = response1.json()

        # Verify policy_0 is selected (best)
        policy_0_name = test_data["policy_names"][0]
        assert policy_0_name in heatmap1["cells"]
        assert heatmap1["cells"][policy_0_name]["eval_1"]["value"] == 95.0

        # Record additional episodes where policy_1 becomes better
        additional_metrics = {
            "policy_0_eval_1": 85.0,  # Policy 0 gets worse
            "policy_0_eval_2": 80.0,
            "policy_1_eval_1": 98.0,  # Policy 1 gets better
            "policy_1_eval_2": 95.0,
        }
        self._record_episodes(stats_client, test_data, eval_names, additional_metrics)

        # Second heatmap call - should use cached data with updates
        response2 = test_client.post(
            "/dashboard/suites/cache_best_suite/metrics/reward/heatmap",
            json={"policy_selector": "best"}
        )
        assert response2.status_code == 200
        heatmap2 = response2.json()

        # Clear cache
        clear_response = test_client.post("/dashboard/clear_heatmap_cache")
        assert clear_response.status_code == 200

        # Third heatmap call - should rebuild from database
        response3 = test_client.post(
            "/dashboard/suites/cache_best_suite/metrics/reward/heatmap",
            json={"policy_selector": "best"}
        )
        assert response3.status_code == 200
        heatmap3 = response3.json()

        # Verify that heatmap2 and heatmap3 are identical (new data should replace old data)
        assert heatmap2["cells"] == heatmap3["cells"]
        assert heatmap2["policyAverageScores"] == heatmap3["policyAverageScores"]

        # Verify that policy_0 is still selected (best after aggregation)
        # Policy 0 avg: (90.0 + 85.0) / 2 = 87.5
        # Policy 1 avg: (89.0 + 85.0) / 2 = 87.0
        # Policy 0 still has higher average
        policy_0_name = test_data["policy_names"][0]
        assert policy_0_name in heatmap3["cells"]
        # Initial: 95.0, Additional: 85.0 -> Average: (95.0 + 85.0) / 2 = 90.0
        assert heatmap3["cells"][policy_0_name]["eval_1"]["value"] == 90.0
        # Initial: 90.0, Additional: 80.0 -> Average: (90.0 + 80.0) / 2 = 85.0
        assert heatmap3["cells"][policy_0_name]["eval_2"]["value"] == 85.0

    def test_heatmap_cache_data_not_replaced_when_appropriate(
        self, test_client: TestClient, stats_client: StatsClient
    ) -> None:
        """Test that cached data is not replaced when new data doesn't override existing data."""
        # Create test data with different training runs
        test_data1 = self._create_test_data(stats_client, "cache_test_no_replace_1")
        test_data2 = self._create_test_data(stats_client, "cache_test_no_replace_2")
        
        # Record episodes for first training run
        eval_names = ["cache_no_replace_suite/eval_1", "cache_no_replace_suite/eval_2"]
        metrics1 = {
            "policy_0_eval_1": 80.0,
            "policy_0_eval_2": 75.0,
            "policy_1_eval_1": 85.0,
            "policy_1_eval_2": 80.0,
        }
        self._record_episodes(stats_client, test_data1, eval_names, metrics1)

        # Record episodes for second training run
        metrics2 = {
            "policy_0_eval_1": 70.0,
            "policy_0_eval_2": 65.0,
            "policy_1_eval_1": 75.0,
            "policy_1_eval_2": 70.0,
        }
        self._record_episodes(stats_client, test_data2, eval_names, metrics2)

        # First heatmap call - should populate cache with both training runs
        response1 = test_client.post(
            "/dashboard/suites/cache_no_replace_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response1.status_code == 200
        heatmap1 = response1.json()

        # Verify both training runs are included (latest selector picks 1 policy per run)
        assert len(heatmap1["cells"]) == 2  # 1 latest policy from each training run

        # Add more episodes to first training run (should not replace data from second run)
        additional_metrics = {
            "policy_0_eval_1": 90.0,
            "policy_0_eval_2": 85.0,
            "policy_1_eval_1": 95.0,
            "policy_1_eval_2": 90.0,
        }
        self._record_episodes(stats_client, test_data1, eval_names, additional_metrics)

        # Second heatmap call - should preserve data from both runs
        response2 = test_client.post(
            "/dashboard/suites/cache_no_replace_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response2.status_code == 200

        # Clear cache and rebuild
        clear_response = test_client.post("/dashboard/clear_heatmap_cache")
        assert clear_response.status_code == 200

        response3 = test_client.post(
            "/dashboard/suites/cache_no_replace_suite/metrics/reward/heatmap",
            json={"policy_selector": "latest"}
        )
        assert response3.status_code == 200
        heatmap3 = response3.json()

        # Verify that both training runs are still present
        assert len(heatmap3["cells"]) == 2
        
        # Verify that data from second training run was not replaced
        # Only the latest policy from each run (policy 1) should be in the heatmap
        policy_1_run1_name = test_data1["policy_names"][1]  # Latest from run 1
        policy_1_run2_name = test_data2["policy_names"][1]  # Latest from run 2
        assert policy_1_run1_name in heatmap3["cells"]
        assert policy_1_run2_name in heatmap3["cells"]
        
        # The values should be the latest recorded values for each policy
        # Training run 2 policy 1 should keep its original values
        assert heatmap3["cells"][policy_1_run2_name]["eval_1"]["value"] == 75.0
        assert heatmap3["cells"][policy_1_run2_name]["eval_2"]["value"] == 70.0

    def test_clear_heatmap_cache_endpoint(self, test_client: TestClient) -> None:
        """Test the clear_heatmap_cache endpoint directly."""
        response = test_client.post("/dashboard/clear_heatmap_cache")
        assert response.status_code == 200
        assert response.json() == {"message": "Heatmap cache cleared successfully"}

    def test_heatmap_both_selectors_consistency(self, test_client: TestClient, stats_client: StatsClient) -> None:
        """Test that both 'latest' and 'best' selectors work consistently with cache."""
        # Create test data
        test_data = self._create_test_data(stats_client, "consistency_test")
        
        # Record episodes
        eval_names = ["cache_consistency_suite/eval_1", "cache_consistency_suite/eval_2"]
        metrics = {
            "policy_0_eval_1": 80.0,
            "policy_0_eval_2": 70.0,
            "policy_1_eval_1": 90.0,
            "policy_1_eval_2": 85.0,
        }
        self._record_episodes(stats_client, test_data, eval_names, metrics)

        # Test both selectors
        for selector in ["latest", "best"]:
            # First call - populate cache
            response1 = test_client.post(
                "/dashboard/suites/cache_consistency_suite/metrics/reward/heatmap",
                json={"policy_selector": selector}
            )
            assert response1.status_code == 200
            heatmap1 = response1.json()

            # Second call - use cache
            response2 = test_client.post(
                "/dashboard/suites/cache_consistency_suite/metrics/reward/heatmap",
                json={"policy_selector": selector}
            )
            assert response2.status_code == 200
            heatmap2 = response2.json()

            # Clear cache
            clear_response = test_client.post("/dashboard/clear_heatmap_cache")
            assert clear_response.status_code == 200

            # Third call - rebuild from database
            response3 = test_client.post(
                "/dashboard/suites/cache_consistency_suite/metrics/reward/heatmap",
                json={"policy_selector": selector}
            )
            assert response3.status_code == 200
            heatmap3 = response3.json()

            # All three calls should return identical data
            assert heatmap1["cells"] == heatmap2["cells"] == heatmap3["cells"]
            assert heatmap1["policyAverageScores"] == heatmap2["policyAverageScores"] == heatmap3["policyAverageScores"]


if __name__ == "__main__":
    # Simple test runner for debugging
    pytest.main([__file__, "-v", "-s"])
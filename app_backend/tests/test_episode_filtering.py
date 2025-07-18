from typing import Dict

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.stats_client import StatsClient


class TestEpisodeFiltering:
    """Test the episode filtering route in dashboard_routes.py."""

    @pytest.fixture(scope="class")
    def test_data(self, stats_client: StatsClient) -> Dict:
        """Create comprehensive test data for episode filtering."""
        # Create training runs with different attributes
        training_run_1 = stats_client.create_training_run(
            name="training_run_alpha",
            attributes={"algorithm": "PPO", "env": "test_env"},
            url="https://wandb.ai/test/run1",
            description="First training run for testing",
            tags=["experiment", "baseline"],
        )

        training_run_2 = stats_client.create_training_run(
            name="training_run_beta",
            attributes={"algorithm": "SAC", "env": "test_env"},
            url="https://wandb.ai/test/run2",
            description="Second training run for testing",
            tags=["experiment", "variant"],
        )

        # Create epochs for each training run
        epoch_1a = stats_client.create_epoch(
            run_id=training_run_1.id, start_training_epoch=0, end_training_epoch=50, attributes={"lr": "1e-4"}
        )

        epoch_1b = stats_client.create_epoch(
            run_id=training_run_1.id, start_training_epoch=51, end_training_epoch=100, attributes={"lr": "5e-5"}
        )

        epoch_2a = stats_client.create_epoch(
            run_id=training_run_2.id, start_training_epoch=0, end_training_epoch=75, attributes={"lr": "2e-4"}
        )

        # Create policies for each epoch
        policy_1a = stats_client.create_policy(
            name="policy_alpha_early", description="Early policy from alpha run", epoch_id=epoch_1a.id
        )

        policy_1b = stats_client.create_policy(
            name="policy_alpha_late", description="Late policy from alpha run", epoch_id=epoch_1b.id
        )

        policy_2a = stats_client.create_policy(
            name="policy_beta_main", description="Main policy from beta run", epoch_id=epoch_2a.id
        )

        # Create episodes with different attributes
        episode_configs = [
            # Episodes from training_run_alpha
            {
                "policy": policy_1a,
                "epoch": epoch_1a,
                "eval_name": "navigation/basic_nav",
                "simulation_suite": "test_suite",
                "attributes": {"difficulty": "easy", "agent_count": 1},
            },
            {
                "policy": policy_1a,
                "epoch": epoch_1a,
                "eval_name": "navigation/complex_nav",
                "simulation_suite": "test_suite",
                "attributes": {"difficulty": "hard", "agent_count": 2},
            },
            {
                "policy": policy_1b,
                "epoch": epoch_1b,
                "eval_name": "navigation/basic_nav",
                "simulation_suite": "test_suite",
                "attributes": {"difficulty": "easy", "agent_count": 1},
            },
            {
                "policy": policy_1b,
                "epoch": epoch_1b,
                "eval_name": "manipulation/object_pickup",
                "simulation_suite": "manipulation_suite",
                "attributes": {"difficulty": "medium", "agent_count": 1},
            },
            # Episodes from training_run_beta
            {
                "policy": policy_2a,
                "epoch": epoch_2a,
                "eval_name": "navigation/basic_nav",
                "simulation_suite": "test_suite",
                "attributes": {"difficulty": "easy", "agent_count": 1},
            },
            {
                "policy": policy_2a,
                "epoch": epoch_2a,
                "eval_name": "manipulation/object_pickup",
                "simulation_suite": "manipulation_suite",
                "attributes": {"difficulty": "hard", "agent_count": 3},
            },
        ]

        # Create episodes and collect their IDs
        episode_ids = []
        for config in episode_configs:
            episode = stats_client.record_episode(
                agent_policies={0: config["policy"].id},
                agent_metrics={0: {"reward": 100.0, "success": 1.0}},
                primary_policy_id=config["policy"].id,
                stats_epoch=config["epoch"].id,
                eval_name=config["eval_name"],
                simulation_suite=config["simulation_suite"],
                attributes=config["attributes"],
            )
            episode_ids.append(episode.id)

        return {
            "training_runs": [training_run_1, training_run_2],
            "epochs": [epoch_1a, epoch_1b, epoch_2a],
            "policies": [policy_1a, policy_1b, policy_2a],
            "episode_ids": episode_ids,
            "episode_configs": episode_configs,
        }

    def test_episode_filtering_no_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering without any filter query."""
        response = test_client.get("/episodes", headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "episodes" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

        # Check default pagination
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert data["total_count"] >= 6  # At least our test episodes
        assert data["total_pages"] >= 1

        # Verify episode structure
        if data["episodes"]:
            episode = data["episodes"][0]
            required_fields = [
                "id",
                "created_at",
                "primary_policy_id",
                "eval_category",
                "env_name",
                "attributes",
                "policy_name",
                "training_run_id",
                "training_run_name",
                "training_run_user_id",
                "tags",
            ]

            for field in required_fields:
                assert field in episode

    def test_episode_filtering_with_policy_name_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with policy name filter."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "policy_name = 'policy_alpha_early'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only episodes with policy_alpha_early
        assert data["total_count"] == 2  # Two episodes with this policy
        assert len(data["episodes"]) == 2

        for episode in data["episodes"]:
            assert episode["policy_name"] == "policy_alpha_early"

    def test_episode_filtering_with_training_run_name_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with training run name filter."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "training_run_name = 'training_run_alpha'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only episodes from training_run_alpha
        assert data["total_count"] == 4  # Four episodes from this run
        assert len(data["episodes"]) == 4

        for episode in data["episodes"]:
            assert episode["training_run_name"] == "training_run_alpha"

    def test_episode_filtering_with_eval_category_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with eval category filter."""
        response = test_client.get(
            "/episodes", params={"filter_query": "eval_category = 'navigation'"}, headers=test_user_headers
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only navigation episodes
        assert data["total_count"] == 4  # Four navigation episodes
        assert len(data["episodes"]) == 4

        for episode in data["episodes"]:
            assert episode["eval_category"] == "navigation"

    def test_episode_filtering_with_env_name_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with environment name filter."""
        response = test_client.get(
            "/episodes", params={"filter_query": "env_name = 'basic_nav'"}, headers=test_user_headers
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only basic_nav episodes
        assert data["total_count"] == 3  # Three basic_nav episodes
        assert len(data["episodes"]) == 3

        for episode in data["episodes"]:
            assert episode["env_name"] == "basic_nav"

    def test_episode_filtering_with_compound_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with compound AND filter."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "eval_category = 'navigation' AND env_name = 'basic_nav'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only navigation/basic_nav episodes
        assert data["total_count"] == 3  # Three basic_nav episodes
        assert len(data["episodes"]) == 3

        for episode in data["episodes"]:
            assert episode["eval_category"] == "navigation"
            assert episode["env_name"] == "basic_nav"

    def test_episode_filtering_with_simulation_suite_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with simulation suite filter."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "simulation_suite = 'manipulation_suite'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Should return only manipulation_suite episodes
        assert data["total_count"] == 2  # Two manipulation episodes
        assert len(data["episodes"]) == 2

        # Note: simulation_suite is not included in the response fields but is filterable

    def test_episode_filtering_pagination(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with pagination."""
        # Test first page with page size 2
        response = test_client.get("/episodes", params={"page": 1, "page_size": 2}, headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["episodes"]) == 2
        assert data["total_pages"] >= 3  # At least 3 pages for 6+ episodes

        # Test second page
        response = test_client.get("/episodes", params={"page": 2, "page_size": 2}, headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 2
        assert data["page_size"] == 2
        assert len(data["episodes"]) == 2

    def test_episode_filtering_pagination_with_empty_query(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with pagination and empty query (default Episodes.tsx behavior)."""
        # This matches the default behavior in Episodes.tsx where filterQuery starts as ""
        # and pagination is used to display episodes

        # Test first page with default page size (50) and empty filter
        response = test_client.get(
            "/episodes", params={"page": 1, "page_size": 50, "filter_query": ""}, headers=test_user_headers
        )
        assert response.status_code == 200
        data = response.json()

        # Verify pagination structure
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert data["total_count"] >= 6  # At least our test episodes
        assert data["total_pages"] >= 1
        assert len(data["episodes"]) >= 6  # Should return our test episodes

        # Verify episode structure is complete (all fields present)
        if data["episodes"]:
            episode = data["episodes"][0]
            required_fields = [
                "id",
                "created_at",
                "primary_policy_id",
                "eval_category",
                "env_name",
                "attributes",
                "policy_name",
                "training_run_id",
                "training_run_name",
                "training_run_user_id",
                "tags",
            ]
            for field in required_fields:
                assert field in episode, f"Missing field: {field}"

        # Test smaller page size with empty query to verify pagination works
        response = test_client.get(
            "/episodes", params={"page": 1, "page_size": 3, "filter_query": ""}, headers=test_user_headers
        )
        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 1
        assert data["page_size"] == 3
        assert len(data["episodes"]) == 3
        assert data["total_pages"] >= 2  # At least 2 pages for 6+ episodes with page size 3

        # Test second page with empty query
        response = test_client.get(
            "/episodes", params={"page": 2, "page_size": 3, "filter_query": ""}, headers=test_user_headers
        )
        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 2
        assert data["page_size"] == 3
        assert len(data["episodes"]) == 3  # Should have 3 more episodes

        # Verify episodes are ordered by created_at DESC (newest first)
        if len(data["episodes"]) > 1:
            for i in range(len(data["episodes"]) - 1):
                current_time = data["episodes"][i]["created_at"]
                next_time = data["episodes"][i + 1]["created_at"]
                assert current_time >= next_time, "Episodes should be ordered by created_at DESC"

    def test_episode_filtering_with_tags(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test that episode filtering includes tags in the response."""
        # First add some tags to episodes
        episode_ids = test_data["episode_ids"]

        # Add tags to first two episodes
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [str(ep_id) for ep_id in episode_ids[:2]], "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200

        # Add a different tag to the first episode
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [str(episode_ids[0])], "tag": "special_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200

        # Now filter episodes and check tags are included
        response = test_client.get(
            "/episodes",
            params={"page_size": 10},  # Get more episodes
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # Find our tagged episodes
        tagged_episodes = [ep for ep in data["episodes"] if ep["id"] in [str(ep_id) for ep_id in episode_ids[:2]]]
        assert len(tagged_episodes) == 2

        # Check that tags are included
        first_episode = next(ep for ep in tagged_episodes if ep["id"] == str(episode_ids[0]))
        second_episode = next(ep for ep in tagged_episodes if ep["id"] == str(episode_ids[1]))

        assert "test_tag" in first_episode["tags"]
        assert "special_tag" in first_episode["tags"]
        assert "test_tag" in second_episode["tags"]
        assert "special_tag" not in second_episode["tags"]

    def test_episode_filtering_joined_data(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test that episode filtering includes joined policy, epoch, and training run data."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "policy_name = 'policy_alpha_early'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["episodes"]) == 2
        episode = data["episodes"][0]

        # Check policy data
        assert episode["policy_name"] == "policy_alpha_early"

        # Check training run data
        assert episode["training_run_name"] == "training_run_alpha"
        assert episode["training_run_user_id"] == "test_user@example.com"

    def test_episode_filtering_invalid_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with invalid filter query."""
        response = test_client.get(
            "/episodes", params={"filter_query": "invalid_field = 'value'"}, headers=test_user_headers
        )
        # Should return 500 error due to invalid SQL
        assert response.status_code == 500
        assert "Failed to filter episodes" in response.json()["detail"]

    def test_episode_filtering_malformed_sql_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with malformed SQL filter query."""
        response = test_client.get(
            "/episodes", params={"filter_query": "policy_name = 'test AND"}, headers=test_user_headers
        )
        # Should return 500 error due to malformed SQL
        assert response.status_code == 500
        assert "Failed to filter episodes" in response.json()["detail"]

    def test_episode_filtering_empty_result(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering that returns no results."""
        response = test_client.get(
            "/episodes",
            params={"filter_query": "policy_name = 'nonexistent_policy'"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 0
        assert len(data["episodes"]) == 0
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert data["total_pages"] == 0

    def test_episode_filtering_pagination_bounds(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering pagination boundary conditions."""
        # Test page 0 (should be treated as page 1)
        response = test_client.get("/episodes", params={"page": 0, "page_size": 2}, headers=test_user_headers)
        assert response.status_code == 422  # Validation error

        # Test negative page size
        response = test_client.get("/episodes", params={"page": 1, "page_size": 0}, headers=test_user_headers)
        assert response.status_code == 422  # Validation error

        # Test page size over limit
        response = test_client.get("/episodes", params={"page": 1, "page_size": 101}, headers=test_user_headers)
        assert response.status_code == 422  # Validation error

    def test_episode_filtering_authentication_required(self, test_client: TestClient, test_data: Dict) -> None:
        """Test that episode filtering requires authentication."""
        response = test_client.get("/episodes")
        assert response.status_code == 401

    def test_episode_filtering_empty_filter_query(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with empty filter query."""
        response = test_client.get("/episodes", params={"filter_query": ""}, headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        # Should return all episodes (same as no filter)
        assert data["total_count"] >= 6
        assert len(data["episodes"]) >= 6

    def test_episode_filtering_whitespace_filter(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test episode filtering with whitespace-only filter query."""
        response = test_client.get("/episodes", params={"filter_query": "   "}, headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        # Should return all episodes (same as no filter)
        assert data["total_count"] >= 6
        assert len(data["episodes"]) >= 6

    def test_episode_filtering_order_by_created_at(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_data: Dict
    ) -> None:
        """Test that episodes are ordered by created_at DESC."""
        response = test_client.get("/episodes", params={"page_size": 10}, headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()

        # Episodes should be ordered by created_at DESC
        episodes = data["episodes"]
        if len(episodes) > 1:
            for i in range(len(episodes) - 1):
                current_time = episodes[i]["created_at"]
                next_time = episodes[i + 1]["created_at"]
                assert current_time >= next_time  # DESC order


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

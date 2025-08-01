import uuid
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient


class TestEpisodeTagRoutes:
    """Test episode tag routes."""

    @pytest.fixture
    def test_policy_id(self, stats_client: StatsClient) -> str:
        """Create a test policy and return its ID."""
        # Create training run, epoch, and policy
        training_run = stats_client.create_training_run(
            name=f"test_episode_tag_run_{uuid.uuid4().hex[:8]}",
            attributes={"test": "true"},
        )

        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
        )

        policy = stats_client.create_policy(
            name=f"test_episode_tag_policy_{uuid.uuid4().hex[:8]}",
            description="Test policy for episode tags",
            epoch_id=epoch.id,
        )

        return str(policy.id)

    def test_episode_tag_workflow(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_policy_id: str
    ):
        """Test the complete episode tagging workflow."""
        # Create some episodes
        episode1_response = test_client.post(
            "/stats/episodes",
            json={
                "agent_policies": {0: test_policy_id},
                "agent_metrics": {0: {"reward": 100.0, "steps": 50.0}},
                "primary_policy_id": test_policy_id,
                "eval_name": "test_eval/test_env",
                "simulation_suite": "test_suite",
            },
            headers=test_user_headers,
        )
        assert episode1_response.status_code == 200
        episode1_id = episode1_response.json()["id"]

        episode2_response = test_client.post(
            "/stats/episodes",
            json={
                "agent_policies": {0: test_policy_id},
                "agent_metrics": {0: {"reward": 85.0, "steps": 45.0}},
                "primary_policy_id": test_policy_id,
                "eval_name": "test_eval/test_env",
                "simulation_suite": "test_suite",
            },
            headers=test_user_headers,
        )
        assert episode2_response.status_code == 200
        episode2_id = episode2_response.json()["id"]

        episode_ids = [episode1_id, episode2_id]

        # Add tags to episodes
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": episode_ids, "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 2

        # Try to add the same tag again (should not affect any episodes due to constraint)
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": episode_ids, "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 0

        # Add a different tag to one episode
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [episode1_id], "tag": "another_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 1

        # Get tags for episodes
        # FastAPI expects repeated query parameters for lists
        params = []
        for episode_id in episode_ids:
            params.append(("episode_ids", episode_id))

        response = test_client.get(
            "/episodes/tags",
            params=params,
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        tags_by_episode = data["tags_by_episode"]

        # Episode 1 should have both tags
        assert sorted(tags_by_episode[episode1_id]) == ["another_tag", "test_tag"]
        # Episode 2 should have only one tag
        assert tags_by_episode[episode2_id] == ["test_tag"]

        # Remove one tag from both episodes
        response = test_client.post(
            "/episodes/tags/remove",
            json={"episode_ids": episode_ids, "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 2

        # Get tags again
        params = []
        for episode_id in episode_ids:
            params.append(("episode_ids", episode_id))

        response = test_client.get(
            "/episodes/tags",
            params=params,
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        tags_by_episode = data["tags_by_episode"]

        # Episode 1 should have only one tag
        assert tags_by_episode[episode1_id] == ["another_tag"]
        # Episode 2 should have no tags
        assert episode2_id not in tags_by_episode

        # Try to remove a tag that doesn't exist
        response = test_client.post(
            "/episodes/tags/remove",
            json={"episode_ids": episode_ids, "tag": "nonexistent_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 0

    def test_episode_tag_empty_lists(self, test_client: TestClient, test_user_headers: Dict[str, str]):
        """Test episode tagging with empty lists."""
        # Add tags to empty list
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [], "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 0

        # Remove tags from empty list
        response = test_client.post(
            "/episodes/tags/remove",
            json={"episode_ids": [], "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["episodes_affected"] == 0

        # Get tags for empty list
        response = test_client.get(
            "/episodes/tags",
            params=[],
            headers=test_user_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tags_by_episode"] == {}

    def test_episode_tag_invalid_uuid(self, test_client: TestClient, test_user_headers: Dict[str, str]):
        """Test episode tagging with invalid UUIDs."""
        # Add tags with invalid UUID
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": ["invalid-uuid"], "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 422

        # Remove tags with invalid UUID
        response = test_client.post(
            "/episodes/tags/remove",
            json={"episode_ids": ["invalid-uuid"], "tag": "test_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 422

        # Get tags with invalid UUID
        response = test_client.get(
            "/episodes/tags",
            params=[("episode_ids", "invalid-uuid")],
            headers=test_user_headers,
        )
        assert response.status_code == 422

    @pytest.mark.slow
    def test_episode_tag_auth_required(self, test_client: TestClient):
        """Test that episode tag routes require authentication."""
        # Try to add tags without authentication
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [str(uuid.uuid4())], "tag": "test_tag"},
        )
        assert response.status_code == 401

        # Try to remove tags without authentication
        response = test_client.post(
            "/episodes/tags/remove",
            json={"episode_ids": [str(uuid.uuid4())], "tag": "test_tag"},
        )
        assert response.status_code == 401

        # Try to get tags without authentication
        response = test_client.get(
            "/episodes/tags",
            params=[("episode_ids", str(uuid.uuid4()))],
        )
        assert response.status_code == 401

        # Try to get all tags without authentication
        response = test_client.get("/episodes/tags/all")
        assert response.status_code == 401

    @pytest.mark.slow
    def test_get_all_episode_tags(
        self, test_client: TestClient, test_user_headers: Dict[str, str], test_policy_id: str
    ):
        """Test getting all episode tags."""
        # Create some episodes
        episode1_response = test_client.post(
            "/stats/episodes",
            json={
                "agent_policies": {0: test_policy_id},
                "agent_metrics": {0: {"reward": 100.0, "steps": 50.0}},
                "primary_policy_id": test_policy_id,
                "eval_name": "test_eval/test_env",
                "simulation_suite": "test_suite",
            },
            headers=test_user_headers,
        )
        assert episode1_response.status_code == 200
        episode1_id = episode1_response.json()["id"]

        episode2_response = test_client.post(
            "/stats/episodes",
            json={
                "agent_policies": {0: test_policy_id},
                "agent_metrics": {0: {"reward": 85.0, "steps": 45.0}},
                "primary_policy_id": test_policy_id,
                "eval_name": "test_eval/test_env",
                "simulation_suite": "test_suite",
            },
            headers=test_user_headers,
        )
        assert episode2_response.status_code == 200
        episode2_id = episode2_response.json()["id"]

        # Initially no tags
        response = test_client.get("/episodes/tags/all", headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tags" in data

        # Add some tags
        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [episode1_id, episode2_id], "tag": "test_tag_1"},
            headers=test_user_headers,
        )
        assert response.status_code == 200

        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [episode1_id], "tag": "test_tag_2"},
            headers=test_user_headers,
        )
        assert response.status_code == 200

        response = test_client.post(
            "/episodes/tags/add",
            json={"episode_ids": [episode2_id], "tag": "another_tag"},
            headers=test_user_headers,
        )
        assert response.status_code == 200

        # Get all tags
        response = test_client.get("/episodes/tags/all", headers=test_user_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tags" in data

        # Should have the new tags
        all_tags = set(data["tags"])
        assert "another_tag" in all_tags
        assert "test_tag_1" in all_tags
        assert "test_tag_2" in all_tags

        # Tags should be sorted alphabetically
        sorted_tags = sorted(data["tags"])
        assert data["tags"] == sorted_tags

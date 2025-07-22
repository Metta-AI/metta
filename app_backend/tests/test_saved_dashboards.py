import uuid

import pytest
from fastapi.testclient import TestClient

from metta.app_backend.metta_repo import MettaRepo
from tests.base_async_test import BaseAsyncTest


class TestSavedDashboards(BaseAsyncTest):
    """Tests for the saved dashboard functionality."""

    @pytest.fixture(scope="class")
    def user_id(self) -> str:
        """Create a test user ID."""
        return "test_user@example.com"

    @pytest.mark.asyncio
    async def test_create_saved_dashboard(self, stats_repo: MettaRepo, user_id: str) -> None:
        """Test creating a saved dashboard."""
        dashboard_state = {
            "suite": "navigation",
            "metric": "reward",
            "group_metric": "",
            "num_policies_to_show": 20,
        }

        dashboard_id = await stats_repo.create_saved_dashboard(
            user_id=user_id,
            name="Test Dashboard",
            description="A test dashboard",
            dashboard_type="heatmap",
            dashboard_state=dashboard_state,
        )

        assert isinstance(dashboard_id, uuid.UUID)

        # Verify the dashboard was created
        dashboard = await stats_repo.get_saved_dashboard(str(dashboard_id))
        assert dashboard is not None
        assert dashboard["name"] == "Test Dashboard"
        assert dashboard["description"] == "A test dashboard"
        assert dashboard["type"] == "heatmap"
        assert dashboard["dashboard_state"] == dashboard_state

    @pytest.mark.asyncio
    async def test_delete_saved_dashboard(self, stats_repo: MettaRepo, user_id: str) -> None:
        """Test deleting a saved dashboard."""
        dashboard_state = {
            "suite": "navigation",
            "metric": "reward",
            "group_metric": "",
            "num_policies_to_show": 10,
        }

        # Create a test dashboard
        dashboard_id = await stats_repo.create_saved_dashboard(
            user_id=user_id,
            name="Test Dashboard to Delete",
            description="This will be deleted",
            dashboard_type="heatmap",
            dashboard_state=dashboard_state,
        )

        # Verify it exists
        dashboard = await stats_repo.get_saved_dashboard(str(dashboard_id))
        assert dashboard is not None

        # Delete it
        success = await stats_repo.delete_saved_dashboard(user_id, str(dashboard_id))
        assert success is True

        # Verify it's gone
        dashboard = await stats_repo.get_saved_dashboard(str(dashboard_id))
        assert dashboard is None

    @pytest.mark.asyncio
    async def test_update_saved_dashboard(self, stats_repo: MettaRepo, user_id: str) -> None:
        """Test updating a saved dashboard by creating with the same name."""
        initial_state = {
            "suite": "navigation",
            "metric": "reward",
            "group_metric": "",
            "num_policies_to_show": 20,
        }

        updated_state = {
            "suite": "object_use",
            "metric": "success_rate",
            "group_metric": "group1",
            "num_policies_to_show": 15,
        }

        # Create initial dashboard
        dashboard_id1 = await stats_repo.create_saved_dashboard(
            user_id=user_id,
            name="Update Test Dashboard",
            description="Initial description",
            dashboard_type="heatmap",
            dashboard_state=initial_state,
        )

        # Update by creating with same name
        dashboard_id2 = await stats_repo.create_saved_dashboard(
            user_id=user_id,
            name="Update Test Dashboard",
            description="Updated description",
            dashboard_type="heatmap",
            dashboard_state=updated_state,
        )

        # Should be different IDs (no upsert behavior, always creates new row)
        assert dashboard_id1 != dashboard_id2

        # Verify the new dashboard
        dashboard = await stats_repo.get_saved_dashboard(str(dashboard_id2))
        assert dashboard is not None
        assert dashboard["description"] == "Updated description"
        assert dashboard["dashboard_state"] == updated_state

    def test_update_saved_dashboard_route(self, test_client: TestClient, user_id: str) -> None:
        """Test the update saved dashboard API route."""
        initial_state = {
            "suite": "navigation",
            "metric": "reward",
            "group_metric": "",
            "num_policies_to_show": 20,
        }

        updated_state = {
            "suite": "object_use",
            "metric": "success_rate",
            "group_metric": "group1",
            "num_policies_to_show": 15,
        }

        # Create initial dashboard
        create_response = test_client.post(
            "/dashboard/saved",
            json={
                "name": "Route Update Test Dashboard",
                "description": "Initial description",
                "type": "heatmap",
                "dashboard_state": initial_state,
            },
            headers={"X-Auth-Request-Email": user_id},
        )
        assert create_response.status_code == 200
        dashboard_data = create_response.json()
        dashboard_id = dashboard_data["id"]

        # Update the dashboard
        update_response = test_client.put(
            f"/dashboard/saved/{dashboard_id}",
            json={
                "name": "Route Update Test Dashboard",
                "description": "Updated description",
                "type": "heatmap",
                "dashboard_state": updated_state,
            },
            headers={"X-Auth-Request-Email": user_id},
        )
        assert update_response.status_code == 200
        updated_dashboard = update_response.json()

        # Verify the update
        assert updated_dashboard["id"] == dashboard_id
        assert updated_dashboard["description"] == "Updated description"
        assert updated_dashboard["dashboard_state"] == updated_state

        # Try to update with non-existent ID
        fake_id = "00000000-0000-0000-0000-000000000000"
        fake_update_response = test_client.put(
            f"/dashboard/saved/{fake_id}",
            json={
                "name": "Fake Dashboard",
                "description": "Fake description",
                "type": "heatmap",
                "dashboard_state": {},
            },
            headers={"X-Auth-Request-Email": user_id},
        )
        assert fake_update_response.status_code == 404

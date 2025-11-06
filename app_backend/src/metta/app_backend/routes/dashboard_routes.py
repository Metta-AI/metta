import logging
import typing

import fastapi
import pydantic

import metta.app_backend.auth
import metta.app_backend.metta_repo
import metta.app_backend.route_logger

# Set up logging for scorecard performance analysis
logger = logging.getLogger("dashboard_performance")
logger.setLevel(logging.INFO)


class SavedDashboardCreate(pydantic.BaseModel):
    name: str
    description: typing.Optional[str] = None
    type: str
    dashboard_state: typing.Dict[str, typing.Any]


class SavedDashboardResponse(pydantic.BaseModel):
    id: str
    name: str
    description: typing.Optional[str]
    type: str
    dashboard_state: typing.Dict[str, typing.Any]
    created_at: str
    updated_at: str
    user_id: str

    @classmethod
    def from_db(cls, dashboard: metta.app_backend.metta_repo.SavedDashboardRow) -> "SavedDashboardResponse":
        return cls(
            id=str(dashboard.id),
            name=dashboard.name,
            description=dashboard.description,
            type=dashboard.type,
            dashboard_state=dashboard.dashboard_state,
            created_at=dashboard.created_at.isoformat(),
            updated_at=dashboard.updated_at.isoformat(),
            user_id=dashboard.user_id,
        )


class SavedDashboardListResponse(pydantic.BaseModel):
    dashboards: typing.List[SavedDashboardResponse]


def create_dashboard_router(metta_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    """Create a dashboard router with the given StatsRepo instance."""
    router = fastapi.APIRouter(prefix="/dashboard", tags=["dashboard"])

    # Create the user-or-token authentication dependency
    user_or_token = fastapi.Depends(dependency=metta.app_backend.auth.create_user_or_token_dependency(metta_repo))

    @router.get("/saved")
    @metta.app_backend.route_logger.timed_route("list_saved_dashboards")
    async def list_saved_dashboards(user_id: str = user_or_token) -> SavedDashboardListResponse:  # type: ignore[reportUnusedFunction]
        """List all saved dashboards."""
        dashboards = await metta_repo.list_saved_dashboards()
        return SavedDashboardListResponse(
            dashboards=[SavedDashboardResponse.from_db(dashboard) for dashboard in dashboards]
        )

    @router.get("/saved/{dashboard_id}")
    @metta.app_backend.route_logger.timed_route("get_saved_dashboard")
    async def get_saved_dashboard(dashboard_id: str, user_id: str = user_or_token) -> SavedDashboardResponse:  # type: ignore[reportUnusedFunction]
        """Get a specific saved dashboard by ID."""
        dashboard = await metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise fastapi.HTTPException(status_code=404, detail="Dashboard not found")

        return SavedDashboardResponse.from_db(dashboard)

    @router.post("/saved")
    @metta.app_backend.route_logger.timed_route("create_saved_dashboard")
    async def create_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_data: SavedDashboardCreate,
        user_or_token: str = user_or_token,
    ) -> SavedDashboardResponse:
        """Create a new saved dashboard (always creates a new row, even if name is duplicate)."""
        dashboard_id = await metta_repo.create_saved_dashboard(
            user_id=user_or_token,
            name=dashboard_data.name,
            description=dashboard_data.description,
            dashboard_type=dashboard_data.type,
            dashboard_state=dashboard_data.dashboard_state,
        )

        # Fetch the created dashboard to return
        dashboard = await metta_repo.get_saved_dashboard(str(dashboard_id))
        if not dashboard:
            raise fastapi.HTTPException(status_code=500, detail="Failed to create dashboard")

        return SavedDashboardResponse.from_db(dashboard)

    @router.put("/saved/{dashboard_id}")
    @metta.app_backend.route_logger.timed_route("update_saved_dashboard")
    async def update_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str,
        dashboard_state: typing.Dict[str, typing.Any],
        user_or_token: str = user_or_token,
    ) -> SavedDashboardResponse:
        """Update an existing saved dashboard."""
        success = await metta_repo.update_dashboard_state(
            user_id=user_or_token,
            dashboard_id=dashboard_id,
            dashboard_state=dashboard_state,
        )

        if not success:
            raise fastapi.HTTPException(status_code=404, detail="Dashboard not found")

        # Fetch the updated dashboard to return
        dashboard = await metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise fastapi.HTTPException(status_code=500, detail="Failed to fetch updated dashboard")

        return SavedDashboardResponse.from_db(dashboard)

    @router.delete("/saved/{dashboard_id}")
    @metta.app_backend.route_logger.timed_route("delete_saved_dashboard")
    async def delete_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str, user_or_token: str = user_or_token
    ) -> typing.Dict[str, str]:
        """Delete a saved dashboard."""
        success = await metta_repo.delete_saved_dashboard(user_or_token, dashboard_id)
        if not success:
            raise fastapi.HTTPException(status_code=404, detail="Dashboard not found")
        return {"message": "Dashboard deleted successfully"}

    return router

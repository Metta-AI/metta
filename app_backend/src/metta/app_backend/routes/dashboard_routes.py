import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route

# Set up logging for heatmap performance analysis
logger = logging.getLogger("dashboard_performance")
logger.setLevel(logging.INFO)


class SavedDashboardCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    dashboard_state: Dict[str, Any]


class SavedDashboardResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    type: str
    dashboard_state: Dict[str, Any]
    created_at: str
    updated_at: str
    user_id: str


class SavedDashboardListResponse(BaseModel):
    dashboards: List[SavedDashboardResponse]


class TrainingRun(BaseModel):
    id: str
    name: str
    created_at: str
    user_id: str
    finished_at: Optional[str]
    status: str
    url: Optional[str]
    description: Optional[str]
    tags: List[str]


class TrainingRunListResponse(BaseModel):
    training_runs: List[TrainingRun]


class TrainingRunDescriptionUpdate(BaseModel):
    description: str


class TrainingRunTagsUpdate(BaseModel):
    tags: List[str]


# ============================================================================
# Metrics Cache Implementation
# ============================================================================

GET_ALL_METRICS_QUERY = """
    SELECT DISTINCT eam.metric
    FROM episode_agent_metrics eam
    WHERE eam.episode_internal_id > %s
    ORDER BY eam.metric
"""

GET_MAX_EPISODE_QUERY = "SELECT MAX(internal_id) FROM episodes"


@dataclass
class CachedMetrics:
    """Cached metrics data with metadata."""

    last_episode_id: int
    metrics: List[str]


class MetricsCache:
    """Cache for all distinct metrics."""

    def __init__(self, metta_repo: MettaRepo):
        self.metta_repo = metta_repo
        self.cache: Optional[CachedMetrics] = None

    async def get(self) -> List[str]:
        """Get all distinct metrics, using cache when possible."""
        async with self.metta_repo.connect() as con:
            cached = self.cache

            if cached:
                # Check for new data
                new_max_episode_id = await self._get_max_episode_id(con)

                if new_max_episode_id > cached.last_episode_id:
                    # Update cache with fresh data
                    cached = await self._build_cache_entry(con, cached.last_episode_id)

                return cached.metrics
            else:
                # Build new cache entry
                cached = await self._build_cache_entry(con, 0)
                return cached.metrics

    async def _build_cache_entry(self, con: AsyncConnection, min_episode_id: int) -> CachedMetrics:
        """Build a new cache entry, optionally merging with existing cache."""
        if min_episode_id == 0:
            # Initial cache build - fetch all metrics
            metrics_rows = await execute_query_and_log(
                con,
                "SELECT DISTINCT eam.metric FROM episode_agent_metrics eam ORDER BY eam.metric",
                (),
                "get_all_metrics_initial",
            )
            new_metrics = [row[0] for row in metrics_rows if row[0] is not None]
            all_metrics = new_metrics
        else:
            # Incremental update - fetch only new metrics and merge with existing
            metrics_rows = await execute_query_and_log(con, GET_ALL_METRICS_QUERY, (min_episode_id,), "get_new_metrics")
            new_metrics = [row[0] for row in metrics_rows if row[0] is not None]

            # Merge with existing cached metrics
            existing_metrics = self.cache.metrics if self.cache else []
            all_metrics = sorted(set(existing_metrics + new_metrics))

        # Get max episode ID
        max_id = await self._get_max_episode_id(con)

        # Store in cache
        entry = CachedMetrics(max_id, all_metrics)
        self.cache = entry

        return entry

    async def _get_max_episode_id(self, con: AsyncConnection) -> int:
        """Get the maximum episode ID."""
        result = await con.execute(GET_MAX_EPISODE_QUERY)
        row = await result.fetchone()
        return row[0] if row and row[0] else 0

    def clear(self) -> None:
        """Clear the cache."""
        self.cache = None


def create_dashboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a dashboard router with the given StatsRepo instance."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    # Create the metrics cache
    metrics_cache = MetricsCache(metta_repo)

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.get("/suites")
    @timed_route("get_suites")
    async def get_suites() -> List[str]:  # type: ignore[reportUnusedFunction]
        return await metta_repo.get_suites()

    @router.get("/metrics")
    @timed_route("get_all_metrics")
    async def get_all_metrics() -> List[str]:  # type: ignore[reportUnusedFunction]
        """Get all distinct metrics across all suites."""
        return await metrics_cache.get()

    @router.get("/suites/{suite}/metrics")
    @timed_route("get_metrics")
    async def get_metrics(suite: str) -> List[str]:  # type: ignore[reportUnusedFunction]
        return await metta_repo.get_metrics(suite)

    @router.get("/suites/{suite}/group-ids")
    @timed_route("get_group_ids")
    async def get_group_ids(suite: str) -> List[str]:  # type: ignore[reportUnusedFunction]
        return await metta_repo.get_group_ids(suite)

    @router.get("/saved")
    @timed_route("list_saved_dashboards")
    async def list_saved_dashboards() -> SavedDashboardListResponse:  # type: ignore[reportUnusedFunction]
        """List all saved dashboards."""
        dashboards = await metta_repo.list_saved_dashboards()
        return SavedDashboardListResponse(
            dashboards=[
                SavedDashboardResponse(
                    id=dashboard["id"],
                    name=dashboard["name"],
                    description=dashboard["description"],
                    type=dashboard["type"],
                    dashboard_state=dashboard["dashboard_state"],
                    created_at=dashboard["created_at"].isoformat(),
                    updated_at=dashboard["updated_at"].isoformat(),
                    user_id=dashboard["user_id"],
                )
                for dashboard in dashboards
            ]
        )

    @router.get("/saved/{dashboard_id}")
    @timed_route("get_saved_dashboard")
    async def get_saved_dashboard(dashboard_id: str) -> SavedDashboardResponse:  # type: ignore[reportUnusedFunction]
        """Get a specific saved dashboard by ID."""
        dashboard = await metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.post("/saved")
    @timed_route("create_saved_dashboard")
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
            raise HTTPException(status_code=500, detail="Failed to create dashboard")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.put("/saved/{dashboard_id}")
    @timed_route("update_saved_dashboard")
    async def update_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str,
        dashboard_data: SavedDashboardCreate,
        user_or_token: str = user_or_token,
    ) -> SavedDashboardResponse:
        """Update an existing saved dashboard."""
        success = await metta_repo.update_saved_dashboard(
            user_id=user_or_token,
            dashboard_id=dashboard_id,
            name=dashboard_data.name,
            description=dashboard_data.description,
            dashboard_type=dashboard_data.type,
            dashboard_state=dashboard_data.dashboard_state,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        # Fetch the updated dashboard to return
        dashboard = await metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise HTTPException(status_code=500, detail="Failed to fetch updated dashboard")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.delete("/saved/{dashboard_id}")
    @timed_route("delete_saved_dashboard")
    async def delete_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str, user_or_token: str = user_or_token
    ) -> Dict[str, str]:
        """Delete a saved dashboard."""
        success = await metta_repo.delete_saved_dashboard(user_or_token, dashboard_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return {"message": "Dashboard deleted successfully"}

    @router.get("/training-runs")
    @timed_route("get_training_runs")
    async def get_training_runs() -> TrainingRunListResponse:  # type: ignore[reportUnusedFunction]
        """Get all training runs."""
        training_runs = await metta_repo.get_training_runs()
        return TrainingRunListResponse(
            training_runs=[
                TrainingRun(
                    id=run["id"],
                    name=run["name"],
                    created_at=run["created_at"],
                    user_id=run["user_id"],
                    finished_at=run["finished_at"],
                    status=run["status"],
                    url=run["url"],
                    description=run["description"],
                    tags=run["tags"],
                )
                for run in training_runs
            ]
        )

    @router.get("/training-runs/{run_id}")
    @timed_route("get_training_run")
    async def get_training_run(run_id: str) -> TrainingRun:  # type: ignore[reportUnusedFunction]
        """Get a specific training run by ID."""
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")

        return TrainingRun(
            id=training_run["id"],
            name=training_run["name"],
            created_at=training_run["created_at"],
            user_id=training_run["user_id"],
            finished_at=training_run["finished_at"],
            status=training_run["status"],
            url=training_run["url"],
            description=training_run["description"],
            tags=training_run["tags"],
        )

    @router.put("/training-runs/{run_id}/description")
    @timed_route("update_training_run_description")
    async def update_training_run_description(  # type: ignore[reportUnusedFunction]
        run_id: str,
        description_update: TrainingRunDescriptionUpdate,
        user_or_token: str = user_or_token,
    ) -> TrainingRun:
        """Update the description of a training run."""
        success = await metta_repo.update_training_run_description(
            user_id=user_or_token,
            run_id=run_id,
            description=description_update.description,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Training run not found or access denied")

        # Return the updated training run
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=500, detail="Failed to fetch updated training run")

        return TrainingRun(
            id=training_run["id"],
            name=training_run["name"],
            created_at=training_run["created_at"],
            user_id=training_run["user_id"],
            finished_at=training_run["finished_at"],
            status=training_run["status"],
            url=training_run["url"],
            description=training_run["description"],
            tags=training_run["tags"],
        )

    @router.put("/training-runs/{run_id}/tags")
    @timed_route("update_training_run_tags")
    async def update_training_run_tags(  # type: ignore[reportUnusedFunction]
        run_id: str,
        tags_update: TrainingRunTagsUpdate,
        user_or_token: str = user_or_token,
    ) -> TrainingRun:
        """Update the tags of a training run."""
        success = await metta_repo.update_training_run_tags(
            user_id=user_or_token,
            run_id=run_id,
            tags=tags_update.tags,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Training run not found or access denied")

        # Return the updated training run
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=500, detail="Failed to fetch updated training run")

        return TrainingRun(
            id=training_run["id"],
            name=training_run["name"],
            created_at=training_run["created_at"],
            user_id=training_run["user_id"],
            finished_at=training_run["finished_at"],
            status=training_run["status"],
            url=training_run["url"],
            description=training_run["description"],
            tags=training_run["tags"],
        )

    @router.post("/clear_metrics_cache")
    @timed_route("clear_metrics_cache")
    async def clear_metrics_cache() -> Dict[str, str]:  # type: ignore[reportUnusedFunction]
        """Clear the metrics cache."""
        metrics_cache.clear()
        return {"message": "Metrics cache cleared successfully"}

    return router

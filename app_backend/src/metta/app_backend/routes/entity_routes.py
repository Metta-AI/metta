from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route


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


class TrainingRunPolicy(BaseModel):
    """Training run policy with epoch information."""

    policy_name: str
    policy_id: str
    epoch_start: Optional[int]
    epoch_end: Optional[int]


async def get_training_run_policies(con: AsyncConnection, training_run_id: str) -> List[TrainingRunPolicy]:
    """Get policies for a training run with epoch information."""
    query = """
    SELECT
        p.name as policy_name,
        p.id as policy_id,
        e.start_training_epoch as epoch_start,
        e.end_training_epoch as epoch_end
    FROM policies p
    JOIN epochs e ON p.epoch_id = e.id
    WHERE e.run_id = %s
    ORDER BY e.start_training_epoch ASC, p.name ASC
    """

    rows = await execute_query_and_log(con, query, (training_run_id,), "get_training_run_policies")

    return [
        TrainingRunPolicy(policy_name=row[0], policy_id=str(row[1]), epoch_start=row[2], epoch_end=row[3])
        for row in rows
    ]


def create_entity_router(metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(tags=["entity"])

    user_or_token = Depends(dependency=create_user_or_token_dependency(metta_repo))

    @router.get("/training-runs")
    @timed_route("get_training_runs")
    async def get_training_runs(user_id: str = user_or_token) -> TrainingRunListResponse:  # type: ignore[reportUnusedFunction]
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
    async def get_training_run(run_id: str, user_id: str = user_or_token) -> TrainingRun:  # type: ignore[reportUnusedFunction]
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

    @router.get("/training-runs/{run_id}/policies")
    @timed_route("get_training_run_policies")
    async def get_training_run_policies_endpoint(run_id: str) -> List[TrainingRunPolicy]:
        """Get policies for a training run with epoch information."""
        async with metta_repo.connect() as con:
            return await get_training_run_policies(con, run_id)

    return router

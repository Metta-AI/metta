import typing

import fastapi
import psycopg
import pydantic

import metta.app_backend.auth
import metta.app_backend.metta_repo
import metta.app_backend.query_logger
import metta.app_backend.route_logger


class TrainingRunResponse(pydantic.BaseModel):
    id: str
    name: str
    created_at: str
    user_id: str
    finished_at: typing.Optional[str]
    status: str
    url: typing.Optional[str]
    description: typing.Optional[str]
    tags: typing.List[str]

    @classmethod
    def from_db(cls, training_run: metta.app_backend.metta_repo.TrainingRunRow) -> "TrainingRunResponse":
        return cls(
            id=str(training_run.id),
            name=training_run.name,
            created_at=training_run.created_at.isoformat(),
            user_id=training_run.user_id,
            finished_at=training_run.finished_at.isoformat() if training_run.finished_at else None,
            status=training_run.status,
            url=training_run.url,
            description=training_run.description,
            tags=training_run.tags,
        )


class TrainingRunListResponse(pydantic.BaseModel):
    training_runs: typing.List[TrainingRunResponse]


class TrainingRunDescriptionUpdate(pydantic.BaseModel):
    description: str


class TrainingRunTagsUpdate(pydantic.BaseModel):
    tags: typing.List[str]


class TrainingRunPolicy(pydantic.BaseModel):
    """Training run policy with epoch information."""

    policy_name: str
    policy_id: str
    epoch_start: typing.Optional[int]
    epoch_end: typing.Optional[int]


async def get_training_run_policies(
    con: psycopg.AsyncConnection, training_run_id: str
) -> typing.List[TrainingRunPolicy]:
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

    rows = await metta.app_backend.query_logger.execute_query_and_log(
        con, query, (training_run_id,), "get_training_run_policies"
    )

    return [
        TrainingRunPolicy(policy_name=row[0], policy_id=str(row[1]), epoch_start=row[2], epoch_end=row[3])
        for row in rows
    ]


def create_entity_router(metta_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    router = fastapi.APIRouter(tags=["entity"])

    user_or_token = fastapi.Depends(dependency=metta.app_backend.auth.create_user_or_token_dependency(metta_repo))

    @router.get("/training-runs")
    @metta.app_backend.route_logger.timed_route("get_training_runs")
    async def get_training_runs(user_id: str = user_or_token) -> TrainingRunListResponse:  # type: ignore[reportUnusedFunction]
        """Get all training runs."""
        training_runs = await metta_repo.get_training_runs()
        return TrainingRunListResponse(training_runs=[TrainingRunResponse.from_db(run) for run in training_runs])

    @router.get("/training-runs/{run_id}")
    @metta.app_backend.route_logger.timed_route("get_training_run")
    async def get_training_run(run_id: str, user_id: str = user_or_token) -> TrainingRunResponse:  # type: ignore[reportUnusedFunction]
        """Get a specific training run by ID."""
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise fastapi.HTTPException(status_code=404, detail="Training run not found")

        return TrainingRunResponse.from_db(training_run)

    @router.put("/training-runs/{run_id}/description")
    @metta.app_backend.route_logger.timed_route("update_training_run_description")
    async def update_training_run_description(  # type: ignore[reportUnusedFunction]
        run_id: str,
        description_update: TrainingRunDescriptionUpdate,
        user_or_token: str = user_or_token,
    ) -> TrainingRunResponse:
        """Update the description of a training run."""
        success = await metta_repo.update_training_run_description(
            user_id=user_or_token,
            run_id=run_id,
            description=description_update.description,
        )

        if not success:
            raise fastapi.HTTPException(status_code=404, detail="Training run not found or access denied")

        # Return the updated training run
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise fastapi.HTTPException(status_code=500, detail="Failed to fetch updated training run")

        return TrainingRunResponse.from_db(training_run)

    @router.put("/training-runs/{run_id}/tags")
    @metta.app_backend.route_logger.timed_route("update_training_run_tags")
    async def update_training_run_tags(  # type: ignore[reportUnusedFunction]
        run_id: str,
        tags_update: TrainingRunTagsUpdate,
        user_or_token: str = user_or_token,
    ) -> TrainingRunResponse:
        """Update the tags of a training run."""
        success = await metta_repo.update_training_run_tags(
            user_id=user_or_token,
            run_id=run_id,
            tags=tags_update.tags,
        )

        if not success:
            raise fastapi.HTTPException(status_code=404, detail="Training run not found or access denied")

        # Return the updated training run
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise fastapi.HTTPException(status_code=500, detail="Failed to fetch updated training run")

        return TrainingRunResponse.from_db(training_run)

    @router.get("/training-runs/{run_id}/policies")
    @metta.app_backend.route_logger.timed_route("get_training_run_policies")
    async def get_training_run_policies_endpoint(run_id: str) -> typing.List[TrainingRunPolicy]:
        """Get policies for a training run with epoch information."""
        async with metta_repo.connect() as con:
            return await get_training_run_policies(con, run_id)

    return router

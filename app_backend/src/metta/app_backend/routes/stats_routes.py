import typing
import uuid

import fastapi
import pydantic

import metta.app_backend.auth
import metta.app_backend.metta_repo
import metta.app_backend.route_logger


# Request/Response Models
class PolicyIdResponse(pydantic.BaseModel):
    policy_ids: dict[str, uuid.UUID]


class TrainingRunCreate(pydantic.BaseModel):
    name: str
    attributes: dict[str, str] = pydantic.Field(default_factory=dict)
    url: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class TrainingRunResponse(pydantic.BaseModel):
    id: uuid.UUID


class EpochCreate(pydantic.BaseModel):
    start_training_epoch: int
    end_training_epoch: int
    attributes: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class EpochResponse(pydantic.BaseModel):
    id: uuid.UUID


class PolicyCreate(pydantic.BaseModel):
    name: str
    description: str | None = None
    url: str | None = None
    epoch_id: uuid.UUID | None = None


class PolicyResponse(pydantic.BaseModel):
    id: uuid.UUID


class EpisodeCreate(pydantic.BaseModel):
    agent_policies: dict[int, uuid.UUID]
    # agent_id -> metric_name -> metric_value
    agent_metrics: dict[int, dict[str, float]]
    primary_policy_id: uuid.UUID
    sim_suite: str
    env_name: str
    stats_epoch: uuid.UUID | None = None
    replay_url: str | None = None
    attributes: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    eval_task_id: uuid.UUID | None = None
    tags: list[str] | None = None
    thumbnail_url: str | None = None


class EpisodeResponse(pydantic.BaseModel):
    id: uuid.UUID


def create_stats_router(stats_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = fastapi.APIRouter(prefix="/stats", tags=["stats"])

    # Create the user-or-token authentication dependency
    user_or_token = fastapi.Depends(metta.app_backend.auth.create_user_or_token_dependency(stats_repo))

    @router.get("/policies/ids", response_model=PolicyIdResponse)
    @metta.app_backend.route_logger.timed_route("get_policy_ids")
    async def get_policy_ids(
        policy_names: list[str] = fastapi.Query(default=[]), user: str = user_or_token
    ) -> PolicyIdResponse:
        """Get policy IDs for given policy names."""
        try:
            policy_ids = await stats_repo.get_policy_ids(policy_names)
            return PolicyIdResponse(policy_ids=policy_ids)
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to get policy IDs: {str(e)}") from e

    @router.post("/training-runs", response_model=TrainingRunResponse)
    @metta.app_backend.route_logger.timed_route("create_training_run")
    async def create_training_run(training_run: TrainingRunCreate, user: str = user_or_token) -> TrainingRunResponse:
        """Create a new training run."""
        try:
            run_id = await stats_repo.create_training_run(
                name=training_run.name,
                user_id=user,
                attributes=training_run.attributes,
                url=training_run.url,
                description=training_run.description,
                tags=training_run.tags,
            )
            return TrainingRunResponse(id=run_id)
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to create training run: {str(e)}") from e

    @router.patch("/training-runs/{run_id}/status", status_code=204)
    @metta.app_backend.route_logger.timed_route("update_training_run_status")
    async def update_training_run_status(run_id: str, status_update: dict[str, str], user: str = user_or_token) -> None:
        """Update the status of a training run."""
        # Validate status value first, outside try block so HTTPExceptions can bubble up
        status = status_update.get("status")
        if not status:
            raise fastapi.HTTPException(status_code=400, detail="Missing 'status' field")

        valid_statuses = {"running", "completed", "failed"}
        if status not in valid_statuses:
            raise fastapi.HTTPException(
                status_code=400, detail=f"Invalid status '{status}'. Must be one of: {', '.join(valid_statuses)}"
            )

        try:
            run_id_uuid = uuid.UUID(run_id)
            await stats_repo.update_training_run_status(run_id_uuid, status)

        except ValueError as e:
            error_msg = str(e)
            if (
                "invalid literal for int()" in error_msg
                or "badly formed hexadecimal" in error_msg
                or "UUID" in error_msg
            ):
                raise fastapi.HTTPException(status_code=400, detail="Invalid UUID format") from e
            elif "not found" in error_msg.lower():
                raise fastapi.HTTPException(status_code=404, detail=error_msg) from e
            else:
                # Let other ValueErrors bubble up to be handled appropriately
                raise fastapi.HTTPException(status_code=400, detail=f"Validation error: {error_msg}") from e
        except Exception as e:
            raise fastapi.HTTPException(
                status_code=500, detail=f"Failed to update training run status: {str(e)}"
            ) from e

    @router.post("/training-runs/{run_id}/epochs", response_model=EpochResponse)
    @metta.app_backend.route_logger.timed_route("create_epoch")
    async def create_epoch(run_id: str, epoch: EpochCreate, user: str = user_or_token) -> EpochResponse:
        """Create a new policy epoch."""
        try:
            run_id_uuid = uuid.UUID(run_id)
            epoch_id = await stats_repo.create_epoch(
                run_id=run_id_uuid,
                start_training_epoch=epoch.start_training_epoch,
                end_training_epoch=epoch.end_training_epoch,
                attributes=epoch.attributes,
            )
            return EpochResponse(id=epoch_id)
        except ValueError as e:
            raise fastapi.HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to create policy epoch: {str(e)}") from e

    @router.post("/policies", response_model=PolicyResponse)
    @metta.app_backend.route_logger.timed_route("create_policy")
    async def create_policy(policy: PolicyCreate, user: str = user_or_token) -> PolicyResponse:
        """Create a new policy."""
        try:
            policy_id = await stats_repo.create_policy(
                name=policy.name, description=policy.description, url=policy.url, epoch_id=policy.epoch_id
            )
            return PolicyResponse(id=policy_id)
        except ValueError as e:
            raise fastapi.HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}") from e

    @router.post("/episodes", response_model=EpisodeResponse)
    @metta.app_backend.route_logger.timed_route("record_episode")
    async def record_episode(episode: EpisodeCreate, user: str = user_or_token) -> EpisodeResponse:
        """Record a new episode with agent policies and metrics."""
        eval_name = f"{episode.sim_suite}/{episode.env_name}"
        try:
            episode_id = await stats_repo.record_episode(
                agent_policies=episode.agent_policies,
                agent_metrics=episode.agent_metrics,
                primary_policy_id=episode.primary_policy_id,
                eval_name=eval_name,
                stats_epoch=episode.stats_epoch,
                replay_url=episode.replay_url,
                attributes=episode.attributes,
                eval_task_id=episode.eval_task_id,
                tags=episode.tags,
                thumbnail_url=episode.thumbnail_url,
            )
            return EpisodeResponse(id=episode_id)
        except ValueError as e:
            raise fastapi.HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to record episode: {str(e)}") from e

    return router

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.route_logger import timed_route


# Request/Response Models
class PolicyIdResponse(BaseModel):
    policy_ids: dict[str, uuid.UUID]


class TrainingRunCreate(BaseModel):
    name: str
    attributes: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class TrainingRunResponse(BaseModel):
    id: uuid.UUID


class EpochCreate(BaseModel):
    start_training_epoch: int
    end_training_epoch: int
    attributes: dict[str, str] = Field(default_factory=dict)


class EpochResponse(BaseModel):
    id: uuid.UUID


class PolicyCreate(BaseModel):
    name: str
    description: str | None = None
    url: str | None = None
    epoch_id: uuid.UUID | None = None


class PolicyResponse(BaseModel):
    id: uuid.UUID


class EpisodeCreate(BaseModel):
    agent_policies: dict[int, uuid.UUID]
    # agent_id -> metric_name -> metric_value
    agent_metrics: dict[int, dict[str, float]]
    primary_policy_id: uuid.UUID
    stats_epoch: uuid.UUID | None = None
    eval_name: str | None = None
    simulation_suite: str | None = None
    replay_url: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    eval_task_id: uuid.UUID | None = None
    tags: list[str] | None = None


class EpisodeResponse(BaseModel):
    id: uuid.UUID


def create_stats_router(stats_repo: MettaRepo) -> APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = APIRouter(prefix="/stats", tags=["stats"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.get("/policies/ids", response_model=PolicyIdResponse)
    @timed_route("get_policy_ids")
    async def get_policy_ids(
        policy_names: list[str] = Query(default=[]), user: str = user_or_token
    ) -> PolicyIdResponse:
        """Get policy IDs for given policy names."""
        try:
            policy_ids = await stats_repo.get_policy_ids(policy_names)
            return PolicyIdResponse(policy_ids=policy_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get policy IDs: {str(e)}") from e

    @router.post("/training-runs", response_model=TrainingRunResponse)
    @timed_route("create_training_run")
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
            raise HTTPException(status_code=500, detail=f"Failed to create training run: {str(e)}") from e

    @router.post("/training-runs/{run_id}/epochs", response_model=EpochResponse)
    @timed_route("create_epoch")
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
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy epoch: {str(e)}") from e

    @router.post("/policies", response_model=PolicyResponse)
    @timed_route("create_policy")
    async def create_policy(policy: PolicyCreate, user: str = user_or_token) -> PolicyResponse:
        """Create a new policy."""
        try:
            policy_id = await stats_repo.create_policy(
                name=policy.name, description=policy.description, url=policy.url, epoch_id=policy.epoch_id
            )
            return PolicyResponse(id=policy_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}") from e

    @router.post("/episodes", response_model=EpisodeResponse)
    @timed_route("record_episode")
    async def record_episode(episode: EpisodeCreate, user: str = user_or_token) -> EpisodeResponse:
        """Record a new episode with agent policies and metrics."""
        try:
            episode_id = await stats_repo.record_episode(
                agent_policies=episode.agent_policies,
                agent_metrics=episode.agent_metrics,
                primary_policy_id=episode.primary_policy_id,
                stats_epoch=episode.stats_epoch,
                eval_name=episode.eval_name,
                simulation_suite=episode.simulation_suite,
                replay_url=episode.replay_url,
                attributes=episode.attributes,
                eval_task_id=episode.eval_task_id,
                tags=episode.tags,
            )
            return EpisodeResponse(id=episode_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to record episode: {str(e)}") from e

    return router

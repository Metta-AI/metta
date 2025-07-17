import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.route_logger import timed_route


# Request/Response Models
class PolicyIdResponse(BaseModel):
    policy_ids: Dict[str, str]


class TrainingRunCreate(BaseModel):
    name: str
    attributes: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class TrainingRunResponse(BaseModel):
    id: str


class EpochCreate(BaseModel):
    start_training_epoch: int
    end_training_epoch: int
    attributes: Dict[str, str] = Field(default_factory=dict)


class EpochResponse(BaseModel):
    id: str


class PolicyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    epoch_id: Optional[str] = None


class PolicyResponse(BaseModel):
    id: str


class EpisodeCreate(BaseModel):
    agent_policies: Dict[int, str]
    agent_metrics: Dict[int, Dict[str, float]]
    primary_policy_id: str
    stats_epoch: Optional[str] = None
    eval_name: Optional[str] = None
    simulation_suite: Optional[str] = None
    replay_url: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    eval_task_id: Optional[str] = None


class EpisodeResponse(BaseModel):
    id: str


def create_stats_router(stats_repo: MettaRepo) -> APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = APIRouter(prefix="/stats", tags=["stats"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.get("/policies/ids", response_model=PolicyIdResponse)
    @timed_route("get_policy_ids")
    async def get_policy_ids(
        policy_names: List[str] = Query(default=[]), user: str = user_or_token
    ) -> PolicyIdResponse:
        """Get policy IDs for given policy names."""
        try:
            policy_ids = await stats_repo.get_policy_ids(policy_names)
            # Convert UUIDs to strings for JSON serialization
            policy_ids_str = {name: str(uuid_val) for name, uuid_val in policy_ids.items()}
            return PolicyIdResponse(policy_ids=policy_ids_str)
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
            return TrainingRunResponse(id=str(run_id))
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
            return EpochResponse(id=str(epoch_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy epoch: {str(e)}") from e

    @router.post("/policies", response_model=PolicyResponse)
    @timed_route("create_policy")
    async def create_policy(policy: PolicyCreate, user: str = user_or_token) -> PolicyResponse:
        """Create a new policy."""
        try:
            epoch_id_uuid = uuid.UUID(policy.epoch_id) if policy.epoch_id else None
            policy_id = await stats_repo.create_policy(
                name=policy.name, description=policy.description, url=policy.url, epoch_id=epoch_id_uuid
            )
            return PolicyResponse(id=str(policy_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}") from e

    @router.post("/episodes", response_model=EpisodeResponse)
    @timed_route("record_episode")
    async def record_episode(episode: EpisodeCreate, user: str = user_or_token) -> EpisodeResponse:
        """Record a new episode with agent policies and metrics."""
        try:
            # Convert string UUIDs to UUID objects
            agent_policies_uuid = {
                agent_id: uuid.UUID(policy_id) for agent_id, policy_id in episode.agent_policies.items()
            }
            primary_policy_id_uuid = uuid.UUID(episode.primary_policy_id)
            stats_epoch_uuid = uuid.UUID(episode.stats_epoch) if episode.stats_epoch else None
            eval_task_id_uuid = uuid.UUID(episode.eval_task_id) if episode.eval_task_id else None

            episode_id = await stats_repo.record_episode(
                agent_policies=agent_policies_uuid,
                agent_metrics=episode.agent_metrics,
                primary_policy_id=primary_policy_id_uuid,
                stats_epoch=stats_epoch_uuid,
                eval_name=episode.eval_name,
                simulation_suite=episode.simulation_suite,
                replay_url=episode.replay_url,
                attributes=episode.attributes,
                eval_task_id=eval_task_id_uuid,
            )
            return EpisodeResponse(id=str(episode_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to record episode: {str(e)}") from e

    return router

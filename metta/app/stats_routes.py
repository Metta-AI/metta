from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app.stats_repo import StatsRepo


# Request/Response Models
class PolicyIdResponse(BaseModel):
    policy_ids: Dict[str, int]


class TrainingRunCreate(BaseModel):
    name: str
    user_id: str
    attributes: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None


class TrainingRunResponse(BaseModel):
    id: int


class EpochCreate(BaseModel):
    start_training_epoch: int
    end_training_epoch: int
    attributes: Dict[str, str] = Field(default_factory=dict)


class PolicyEpochResponse(BaseModel):
    id: int


class PolicyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    epoch_id: Optional[int] = None


class PolicyResponse(BaseModel):
    id: int


class EpisodeCreate(BaseModel):
    agent_policies: Dict[int, int]
    agent_metrics: Dict[int, Dict[str, float]]
    primary_policy_id: int
    training_epoch: Optional[int] = None
    eval_name: Optional[str] = None
    simulation_suite: Optional[str] = None
    replay_url: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EpisodeResponse(BaseModel):
    id: int


def create_stats_router(stats_repo: StatsRepo) -> APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = APIRouter(prefix="/stats", tags=["stats"])

    @router.get("/policies/ids", response_model=PolicyIdResponse)
    async def get_policy_ids(policy_names: List[str] = Query(default=[])) -> PolicyIdResponse:
        """Get policy IDs for given policy names."""
        try:
            policy_ids = stats_repo.get_policy_ids(policy_names)
            return PolicyIdResponse(policy_ids=policy_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get policy IDs: {str(e)}") from e

    @router.post("/training-runs", response_model=TrainingRunResponse)
    async def create_training_run(training_run: TrainingRunCreate) -> TrainingRunResponse:
        """Create a new training run."""
        try:
            run_id = stats_repo.create_training_run(
                name=training_run.name,
                user_id=training_run.user_id,
                attributes=training_run.attributes,
                url=training_run.url,
            )
            return TrainingRunResponse(id=run_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create training run: {str(e)}") from e

    @router.post("/training-runs/{run_id}/epochs", response_model=PolicyEpochResponse)
    async def create_epoch(run_id: int, epoch: EpochCreate) -> PolicyEpochResponse:
        """Create a new policy epoch."""
        try:
            epoch_id = stats_repo.create_epoch(
                run_id=run_id,
                start_training_epoch=epoch.start_training_epoch,
                end_training_epoch=epoch.end_training_epoch,
                attributes=epoch.attributes,
            )
            return PolicyEpochResponse(id=epoch_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy epoch: {str(e)}") from e

    @router.post("/policies", response_model=PolicyResponse)
    async def create_policy(policy: PolicyCreate) -> PolicyResponse:
        """Create a new policy."""
        try:
            policy_id = stats_repo.create_policy(
                name=policy.name, description=policy.description, url=policy.url, epoch_id=policy.epoch_id
            )
            return PolicyResponse(id=policy_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}") from e

    @router.post("/episodes", response_model=EpisodeResponse)
    async def record_episode(episode: EpisodeCreate) -> EpisodeResponse:
        """Record a new episode with agent policies and metrics."""
        try:
            episode_id = stats_repo.record_episode(
                agent_policies=episode.agent_policies,
                agent_metrics=episode.agent_metrics,
                primary_policy_id=episode.primary_policy_id,
                training_epoch=episode.training_epoch,
                eval_name=episode.eval_name,
                simulation_suite=episode.simulation_suite,
                replay_url=episode.replay_url,
                attributes=episode.attributes,
            )
            return EpisodeResponse(id=episode_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to record episode: {str(e)}") from e

    return router

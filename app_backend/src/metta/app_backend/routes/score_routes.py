import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.stats_repo import StatsRepo

logger = logging.getLogger("score_routes")


class MetricStats(BaseModel):
    min: float
    max: float
    avg: float


class PolicyScoresRequest(BaseModel):
    policy_ids: list[uuid.UUID]
    eval_names: list[str]
    metrics: list[str]


class PolicyScoresData(BaseModel):
    scores: dict[uuid.UUID, dict[str, dict[str, MetricStats]]]


# Note: POLICY_METRIC_STATS_QUERY was removed - now using StatsRepo.get_policy_metric_stats()


async def fetch_policy_scores(
    stats_repo: StatsRepo,
    policy_ids: list[uuid.UUID],
    eval_names: list[str],
    metrics: list[str],
) -> dict[uuid.UUID, dict[str, dict[str, MetricStats]]]:
    # Convert UUIDs to strings for ClickHouse
    policy_ids_str = [str(policy_id) for policy_id in policy_ids]
    rows = await stats_repo.get_policy_metric_stats(policy_ids_str, eval_names, metrics)

    result: dict[uuid.UUID, dict[str, dict[str, MetricStats]]] = {}
    for policy_id, eval_name, metric, min_value, max_value, avg_value in rows:
        if policy_id not in result:
            result[policy_id] = {}
        if eval_name not in result[policy_id]:
            result[policy_id][eval_name] = {}
        result[policy_id][eval_name][metric] = MetricStats(
            min=float(min_value) if min_value is not None else 0.0,
            max=float(max_value) if max_value is not None else 0.0,
            avg=float(avg_value) if avg_value is not None else 0.0,
        )

    return result


def create_score_router(stats_repo: StatsRepo, metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(tags=["score"], prefix="/scorecard")

    @router.post("/score")
    async def get_policy_scores(request: PolicyScoresRequest) -> PolicyScoresData:
        if not request.policy_ids or not request.eval_names or not request.metrics:
            raise HTTPException(status_code=400, detail="Missing required parameters")

        scores = await fetch_policy_scores(stats_repo, request.policy_ids, request.eval_names, request.metrics)
        return PolicyScoresData(scores=scores)

    return router

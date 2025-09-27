import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from softmax.orchestrator.metta_repo import MettaRepo
from softmax.orchestrator.query_logger import execute_query_and_log

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


POLICY_METRIC_STATS_QUERY = """
    SELECT
        we.primary_policy_id::text as policy_id,
        we.eval_name as eval_name,
        eam.metric as metric,
        MIN(eam.value) as min_value,
        MAX(eam.value) as max_value,
        AVG(eam.value) as avg_value
    FROM episode_agent_metrics eam
    JOIN wide_episodes we ON we.internal_id = eam.episode_internal_id
    WHERE
        we.primary_policy_id = ANY(%s)
        AND we.eval_name = ANY(%s)
        AND eam.metric = ANY(%s)
    GROUP BY we.primary_policy_id, we.eval_name, eam.metric
"""


async def fetch_policy_scores(
    con: Any,
    policy_ids: list[uuid.UUID],
    eval_names: list[str],
    metrics: list[str],
) -> dict[uuid.UUID, dict[str, dict[str, MetricStats]]]:
    rows = await execute_query_and_log(
        con,
        POLICY_METRIC_STATS_QUERY,
        (policy_ids, eval_names, metrics),
        "get_policy_metric_stats",
    )

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


def create_score_router(metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(tags=["score"], prefix="/scorecard")

    @router.post("/score")
    async def get_policy_scores(request: PolicyScoresRequest) -> PolicyScoresData:
        if not request.policy_ids or not request.eval_names or not request.metrics:
            raise HTTPException(status_code=400, detail="Missing required parameters")

        async with metta_repo.connect() as con:
            scores = await fetch_policy_scores(con, request.policy_ids, request.eval_names, request.metrics)
        return PolicyScoresData(scores=scores)

    return router

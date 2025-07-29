import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, LiteralString, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from psycopg import AsyncConnection
from psycopg.rows import class_row
from pydantic import BaseModel

from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route

logger = logging.getLogger("heatmap_routes")

# ============================================================================
# Data Models
# ============================================================================


class HeatmapCell(BaseModel):
    """Single cell in the heatmap grid."""

    evalName: str
    replayUrl: Optional[str]
    value: float


class HeatmapData(BaseModel):
    """Complete heatmap data structure."""

    evalNames: List[str]
    cells: Dict[str, Dict[str, HeatmapCell]]
    policyAverageScores: Dict[str, float]
    evalAverageScores: Dict[str, float]
    evalMaxScores: Dict[str, float]


@dataclass
class PolicyEvaluation:
    """Represents a single policy evaluation result."""

    policy_uri: str
    eval_name: str
    replay_url: Optional[str]
    total_score: float  # Total score across all agents/episodes
    num_agents: int  # Total number of agents across all episodes
    episode_id: int  # The episode ID of the latest episode for this policy/eval pair

    run_id: Optional[str] = None
    epoch: Optional[int] = None

    @property
    def value(self) -> float:
        """Average score per agent."""
        return self.total_score / self.num_agents if self.num_agents > 0 else 0.0


# ============================================================================
# SQL Queries
# ============================================================================

# Base query for fetching evaluation data with policy filtering
EVALUATION_DATA_QUERY_TEMPLATE = """
    WITH
    {policy_filter},
    pre_aggregated AS (
        SELECT
            episode_internal_id,
            SUM(value) as total_value,
            COUNT(*) as agent_count
        FROM episode_agent_metrics
        WHERE metric = %s
        AND episode_internal_id >= %s
        GROUP BY episode_internal_id
    )
    SELECT
        p.name as policy_uri,
        e.env_name as eval_name,
        ANY_VALUE(e.replay_url) as replay_url,
        SUM(pa.total_value) as total_score,
        SUM(pa.agent_count)::INTEGER as num_agents,
        MAX(e.internal_id) as episode_id,
        p.run_id,
        p.end_training_epoch as epoch
    FROM episodes e
    JOIN pre_aggregated pa ON e.internal_id = pa.episode_internal_id
    JOIN filtered_policies p ON e.primary_policy_id = p.id
    WHERE e.eval_category = %s
    GROUP BY p.name, e.env_name, p.run_id, p.end_training_epoch
    ORDER BY p.run_id, p.end_training_epoch DESC
"""

# Policy filters for different contexts
ALL_POLICIES_FILTER = """
    filtered_policies AS (
        SELECT p.id, p.name, ep.run_id, ep.end_training_epoch
        FROM policies p
        LEFT JOIN epochs ep ON p.epoch_id = ep.id
    )
"""

TRAINING_RUN_FILTER = """
    filtered_policies AS (
        SELECT p.id, p.name, ep.run_id, ep.end_training_epoch
        FROM policies p
        JOIN epochs ep ON p.epoch_id = ep.id
        WHERE ep.run_id = %s
    )
"""

SPECIFIC_POLICIES_FILTER = """
    filtered_policies AS (
        SELECT p.id, p.name, ep.run_id, ep.end_training_epoch
        FROM (
            SELECT DISTINCT e.primary_policy_id
            FROM episodes e
            WHERE e.eval_category = %s
            AND e.internal_id > %s
        ) e
        JOIN policies p ON e.primary_policy_id = p.id
        LEFT JOIN epochs ep ON p.epoch_id = ep.id
    )
"""

GET_EVAL_NAMES_QUERY = "SELECT DISTINCT env_name FROM episodes WHERE eval_category = %s"
GET_MAX_EPISODE_QUERY = "SELECT MAX(internal_id) FROM episodes WHERE eval_category = %s"
HAS_NEW_DATA_QUERY = "SELECT 1 FROM episodes WHERE eval_category = %s AND internal_id > %s LIMIT 1"


# ============================================================================
# Core Functions
# ============================================================================


async def fetch_evaluation_data(
    con: AsyncConnection,
    suite: str,
    metric: str,
    min_episode_id: int = 0,
    policy_filter: LiteralString = ALL_POLICIES_FILTER,
    filter_params: Tuple = (),
) -> List[PolicyEvaluation]:
    """Fetch evaluation data from database."""
    query = EVALUATION_DATA_QUERY_TEMPLATE.format(policy_filter=policy_filter)
    params = filter_params + (metric, min_episode_id, suite)

    async with con.cursor(row_factory=class_row(PolicyEvaluation)) as cursor:
        await cursor.execute(query, params)
        return await cursor.fetchall()


async def get_evaluation_names(con: AsyncConnection, suite: str) -> List[str]:
    """Get all evaluation names for a suite."""
    rows = await execute_query_and_log(con, GET_EVAL_NAMES_QUERY, (suite,), "get_evaluation_names")
    return [row[0] for row in rows if row[0] is not None]


def group_by_run(
    evaluations: List[PolicyEvaluation],
) -> Tuple[Dict[str, List[PolicyEvaluation]], List[PolicyEvaluation]]:
    """Group evaluations by run_id, separating those without runs."""
    by_run = defaultdict(list)
    no_run = []

    for eval in evaluations:
        if eval.run_id:
            by_run[eval.run_id].append(eval)
        else:
            no_run.append(eval)

    return dict(by_run), no_run


def select_latest_per_run(evaluations: List[PolicyEvaluation]) -> List[PolicyEvaluation]:
    """Select evaluations for the latest policy in each training run."""
    by_run, no_run = group_by_run(evaluations)
    selected = []

    for _run_id, run_evals in by_run.items():
        # Find the latest epoch in this run
        latest_epoch = max(e.epoch or 0 for e in run_evals)
        latest_policies = {e.policy_uri for e in run_evals if (e.epoch or 0) == latest_epoch}

        # If multiple policies have the same epoch, pick the first one
        if latest_policies:
            latest_policy = sorted(latest_policies)[0]
            selected.extend(e for e in run_evals if e.policy_uri == latest_policy)

    return selected + no_run


def select_best_per_run(evaluations: List[PolicyEvaluation], all_eval_names: Set[str]) -> List[PolicyEvaluation]:
    """Select evaluations for the best performing policy in each training run."""
    by_run, no_run = group_by_run(evaluations)
    selected = []

    for _run_id, run_evals in by_run.items():
        # Group by policy and calculate average scores
        policy_scores = defaultdict(lambda: defaultdict(float))
        policy_epochs = {}

        for eval in run_evals:
            policy_scores[eval.policy_uri][eval.eval_name] = eval.value
            policy_epochs[eval.policy_uri] = eval.epoch or 0

        # Calculate average score for each policy across all evaluations
        best_policy = None
        best_score = float("-inf")
        best_epoch = -1

        for policy_uri, eval_scores in policy_scores.items():
            # Average across all evaluations (missing ones count as 0)
            total_score = sum(eval_scores.get(name, 0.0) for name in all_eval_names)
            avg_score = total_score / len(all_eval_names) if all_eval_names else 0.0
            epoch = policy_epochs[policy_uri]

            # Select best, with ties broken by latest epoch
            if avg_score > best_score or (avg_score == best_score and epoch > best_epoch):
                best_policy = policy_uri
                best_score = avg_score
                best_epoch = epoch

        if best_policy:
            selected.extend(e for e in run_evals if e.policy_uri == best_policy)

    return selected + no_run


def build_heatmap(evaluations: List[PolicyEvaluation], all_eval_names: List[str]) -> HeatmapData:
    """Build heatmap data structure from evaluations."""
    # Group evaluations by policy and eval
    data_map = {(e.policy_uri, e.eval_name): e for e in evaluations}
    policy_uris = sorted({e.policy_uri for e in evaluations})

    # Build cells
    cells = {}
    for policy_uri in policy_uris:
        cells[policy_uri] = {}
        for eval_name in all_eval_names:
            eval = data_map.get((policy_uri, eval_name))
            cells[policy_uri][eval_name] = HeatmapCell(
                evalName=eval_name, replayUrl=eval.replay_url if eval else None, value=eval.value if eval else 0.0
            )

    # Calculate averages
    policy_averages = {}
    for policy_uri in policy_uris:
        scores = [cells[policy_uri][name].value for name in all_eval_names]
        policy_averages[policy_uri] = sum(scores) / len(scores) if scores else 0.0

    eval_averages = {}
    eval_max_scores = {}
    for eval_name in all_eval_names:
        scores = [cells[p][eval_name].value for p in policy_uris if p in cells]
        eval_averages[eval_name] = sum(scores) / len(scores) if scores else 0.0
        eval_max_scores[eval_name] = max(scores) if scores else 0.0

    return HeatmapData(
        evalNames=all_eval_names,
        cells=cells,
        policyAverageScores=policy_averages,
        evalAverageScores=eval_averages,
        evalMaxScores=eval_max_scores,
    )


# ============================================================================
# API Routes
# ============================================================================


def create_heatmap_router(metta_repo: MettaRepo) -> APIRouter:
    """Create FastAPI router for heatmap endpoints."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    @router.get("/training-runs/{run_id}/suites/{suite}/metrics/{metric}/heatmap")
    @timed_route("get_training_run_heatmap_data")
    async def get_training_run_heatmap_data(run_id: str, suite: str, metric: str) -> HeatmapData:
        """Get heatmap data for a specific training run."""
        # Verify training run exists
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")

        async with metta_repo.connect() as con:
            # Fetch data for this specific run
            evaluations = await fetch_evaluation_data(
                con, suite, metric, policy_filter=TRAINING_RUN_FILTER, filter_params=(run_id,)
            )

            # Build heatmap
            eval_names = await get_evaluation_names(con, suite)
            return build_heatmap(evaluations, eval_names)

    return router

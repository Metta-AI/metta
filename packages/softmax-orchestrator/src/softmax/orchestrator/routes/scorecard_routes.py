import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from psycopg import AsyncConnection
from psycopg.rows import class_row
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from softmax.orchestrator.metta_repo import MettaRepo
from softmax.orchestrator.query_logger import execute_query_and_log
from softmax.orchestrator.route_logger import timed_route

logger = logging.getLogger("policy_scorecard_routes")

# ============================================================================
# Data Models
# ============================================================================


class PaginationRequest(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)


class PoliciesSearchRequest(BaseModel):
    """Search parameters for policies."""

    search: Optional[str] = Field(default=None, description="Search term for policy names")
    policy_type: Optional[str] = Field(default=None, description="Filter by policy type: 'training_run' or 'policy'")
    tags: Optional[List[str]] = Field(
        default=None, description="Filter by tags (policies must have at least one matching tag)"
    )
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


class UnifiedPolicyInfo(BaseModel):
    """Unified policy/training run information."""

    id: str
    type: str  # 'training_run' or 'policy'
    name: str
    user_id: Optional[str]
    created_at: str
    tags: List[str]


class PoliciesResponse(BaseModel):
    """Response containing unified policies and training runs."""

    policies: List[UnifiedPolicyInfo]


class EvalsRequest(BaseModel):
    """Request body for getting eval categories based on selected training runs and policies."""

    training_run_ids: List[str]
    run_free_policy_ids: List[str]


class MetricsRequest(BaseModel):
    """Request body for getting available metrics."""

    training_run_ids: List[str]
    run_free_policy_ids: List[str]
    eval_names: List[str]


class ScorecardRequest(BaseModel):
    """Request body for generating policy-based scorecard."""

    training_run_ids: List[str]
    run_free_policy_ids: List[str]
    eval_names: List[str]
    training_run_policy_selector: Literal["best", "latest"] = Field(default="latest")
    metric: str


class TrainingRunScorecardRequest(BaseModel):
    """Request body for generating training run scorecard with ALL policies."""

    eval_names: List[str]
    metric: str


class ScorecardCell(BaseModel):
    """Single cell in the policy scorecard grid."""

    model_config = ConfigDict(populate_by_name=True)

    evalName: str
    replayUrl: Optional[str]
    thumbnailUrl: Optional[str] = Field(default=None)
    value: float


class ScorecardData(BaseModel):
    """Complete policy scorecard data structure."""

    evalNames: List[str]
    policyNames: List[str]
    cells: Dict[str, Dict[str, ScorecardCell]]
    policyAverageScores: Dict[str, float]
    evalAverageScores: Dict[str, float]
    evalMaxScores: Dict[str, float]


class LeaderboardScorecardRequest(BaseModel):
    """Request body for generating leaderboard scorecard."""

    leaderboard_id: uuid.UUID
    selector: Literal["latest", "best"]
    num_policies: int


@dataclass
class PolicyEvaluationResult:
    """Represents a single policy evaluation result for the new system."""

    policy_id: uuid.UUID
    policy_name: str
    eval_category: str
    env_name: str
    replay_url: Optional[str]
    thumbnail_url: Optional[str]
    total_score: float
    num_agents: int
    episode_id: int
    run_id: Optional[uuid.UUID] = None
    epoch: Optional[int] = None

    @property
    def value(self) -> float:
        """Average score per agent."""
        return self.total_score / self.num_agents if self.num_agents > 0 else 0.0

    @property
    def eval_name(self) -> str:
        """Combined evaluation name for display."""
        return f"{self.eval_category}/{self.env_name}"

    @property
    def unified_id(self) -> uuid.UUID:
        """Unified ID."""
        if self.run_id:
            return self.run_id
        return self.policy_id


# ============================================================================
# SQL Queries
# ============================================================================

UNIFIED_POLICIES_QUERY = """SELECT * FROM unified_training_runs ORDER BY created_at DESC"""

GET_EVALS_QUERY = """
    SELECT DISTINCT eval_name
    FROM wide_episodes
    WHERE (
        training_run_id = ANY(%s) OR  -- Policies from selected training runs
        (training_run_id IS NULL AND primary_policy_id = ANY(%s))  -- Selected run-free policies
    )
"""

GET_AVAILABLE_METRICS_QUERY = """
    SELECT DISTINCT eam.metric
    FROM episode_agent_metrics eam
    JOIN wide_episodes we ON we.internal_id = eam.episode_internal_id
    WHERE (
        we.training_run_id = ANY(%s) OR  -- Policies from selected training runs
        (we.training_run_id IS NULL AND we.primary_policy_id = ANY(%s))  -- Selected run-free policies
    )
    AND we.eval_name = ANY(%s)
    ORDER BY eam.metric
"""

GET_POLICY_NAMES_BY_IDS_QUERY = """
    SELECT p.name
    FROM policies p
    WHERE p.id = ANY(%s)
    ORDER BY p.name
"""

POLICY_SCORECARD_DATA_QUERY = """
    SELECT
        we.primary_policy_id as policy_id,
        we.policy_name,
        we.eval_category,
        we.env_name,
        ANY_VALUE(we.replay_url) as replay_url,
        ANY_VALUE(we.thumbnail_url) as thumbnail_url,
        SUM(eam.value) as total_score,
        COUNT(eam.*) as num_agents,
        MAX(we.internal_id) as episode_id,
        we.training_run_id::text as run_id,
        we.epoch_end_training_epoch as epoch
    FROM wide_episodes we
    JOIN episode_agent_metrics eam ON we.internal_id = eam.episode_internal_id
    WHERE (
        we.training_run_id = ANY(%s) OR  -- Policies from selected training runs
        (we.primary_policy_id = ANY(%s))  -- Selected policies
    )
    AND eam.metric = %s
    AND we.eval_name = ANY(%s)
    GROUP BY we.primary_policy_id, we.policy_name, we.eval_category, we.env_name, we.training_run_id,
        we.epoch_end_training_epoch
    ORDER BY we.policy_name, we.eval_category, we.env_name
"""


# ============================================================================
# Core Functions
# ============================================================================


async def get_policies_and_training_runs(con: AsyncConnection) -> PoliciesResponse:
    """Get unified training runs and run-free policies with pagination and optional filtering."""

    unified_rows = await execute_query_and_log(con, UNIFIED_POLICIES_QUERY, (), "get_unified_policies")

    policies = [
        UnifiedPolicyInfo(
            id=str(row[0]), type=row[1], name=row[2], user_id=row[3], created_at=str(row[4]), tags=row[5] or []
        )
        for row in unified_rows
    ]

    return PoliciesResponse(policies=policies)


async def search_policies_and_training_runs(
    con: AsyncConnection, search_params: PoliciesSearchRequest
) -> PoliciesResponse:
    """Search unified training runs and run-free policies with filtering."""

    # Build dynamic query based on search parameters
    base_query = "SELECT * FROM unified_training_runs"
    conditions = []
    params = []

    # Add search condition for name
    if search_params.search:
        conditions.append("LOWER(name) LIKE LOWER(%s)")
        params.append(f"%{search_params.search}%")

    # Add policy type filter
    if search_params.policy_type:
        conditions.append("type = %s")
        params.append(search_params.policy_type)

    # Add user_id filter
    if search_params.user_id:
        conditions.append("user_id = %s")
        params.append(search_params.user_id)

    # Add tag filter using array overlap
    if search_params.tags:
        conditions.append("tags && %s")
        params.append(search_params.tags)

    # Combine conditions
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    # Add ordering and pagination
    base_query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([search_params.limit, search_params.offset])

    unified_rows = await execute_query_and_log(con, base_query, tuple(params), "search_policies")

    policies = [
        UnifiedPolicyInfo(
            id=str(row[0]), type=row[1], name=row[2], user_id=row[3], created_at=str(row[4]), tags=row[5] or []
        )
        for row in unified_rows
    ]

    return PoliciesResponse(policies=policies)


async def get_evals_for_selection(
    con: AsyncConnection, training_run_ids: List[str], run_free_policy_ids: List[str]
) -> List[str]:
    """Get evaluation categories and environment names for selected training runs and policies."""
    rows = await execute_query_and_log(
        con, GET_EVALS_QUERY, (training_run_ids, run_free_policy_ids), "get_evals_for_selection"
    )

    return [row[0] for row in rows]


async def get_available_metrics_for_selection(
    con: AsyncConnection,
    training_run_ids: List[str],
    run_free_policy_ids: List[str],
    eval_names: List[str],
) -> List[str]:
    """Get available metrics for the selected training runs, policies and evaluations."""
    rows = await execute_query_and_log(
        con,
        GET_AVAILABLE_METRICS_QUERY,
        (training_run_ids, run_free_policy_ids, eval_names),
        "get_available_metrics_for_selection",
    )

    return [row[0] for row in rows]


async def fetch_policy_scorecard_data(
    con: AsyncConnection,
    training_run_ids: List[str] | List[uuid.UUID],
    run_free_policy_ids: List[str] | List[uuid.UUID],
    eval_names: List[str],
    metric: str,
) -> List[PolicyEvaluationResult]:
    """Fetch evaluation data for policy-based scorecard."""
    async with con.cursor(row_factory=class_row(PolicyEvaluationResult)) as cursor:
        await cursor.execute(POLICY_SCORECARD_DATA_QUERY, (training_run_ids, run_free_policy_ids, metric, eval_names))
        return await cursor.fetchall()


def select_policies_by_training_run_selector(
    evaluations: List[PolicyEvaluationResult], selector: str, all_eval_names: List[str]
) -> List[PolicyEvaluationResult]:
    """Select evaluations based on training run policy selector (latest/best).

    Returns:
        tuple: (selected_evaluations, all_selected_policy_names)
        - selected_evaluations: evaluations for selected policies that have data
        - all_selected_policy_names: names of all policies that should be included
    """
    return _select_policies_per_run(evaluations, selector, all_eval_names)


def _select_policies_per_run(
    evaluations: List[PolicyEvaluationResult], selector: str, all_eval_names: List[str]
) -> List[PolicyEvaluationResult]:
    """Generic function to select policies per training run based on selector strategy."""
    # Group by run_id
    by_run = {}
    no_run = []

    for eval in evaluations:
        if eval.run_id:
            if eval.run_id not in by_run:
                by_run[eval.run_id] = []
            by_run[eval.run_id].append(eval)
        else:
            no_run.append(eval)

    selected = []

    # For each training run, select policy based on selector strategy
    for _, run_evals in by_run.items():
        if selector == "latest":
            best_policy = _select_latest_policy_from_run(run_evals)
        else:  # best
            best_policy = _select_best_policy_from_run(run_evals, all_eval_names)

        if best_policy:
            selected.extend(e for e in run_evals if e.policy_name == best_policy)

    # Add run-free policies
    selected.extend(no_run)

    return selected


def _select_latest_policy_from_run(run_evals: List[PolicyEvaluationResult]) -> Optional[str]:
    """Select the latest policy from a training run based on epoch."""
    latest_epoch = max(e.epoch or 0 for e in run_evals)
    latest_policies = {e.policy_name for e in run_evals if (e.epoch or 0) == latest_epoch}

    if latest_policies:
        # If multiple policies have same epoch, pick first alphabetically
        return sorted(latest_policies)[0]
    return None


def _select_best_policy_from_run(run_evals: List[PolicyEvaluationResult], all_eval_names: List[str]) -> Optional[str]:
    """Select the best performing policy from a training run based on average score."""
    # Group by policy and calculate average scores
    policy_scores = {}
    policy_epochs = {}

    for eval in run_evals:
        if eval.policy_name not in policy_scores:
            policy_scores[eval.policy_name] = {}
        policy_scores[eval.policy_name][eval.eval_name] = eval.value
        policy_epochs[eval.policy_name] = eval.epoch or 0

    # Calculate average score for each policy across all evaluations
    best_policy = None
    best_score = float("-inf")
    best_epoch = -1

    for policy_name, eval_scores in policy_scores.items():
        # Average across all evaluations (missing ones count as 0)
        total_score = sum(eval_scores.get(name, 0.0) for name in all_eval_names)
        avg_score = total_score / len(all_eval_names) if all_eval_names else 0.0
        epoch = policy_epochs[policy_name]

        # Select best, with ties broken by latest epoch
        if avg_score > best_score or (avg_score == best_score and epoch > best_epoch):
            best_policy = policy_name
            best_score = avg_score
            best_epoch = epoch

    return best_policy


def build_policy_scorecard(
    evaluations: List[PolicyEvaluationResult],
    eval_names: List[str],
) -> ScorecardData:
    """Build scorecard data structure from policy evaluations."""

    # Group evaluations by policy and eval
    data_map = {(e.policy_name, e.eval_name): e for e in evaluations}
    # Include all selected policies, even if they have no evaluations
    policy_names = sorted({e.policy_name for e in evaluations})

    # Build cells
    cells = {}
    for policy_name in policy_names:
        cells[policy_name] = {}
        for eval_name in eval_names:
            eval = data_map.get((policy_name, eval_name))
            cells[policy_name][eval_name] = ScorecardCell(
                evalName=eval_name,
                replayUrl=eval.replay_url if eval else None,
                thumbnailUrl=eval.thumbnail_url if eval else None,
                value=eval.value if eval else 0.0,
            )

    # Calculate averages
    policy_averages = {}
    for policy_name in policy_names:
        scores = [cells[policy_name][name].value for name in eval_names]
        policy_averages[policy_name] = sum(scores) / len(scores) if scores else 0.0

    eval_averages = {}
    eval_max_scores = {}
    for eval_name in eval_names:
        scores = [cells[p][eval_name].value for p in policy_names if p in cells]
        eval_averages[eval_name] = sum(scores) / len(scores) if scores else 0.0
        eval_max_scores[eval_name] = max(scores) if scores else 0.0

    return ScorecardData(
        evalNames=eval_names,
        policyNames=policy_names,
        cells=cells,
        policyAverageScores=policy_averages,
        evalAverageScores=eval_averages,
        evalMaxScores=eval_max_scores,
    )


@dataclass
class LeaderboardTrainingRunScore:
    """Represents a training run score for a leaderboard."""

    policy_id: uuid.UUID
    score: float
    selector_score: float


async def get_leaderboard_training_run_scores(
    con: AsyncConnection, leaderboard_id: uuid.UUID, selector: Literal["latest", "best"]
) -> dict[uuid.UUID, LeaderboardTrainingRunScore]:
    """Get the training run scores for a leaderboard."""

    query = """
      SELECT lps.policy_id, lps.score, e.run_id as training_run_id, e.end_training_epoch
      FROM leaderboard_policy_scores lps
      JOIN policies p ON p.id = lps.policy_id
      JOIN epochs e ON p.epoch_id = e.id
      WHERE lps.leaderboard_id = %s
    """

    @dataclass
    class LeaderboardPolicyTrainingRunScore:
        """Represents a policy score and its associated training run info"""

        policy_id: uuid.UUID
        score: float
        training_run_id: uuid.UUID
        end_training_epoch: int

    async with con.cursor(row_factory=class_row(LeaderboardPolicyTrainingRunScore)) as cursor:
        await cursor.execute(query, (leaderboard_id,))
        rows = await cursor.fetchall()
        rows_by_training_run_id: dict[uuid.UUID, LeaderboardTrainingRunScore] = {}
        for row in rows:
            selector_score = row.score if selector == "best" else row.end_training_epoch
            cur_training_run_best = rows_by_training_run_id.get(row.training_run_id)
            best_selector_score = cur_training_run_best.selector_score if cur_training_run_best else -1
            if selector_score > best_selector_score:
                rows_by_training_run_id[row.training_run_id] = LeaderboardTrainingRunScore(
                    policy_id=row.policy_id, score=row.score, selector_score=selector_score
                )

        return rows_by_training_run_id


async def get_leaderboard_free_policy_scores(con: AsyncConnection, leaderboard_id: uuid.UUID) -> dict[uuid.UUID, float]:
    """Get the free policy scores for a leaderboard."""

    query = """
      SELECT lps.policy_id, lps.score
      FROM leaderboard_policy_scores lps
      JOIN policies p ON p.id = lps.policy_id
      WHERE lps.leaderboard_id = %s AND p.epoch_id IS NULL
    """

    @dataclass
    class QueryRow:
        policy_id: uuid.UUID
        score: float

    async with con.cursor(row_factory=class_row(QueryRow)) as cursor:
        await cursor.execute(query, (leaderboard_id,))
        rows = await cursor.fetchall()
        return {row.policy_id: row.score for row in rows}


# ============================================================================
# API Routes
# ============================================================================


def create_policy_scorecard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create FastAPI router for policy-based scorecard endpoints."""
    router = APIRouter(tags=["scorecard"])

    @router.get("/policies")
    @timed_route("get_policies_and_training_runs")
    async def get_policies() -> PoliciesResponse:
        """Get training runs and run-free policies."""
        async with metta_repo.connect() as con:
            return await get_policies_and_training_runs(con)

    @router.post("/policies/search")
    @timed_route("search_policies_and_training_runs")
    async def search_policies(request: PoliciesSearchRequest) -> PoliciesResponse:
        """Search training runs and run-free policies with filtering."""
        async with metta_repo.connect() as con:
            return await search_policies_and_training_runs(con, request)

    @router.post("/evals")
    @timed_route("get_evals")
    async def get_evals(request: EvalsRequest) -> List[str]:
        """Get evaluation categories and environment names for selected training runs and policies."""
        if not request.training_run_ids and not request.run_free_policy_ids:
            return []

        async with metta_repo.connect() as con:
            return await get_evals_for_selection(con, request.training_run_ids, request.run_free_policy_ids)

    @router.post("/metrics")
    @timed_route("get_available_metrics")
    async def get_available_metrics(request: MetricsRequest) -> List[str]:
        """Get available metrics for selected training runs, policies and evaluations."""
        if (not request.training_run_ids and not request.run_free_policy_ids) or not request.eval_names:
            return []

        async with metta_repo.connect() as con:
            return await get_available_metrics_for_selection(
                con, request.training_run_ids, request.run_free_policy_ids, request.eval_names
            )

    @router.post("/scorecard")
    @router.post("/heatmap")
    @timed_route("generate_policy_scorecard")
    async def generate_policy_scorecard(request: ScorecardRequest) -> ScorecardData:
        """Generate scorecard data based on training run and policy selection."""
        if (
            (not request.training_run_ids and not request.run_free_policy_ids)
            or not request.eval_names
            or not request.metric
        ):
            raise HTTPException(status_code=400, detail="Missing required parameters")

        async with metta_repo.connect() as con:
            # Fetch evaluation data
            evaluations = await fetch_policy_scorecard_data(
                con, request.training_run_ids, request.run_free_policy_ids, request.eval_names, request.metric
            )

            # Apply training run policy selector if we have evaluations
            if evaluations:
                selected_evaluations = select_policies_by_training_run_selector(
                    evaluations, request.training_run_policy_selector, request.eval_names
                )
                # Build and return scorecard (includes all selected policies, even those without evaluations)
                return build_policy_scorecard(selected_evaluations, request.eval_names)
            else:
                # No evaluations found at all - return empty scorecard
                return ScorecardData(
                    evalNames=[],
                    policyNames=[],
                    cells={},
                    policyAverageScores={},
                    evalAverageScores={},
                    evalMaxScores={},
                )

    @router.post("/training-run/{run_id}")
    @timed_route("generate_training_run_scorecard")
    async def generate_training_run_scorecard(run_id: str, request: TrainingRunScorecardRequest) -> ScorecardData:
        """Generate scorecard data for a specific training run showing ALL policies."""
        if not request.eval_names or not request.metric:
            raise HTTPException(status_code=400, detail="Missing required parameters")

        async with metta_repo.connect() as con:
            # Fetch evaluation data for this specific training run
            evaluations = await fetch_policy_scorecard_data(con, [run_id], [], request.eval_names, request.metric)

            # Build scorecard with ALL policies (no policy selection)
            if evaluations:
                return build_policy_scorecard(evaluations, request.eval_names)
            else:
                # No evaluations found - return empty scorecard
                return ScorecardData(
                    evalNames=[],
                    policyNames=[],
                    cells={},
                    policyAverageScores={},
                    evalAverageScores={},
                    evalMaxScores={},
                )

    @router.post("/leaderboard")
    @timed_route("generate_leaderboard_scorecard")
    async def generate_leaderboard_scorecard_route(request: LeaderboardScorecardRequest) -> ScorecardData:
        """Generate scorecard data for a leaderboard in the following way:

        1. Use the leaderboard_policy_scores table to get either the 'latest' or 'best' policy_id for each training run
        2. Use the leaderboard_policy_scores table to get the score for each policy that doesn't have a training run
        3. Combine the lists from 1 and 2 and take the top {leaderboard.num_policies} policies by score
        4. Now that we have top N policies, get the eval scores and replay urls and build the scorecard

        """
        async with metta_repo.connect() as con:
            # Get leaderboard configuration
            leaderboard = await metta_repo.get_leaderboard(request.leaderboard_id)
            if not leaderboard:
                raise HTTPException(status_code=404, detail="Leaderboard not found")

            training_run_scores = await get_leaderboard_training_run_scores(
                con, request.leaderboard_id, request.selector
            )
            free_policy_scores = await get_leaderboard_free_policy_scores(con, request.leaderboard_id)

            @dataclass
            class UnifiedScore:
                id: uuid.UUID
                score: float
                type: Literal["training_run", "policy"]
                policy_id: uuid.UUID

            unified_scores: list[UnifiedScore] = []
            for training_run_id, training_run_score in training_run_scores.items():
                unified_scores.append(
                    UnifiedScore(
                        id=training_run_id,
                        score=training_run_score.score,
                        type="training_run",
                        policy_id=training_run_score.policy_id,
                    )
                )
            for policy_id, score in free_policy_scores.items():
                unified_scores.append(UnifiedScore(id=policy_id, score=score, type="policy", policy_id=policy_id))
            unified_scores.sort(key=lambda x: x.score, reverse=True)
            top_n_scores = unified_scores[: request.num_policies]
            top_n_policy_ids = [score.policy_id for score in top_n_scores]

            # Fetch evaluation data for top policies
            evaluations = await fetch_policy_scorecard_data(
                con, [], top_n_policy_ids, leaderboard.evals, leaderboard.metric
            )

            return build_policy_scorecard(evaluations, leaderboard.evals)

    return router

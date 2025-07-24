import asyncio
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, LiteralString, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from psycopg import AsyncConnection
from psycopg.rows import class_row
from pydantic import BaseModel, Field

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


class EpochHeatmapCell(BaseModel):
    """Single cell in the epoch-based heatmap grid."""

    evalName: str
    epoch: int
    replayUrl: Optional[str]
    value: float


class EpochHeatmapData(BaseModel):
    """Epoch-based heatmap data structure showing multiple epochs per policy."""

    evalNames: List[str]
    epochs: List[int]
    cells: Dict[str, Dict[str, Dict[int, EpochHeatmapCell]]]  # policy -> eval -> epoch -> cell
    policyAverageScores: Dict[str, float]
    evalAverageScores: Dict[str, float]
    evalMaxScores: Dict[str, float]


class PolicySelectorRequest(BaseModel):
    """Request body for policy selection."""

    policy_selector: str = Field(default="latest", pattern="^(latest|best)$")


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


def build_epoch_heatmap(evaluations: List[PolicyEvaluation], all_eval_names: List[str]) -> EpochHeatmapData:
    """Build epoch-based heatmap data structure from evaluations."""
    # Group evaluations by policy, eval, and epoch
    data_map = {}
    policy_uris = set()
    all_epochs = set()

    for eval in evaluations:
        if eval.epoch is not None:
            key = (eval.policy_uri, eval.eval_name, eval.epoch)
            data_map[key] = eval
            policy_uris.add(eval.policy_uri)
            all_epochs.add(eval.epoch)

    policy_uris = sorted(policy_uris)
    epochs = sorted(all_epochs)

    # Build cells
    cells = {}
    for policy_uri in policy_uris:
        cells[policy_uri] = {}
        for eval_name in all_eval_names:
            cells[policy_uri][eval_name] = {}
            for epoch in epochs:
                eval = data_map.get((policy_uri, eval_name, epoch))
                cells[policy_uri][eval_name][epoch] = EpochHeatmapCell(
                    evalName=eval_name,
                    epoch=epoch,
                    replayUrl=eval.replay_url if eval else None,
                    value=eval.value if eval else 0.0,
                )

    # Calculate averages across all epochs for each policy
    policy_averages = {}
    for policy_uri in policy_uris:
        all_scores = []
        for eval_name in all_eval_names:
            for epoch in epochs:
                cell = cells[policy_uri][eval_name][epoch]
                if cell.value > 0:  # Only include non-zero values
                    all_scores.append(cell.value)
        policy_averages[policy_uri] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Calculate averages across all policies for each eval
    eval_averages = {}
    eval_max_scores = {}
    for eval_name in all_eval_names:
        all_scores = []
        for policy_uri in policy_uris:
            for epoch in epochs:
                cell = cells[policy_uri][eval_name][epoch]
                if cell.value > 0:  # Only include non-zero values
                    all_scores.append(cell.value)
        eval_averages[eval_name] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        eval_max_scores[eval_name] = max(all_scores) if all_scores else 0.0

    return EpochHeatmapData(
        evalNames=all_eval_names,
        epochs=epochs,
        cells=cells,
        policyAverageScores=policy_averages,
        evalAverageScores=eval_averages,
        evalMaxScores=eval_max_scores,
    )


# ============================================================================
# Cache Implementation
# ============================================================================


class CachedHeatmap:
    """Cached heatmap data with metadata."""

    def __init__(
        self,
        eval_names: List[str],
        latest_evaluations: List[PolicyEvaluation],
        best_evaluations: List[PolicyEvaluation],
        last_episode_id: int,
    ):
        self.eval_names = eval_names
        self.latest_evaluations = latest_evaluations
        self.best_evaluations = best_evaluations
        self.last_episode_id = last_episode_id


class HeatmapCache:
    """LRU cache for heatmap data with async locking."""

    def __init__(self, metta_repo: MettaRepo, max_size: int = 50):
        self.metta_repo = metta_repo
        self.max_size = max_size
        self.cache: Dict[Tuple[str, str], CachedHeatmap] = {}
        self.access_order: List[Tuple[str, str]] = []
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def log_cache_stats(self) -> None:
        """Log cache statistics."""
        total_rows = sum(len(entry.latest_evaluations) + len(entry.best_evaluations) for entry in self.cache.values())
        logger.info(
            f"Heatmap cache stats: {len(self.cache)} entries, {len(self.access_order)} access order, "
            + f"memory usage: {sys.getsizeof(self.cache)} bytes, "
            + f"rows per entry: {total_rows / len(self.cache)}"
        )

    async def get(self, suite: str, metric: str, selector: str) -> HeatmapData:
        """Get heatmap data, using cache when possible."""
        key = (suite, metric)

        # Get or create a lock for this specific cache key
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]

        # Use the specific lock for this cache key
        async with lock:
            async with self.metta_repo.connect() as con:
                cached = self.cache.get(key)

                if cached:
                    # Update LRU order
                    self._touch(key)

                    # Check for new data
                    has_new_data = await self._has_new_data(con, suite, metric, cached.last_episode_id)

                    if has_new_data:
                        # Update cache with new data
                        cached = await self._update_cache(con, suite, metric, cached)

                else:
                    # Build new cache entry
                    cached = await self._build_cache_entry(con, suite, metric)

            if selector == "best":
                return build_heatmap(cached.best_evaluations, cached.eval_names)
            else:
                return build_heatmap(cached.latest_evaluations, cached.eval_names)

    async def _build_cache_entry(self, con: AsyncConnection, suite: str, metric: str) -> CachedHeatmap:
        """Build a new cache entry."""
        # Fetch all data
        evaluations = await fetch_evaluation_data(con, suite, metric)
        eval_names = await get_evaluation_names(con, suite)
        eval_names_set = set(eval_names)

        latest_evaluations = select_latest_per_run(evaluations)
        best_evaluations = select_best_per_run(evaluations, eval_names_set)

        # Store in cache
        entry = CachedHeatmap(eval_names, latest_evaluations, best_evaluations, max(e.episode_id for e in evaluations))
        self._store(suite, metric, entry)
        self.log_cache_stats()
        return entry

    async def _update_cache(
        self, con: AsyncConnection, suite: str, metric: str, cached: CachedHeatmap
    ) -> CachedHeatmap:
        """Update cached data with new evaluations."""

        # Fetch complete historical data for affected policies only
        affected_evals = await fetch_evaluation_data(
            con,
            suite,
            metric,
            min_episode_id=0,
            policy_filter=SPECIFIC_POLICIES_FILTER,
            filter_params=(suite, cached.last_episode_id),
        )

        # Create map starting with cached evaluations
        latest_eval_map = {(e.policy_uri, e.eval_name): e for e in cached.latest_evaluations}
        best_eval_map = {(e.policy_uri, e.eval_name): e for e in cached.best_evaluations}

        # Replace/update with complete data for affected policies
        for eval in affected_evals:
            latest_eval_map[(eval.policy_uri, eval.eval_name)] = eval
            best_eval_map[(eval.policy_uri, eval.eval_name)] = eval

        # Update eval names if needed
        eval_names = await get_evaluation_names(con, suite)

        latest_evaluations = select_latest_per_run(list(latest_eval_map.values()))
        best_evaluations = select_best_per_run(list(best_eval_map.values()), set(eval_names))

        entry = CachedHeatmap(
            eval_names, latest_evaluations, best_evaluations, max(e.episode_id for e in affected_evals)
        )
        self.cache[(suite, metric)] = entry

        self.log_cache_stats()

        return entry

    async def _has_new_data(self, con: AsyncConnection, suite: str, metric: str, last_episode_id: int) -> bool:
        """Check if there are new episodes since the last cached episode ID."""
        result = await con.execute(HAS_NEW_DATA_QUERY, (suite, last_episode_id))
        row = await result.fetchone()
        return row is not None

    def _touch(self, key: Tuple[str, str]) -> None:
        """Update LRU access order. Must be called while holding the key's lock."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _store(self, suite: str, metric: str, entry: CachedHeatmap) -> None:
        """Store entry with LRU eviction. Must be called while holding the key's lock."""
        key = (suite, metric)

        # Evict if needed
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            # Also clean up the lock for the evicted key
            self._locks.pop(lru_key, None)

        self.cache[key] = entry
        self._touch(key)

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._global_lock:
            self.cache.clear()
            self.access_order.clear()
            self._locks.clear()


# ============================================================================
# API Routes
# ============================================================================


def create_heatmap_router(metta_repo: MettaRepo) -> APIRouter:
    """Create FastAPI router for heatmap endpoints."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])
    cache = HeatmapCache(metta_repo)

    @router.post("/suites/{suite}/metrics/{metric}/heatmap")
    @timed_route("get_heatmap_data")
    async def get_heatmap_data(suite: str, metric: str, request: PolicySelectorRequest) -> HeatmapData:
        """Get cached heatmap data for a suite/metric combination."""
        return await cache.get(suite, metric, request.policy_selector)

    @router.post("/suites/{suite}/metrics/{metric}/epoch-heatmap")
    @timed_route("get_epoch_heatmap_data")
    async def get_epoch_heatmap_data(suite: str, metric: str, request: PolicySelectorRequest) -> EpochHeatmapData:
        """Get epoch-based heatmap data showing multiple epochs per policy."""
        async with metta_repo.connect() as con:
            # Fetch all evaluation data without filtering by latest/best
            evaluations = await fetch_evaluation_data(con, suite, metric)
            eval_names = await get_evaluation_names(con, suite)

            # Build epoch-based heatmap
            return build_epoch_heatmap(evaluations, eval_names)

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

    @router.post("/clear_heatmap_cache")
    @timed_route("clear_heatmap_cache")
    async def clear_heatmap_cache() -> dict:
        """Clear the heatmap cache."""
        await cache.clear()
        return {"message": "Heatmap cache cleared successfully"}

    return router

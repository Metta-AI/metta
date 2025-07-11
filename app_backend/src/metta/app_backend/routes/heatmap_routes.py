import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from psycopg import AsyncConnection
from psycopg.rows import class_row
from psycopg.sql import SQL
from pydantic import BaseModel

from metta.app_backend import query_logger
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route

logger = logging.getLogger("dashboard_performance")
logger.setLevel(logging.INFO)


# Pydantic models for API responses
class HeatmapCell(BaseModel):
    evalName: str
    replayUrl: Optional[str]
    value: float


class HeatmapData(BaseModel):
    evalNames: List[str]
    cells: Dict[str, Dict[str, HeatmapCell]]
    policyAverageScores: Dict[str, float]
    evalAverageScores: Dict[str, float]
    evalMaxScores: Dict[str, float]


class GroupHeatmapMetric(BaseModel):
    policy_selector: Literal["latest", "best"] = "latest"


@dataclass
class DataRow:
    policy_uri: str
    eval_name: str
    replay_url: str | None
    num_agents: int
    total_value: float
    run_id: str | None = None
    end_training_epoch: int | None = None


class DataRetriever(ABC):
    """
    Abstract base class for retrieving evaluation data with encapsulated parameters.
    """

    def __init__(self, suite: str, metric: str, filter_arg: str):
        self.suite = suite
        self.metric = metric
        self.filter_arg = filter_arg

    @abstractmethod
    async def get_data(self, con: AsyncConnection) -> List[DataRow]:
        """
        Retrieve evaluation data.

        Args:
            con: Database connection for querying evaluation data

        Returns:
            List of DataRow containing policy evaluation results
        """
        pass


class PolicySelectorDataRetriever(DataRetriever):
    """
    Retrieves data with policy selector filtering ("latest" or "best").

    Used for the main dashboard heatmap where one policy per training run
    is selected based on the specified strategy.
    """

    async def get_data(self, con: AsyncConnection) -> List[DataRow]:
        return await get_data(con, self.suite, self.metric, self.filter_arg)


class TrainingRunDataRetriever(DataRetriever):
    """
    Retrieves all policies from a specific training run.

    Used for training run detail pages where all policies from the
    specified training run should be included in the heatmap.
    """

    async def get_data(self, con: AsyncConnection) -> List[DataRow]:
        return await get_training_run_data(con, self.suite, self.metric, self.filter_arg)


async def _get_data_with_policy_filter(
    con: AsyncConnection, suite: str, metric: str, policy_cte: SQL, extra_params: Tuple[Any, ...] = ()
) -> List[DataRow]:
    """Core data query with configurable policy filtering."""
    query_template = SQL("""
        WITH
        {} ,

        pre_aggregated AS (
          SELECT
            episode_internal_id,
            SUM(value) as total_value,
            COUNT(*) as agent_count
          FROM episode_agent_metrics
          WHERE metric = %s
          GROUP BY episode_internal_id
        )
        SELECT
          p.name as policy_uri,
          e.env_name as eval_name,
          ANY_VALUE(e.replay_url) as replay_url,
          pa.agent_count AS num_agents,
          pa.total_value AS total_value,
          p.run_id,
          p.end_training_epoch
        FROM episodes e
        JOIN pre_aggregated pa ON e.internal_id = pa.episode_internal_id
        JOIN filtered_policies p ON e.primary_policy_id = p.id
        WHERE e.eval_category = %s
        GROUP BY p.name, e.env_name, p.run_id, p.end_training_epoch, pa.agent_count, pa.total_value
        ORDER BY p.run_id, p.end_training_epoch DESC;
    """)

    query = query_template.format(policy_cte)
    # For optimized query: extra_params come first (for CTE), then base params (for main query)
    params = extra_params + (metric, suite)

    start_time = time.time()
    async with con.cursor(row_factory=class_row(DataRow)) as cursor:
        await cursor.execute(query, params)
        results = await cursor.fetchall()

    end_time = time.time()
    logger.info(f"Get data execution time: {end_time - start_time:.3f}s")
    if end_time - start_time > query_logger.SLOW_QUERY_THRESHOLD_SECONDS:
        logger.warning(f"SLOW QUERY ({end_time - start_time:.3f}s): {query.as_string(con)}, Params: {params}")

    return results


async def get_training_run_data(con: AsyncConnection, suite: str, metric: str, run_id: str) -> List[DataRow]:
    """Get all policies from a specific training run for data."""
    training_run_policy_cte = SQL("""
        filtered_policies AS (
          SELECT
            p.id,
            p.name,
            ep.run_id,
            ep.end_training_epoch
          FROM policies p
          JOIN epochs ep ON p.epoch_id = ep.id
          WHERE ep.run_id = %s
        )
    """)

    return await _get_data_with_policy_filter(con, suite, metric, training_run_policy_cte, (run_id,))


async def get_data(con: AsyncConnection, suite: str, metric: str, policy_selector: str = "latest") -> List[DataRow]:
    """Get data for all policies with policy selector filtering."""
    all_policies_cte = SQL("""
        filtered_policies AS (
          SELECT
            p.id,
            p.name,
            ep.run_id,
            ep.end_training_epoch
          FROM policies p
          JOIN epochs ep ON p.epoch_id = ep.id
          UNION
          SELECT
            p2.id,
            p2.name,
            NULL as run_id,
            NULL as end_training_epoch
          FROM policies p2
          WHERE p2.epoch_id IS NULL
        )
    """)

    rows = await _get_data_with_policy_filter(con, suite, metric, all_policies_cte)
    return await _apply_policy_selector(rows, policy_selector, suite, con)


async def _select_best_policies_per_run(rows: List[DataRow], suite: str, con: AsyncConnection) -> List[DataRow]:
    """
    Select the best policy per training run based on average score across all evaluations.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    Ties are broken by selecting the latest policy (highest end_training_epoch).
    """
    # Group rows by run_id and policy_uri
    run_policies: DefaultDict[str, DefaultDict[str, List[DataRow]]] = defaultdict(lambda: defaultdict(list))
    no_run_rows: List[DataRow] = []

    for row in rows:
        if row.run_id is None:
            no_run_rows.append(row)
        else:
            run_policies[row.run_id][row.policy_uri].append(row)

    # Get all eval_names for this suite to handle missing evaluations
    result = await con.execute("SELECT DISTINCT env_name FROM episodes WHERE eval_category = %s", (suite,))
    eval_rows = await result.fetchall()
    all_eval_names: Set[str] = {row[0] for row in eval_rows if row[0] is not None}

    selected_rows: List[DataRow] = []

    # Process each training run
    for _run_id, policies_dict in run_policies.items():
        best_avg_score = float("-inf")
        best_policy_epoch = float("-inf")
        best_policy_rows: List[DataRow] = []

        # Calculate average score for each policy in this run
        for _policy_uri, policy_rows in policies_dict.items():
            # Create a map of eval_name -> average score for this policy
            eval_scores: Dict[str, float] = {}
            policy_epoch = policy_rows[0].end_training_epoch or 0

            for row in policy_rows:
                avg_score = row.total_value / row.num_agents if row.num_agents > 0 else 0.0
                eval_scores[row.eval_name] = avg_score

            # Calculate average across all evaluations (missing ones default to 0)
            total_score = 0.0
            for eval_name in all_eval_names:
                total_score += eval_scores.get(eval_name, 0.0)

            avg_score = total_score / len(all_eval_names) if all_eval_names else 0.0

            # Select best policy, with ties broken by latest epoch
            if avg_score > best_avg_score or (avg_score == best_avg_score and policy_epoch > best_policy_epoch):
                best_avg_score = avg_score
                best_policy_epoch = policy_epoch
                best_policy_rows = policy_rows

        # Add the best policy's rows to selected_rows
        if best_policy_rows:
            selected_rows.extend(best_policy_rows)

    # Add policies without run_id
    selected_rows.extend(no_run_rows)

    return selected_rows


async def _apply_policy_selector(
    rows: List[DataRow], policy_selector: str, suite: str, con: AsyncConnection
) -> List[DataRow]:
    """
    Apply the specified policy selection strategy to the rows.
    """
    if policy_selector == "latest":
        return _select_latest_policies_per_run(rows)
    elif policy_selector == "best":
        return await _select_best_policies_per_run(rows, suite, con)
    else:
        raise ValueError(f"Invalid policy_selector: {policy_selector}")


def _select_latest_policies_per_run(rows: List[DataRow]) -> List[DataRow]:
    """
    Select the latest policy per training run based on end_training_epoch.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    """
    # Group rows by run_id
    run_policies: DefaultDict[str, List[DataRow]] = defaultdict(list)
    no_run_rows: List[DataRow] = []

    for row in rows:
        if row.run_id is None:
            no_run_rows.append(row)
        else:
            run_policies[row.run_id].append(row)

    selected_rows: List[DataRow] = []

    # For each training run, select the policy with the highest end_training_epoch
    for _run_id, run_rows in run_policies.items():
        if not run_rows:
            continue

        # Group by policy to find the latest epoch for each policy
        policy_latest_epoch: Dict[str, int] = {}
        for row in run_rows:
            policy_uri = row.policy_uri
            epoch = row.end_training_epoch or 0
            if policy_uri not in policy_latest_epoch or epoch > policy_latest_epoch[policy_uri]:
                policy_latest_epoch[policy_uri] = epoch

        # Find the policy with the highest epoch
        latest_policy = max(policy_latest_epoch, key=policy_latest_epoch.get)  # type: ignore
        latest_epoch = policy_latest_epoch[latest_policy]

        # Add all rows for the latest policy
        for row in run_rows:
            if row.policy_uri == latest_policy and row.end_training_epoch == latest_epoch:
                selected_rows.append(row)

    # Add policies without run_id
    selected_rows.extend(no_run_rows)

    return selected_rows


async def _build_heatmap_data(
    con: AsyncConnection,
    data_retriever: DataRetriever,
) -> HeatmapData:
    """
    Core heatmap building logic that can be reused for different policy data sources.

    Args:
        con: Database connection for querying evaluation data
        data_retriever: Configured DataRetriever instance with suite, metric, and filter parameters

    Returns:
        HeatmapData containing evaluation cells, policy averages, and evaluation names
    """

    # Step 1: Get evaluation names
    eval_rows = await execute_query_and_log(
        con,
        "SELECT DISTINCT env_name FROM episodes WHERE eval_category = %s",
        (data_retriever.suite,),
        "get_evaluation_names",
    )
    all_eval_names: List[str] = [row[0] for row in eval_rows]

    # Step 2: Get data
    data_rows = await data_retriever.get_data(con)

    # Step 3: Process policy URIs and values
    all_policy_uris: Set[str] = set()
    for row in data_rows:
        all_policy_uris.add(row.policy_uri)

    data_values: Dict[Tuple[str, str], Tuple[float, str | None]] = {}
    for row in data_rows:
        data_values[(row.policy_uri, row.eval_name)] = (row.total_value / row.num_agents, row.replay_url)

    # Step 4: Build heatmap cells
    cells: Dict[str, Dict[str, HeatmapCell]] = {}
    for policy_uri in all_policy_uris:
        cells[policy_uri] = {}  # Dict[str, HeatmapCell]
        for eval_name in all_eval_names:
            cur_value = data_values.get((policy_uri, eval_name), (0, None))

            cells[policy_uri][eval_name] = HeatmapCell(
                evalName=eval_name,
                replayUrl=cur_value[1],
                value=cur_value[0],
            )

    # Step 5: Calculate policy averages
    policy_average_scores: Dict[str, float] = {}
    for policy_uri in all_policy_uris:
        policy_cells = cells[policy_uri]
        policy_average_scores[policy_uri] = sum(cell.value for cell in policy_cells.values()) / len(policy_cells)

    return HeatmapData(
        evalNames=all_eval_names,
        cells=cells,
        policyAverageScores=policy_average_scores,
        evalAverageScores={},
        evalMaxScores={},
    )


def create_heatmap_router(metta_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    @router.post("/suites/{suite}/metrics/{metric}/heatmap")
    @timed_route("get_heatmap_data")
    async def get_heatmap_data(  # type: ignore[reportUnusedFunction]
        suite: str,
        metric: str,
        group_metric: GroupHeatmapMetric,
    ) -> HeatmapData:
        """Get heatmap data for a given suite, metric, and group metric."""
        async with metta_repo.connect() as con:
            data_retriever = PolicySelectorDataRetriever(suite, metric, group_metric.policy_selector)
            return await _build_heatmap_data(con, data_retriever)

    @router.get("/training-runs/{run_id}/suites/{suite}/metrics/{metric}/heatmap")
    @timed_route("get_training_run_heatmap_data")
    async def get_training_run_heatmap_data(  # type: ignore[reportUnusedFunction]
        run_id: str,
        suite: str,
        metric: str,
    ) -> HeatmapData:
        """Get heatmap data for a specific training run."""
        # Verify training run exists
        training_run = await metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")

        async with metta_repo.connect() as con:
            data_retriever = TrainingRunDataRetriever(suite, metric, run_id)
            return await _build_heatmap_data(con, data_retriever)

    return router

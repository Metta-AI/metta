import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException
from psycopg import Connection
from psycopg.rows import class_row
from psycopg.sql import SQL
from pydantic import BaseModel

from app_backend.auth import create_user_or_token_dependency
from app_backend.metta_repo import MettaRepo

# Set up logging for heatmap performance analysis
logger = logging.getLogger("heatmap_performance")
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


class GroupDiff(BaseModel):
    group_1: str
    group_2: str


class GroupHeatmapMetric(BaseModel):
    group_metric: str | GroupDiff
    policy_selector: Literal["latest", "best"] = "latest"


class SavedDashboardCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    dashboard_state: Dict[str, Any]


class SavedDashboardResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    type: str
    dashboard_state: Dict[str, Any]
    created_at: str
    updated_at: str
    user_id: str


class SavedDashboardListResponse(BaseModel):
    dashboards: List[SavedDashboardResponse]


class TrainingRun(BaseModel):
    id: str
    name: str
    created_at: str
    user_id: str
    finished_at: Optional[str]
    status: str
    url: Optional[str]


class TrainingRunListResponse(BaseModel):
    training_runs: List[TrainingRun]


@dataclass
class GroupDataRow:
    policy_uri: str
    eval_name: str
    replay_url: str | None
    num_agents: int
    total_value: float
    run_id: str | None = None
    end_training_epoch: int | None = None


class GroupDataRetriever(ABC):
    """
    Abstract base class for retrieving evaluation data with encapsulated parameters.
    """

    def __init__(self, suite: str, metric: str, filter_arg: str):
        self.suite = suite
        self.metric = metric
        self.filter_arg = filter_arg

    @abstractmethod
    def get_group_data(self, con: Connection, group: str) -> List[GroupDataRow]:
        """
        Retrieve group evaluation data for the specified group.

        Args:
            con: Database connection for querying evaluation data
            group: Agent group identifier (empty string for all groups)

        Returns:
            List of GroupDataRow containing policy evaluation results
        """
        pass


class PolicySelectorDataRetriever(GroupDataRetriever):
    """
    Retrieves group data with policy selector filtering ("latest" or "best").

    Used for the main dashboard heatmap where one policy per training run
    is selected based on the specified strategy.
    """

    def get_group_data(self, con: Connection, group: str) -> List[GroupDataRow]:
        return get_group_data(con, self.suite, self.metric, group, self.filter_arg)


class TrainingRunDataRetriever(GroupDataRetriever):
    """
    Retrieves all policies from a specific training run.

    Used for training run detail pages where all policies from the
    specified training run should be included in the heatmap.
    """

    def get_group_data(self, con: Connection, group: str) -> List[GroupDataRow]:
        return get_training_run_group_data(con, self.suite, self.metric, group, self.filter_arg)


def _get_group_data_with_policy_filter(
    con: Connection, suite: str, metric: str, group: str, policy_cte: SQL, extra_params: Tuple[Any, ...] = ()
) -> List[GroupDataRow]:
    """Core group data query with configurable policy filtering."""
    start_time = time.time()
    logger.info(f"Starting core group data query: suite={suite}, metric={metric}, group='{group}'")
    
    try:
        if group == "":
            # Optimized query for all groups - avoids expensive JSON parsing and reduces CTEs
            logger.info("Using optimized all-groups query (no JSON parsing, direct joins)")
            query_template = SQL("""
                WITH
                {}
                
                SELECT
                  p.name as policy_uri,
                  e.env_name as eval_name,
                  ANY_VALUE(e.replay_url) as replay_url,
                  COUNT(*) AS num_agents,
                  SUM(eam.value) AS total_value,
                  p.run_id,
                  p.end_training_epoch
                FROM episode_agent_metrics eam
                JOIN episodes e ON e.id = eam.episode_id AND e.eval_category = %s
                JOIN filtered_policies p ON e.primary_policy_id = p.id
                WHERE eam.metric = %s
                GROUP BY p.name, e.env_name, p.run_id, p.end_training_epoch
                ORDER BY p.run_id, p.end_training_epoch DESC
            """)
            
            query = query_template.format(policy_cte)
            # For optimized query: extra_params come first (for CTE), then base params (for main query)
            params = extra_params + (suite, metric)
            
        else:
            # Original query with group filtering - includes JSON parsing only when needed
            logger.info(f"Using group-filtered query with JSON parsing for group '{group}'")
            query_template = SQL("""
                WITH
                filtered_episodes AS (
                    SELECT e.id, e.env_name, e.primary_policy_id, e.replay_url, e.attributes
                    FROM episodes e
                    WHERE e.eval_category = %s
                ),
                episode_agent_metrics_with_group_id AS (
                    SELECT
                        eam.episode_id,
                        eam.agent_id,
                        eam.value,
                        CAST ((fe.attributes->'agent_groups')[eam.agent_id] AS INTEGER) as group_id,
                        fe.env_name,
                        fe.primary_policy_id,
                        fe.replay_url
                    FROM episode_agent_metrics eam
                    JOIN filtered_episodes fe ON fe.id = eam.episode_id
                    WHERE eam.metric = %s
                ),
                {}

                SELECT
                  p.name as policy_uri,
                  eam.env_name as eval_name,
                  ANY_VALUE(eam.replay_url) as replay_url,
                  COUNT(*) AS num_agents,
                  SUM(eam.value) AS total_value,
                  p.run_id,
                  p.end_training_epoch
                FROM episode_agent_metrics_with_group_id eam
                JOIN filtered_policies p ON eam.primary_policy_id = p.id
                WHERE eam.group_id = %s
                GROUP BY p.name, eam.env_name, p.run_id, p.end_training_epoch
                ORDER BY p.run_id, p.end_training_epoch DESC
            """)
            
            query = query_template.format(policy_cte)
            base_params = (suite, metric) + extra_params
            params = base_params + (group,)

        query_start = time.time()
        logger.info(f"Executing {'optimized' if group == '' else 'group-filtered'} database query with params: {params}")
        
        with con.cursor(row_factory=class_row(GroupDataRow)) as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            
        query_time = time.time() - query_start
        total_time = time.time() - start_time
        
        logger.info(f"Core query completed: {len(results)} rows in {query_time:.3f}s (total: {total_time:.3f}s)")
        return results
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Core query failed after {total_time:.3f}s: {e}")
        raise


def get_training_run_group_data(
    con: Connection, suite: str, metric: str, group: str, run_id: str
) -> List[GroupDataRow]:
    """Get all policies from a specific training run for group data."""
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

    return _get_group_data_with_policy_filter(con, suite, metric, group, training_run_policy_cte, (run_id,))


def get_group_data(
    con: Connection, suite: str, metric: str, group: str, policy_selector: str = "latest"
) -> List[GroupDataRow]:
    """Get group data for all policies with policy selector filtering."""
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

    rows = _get_group_data_with_policy_filter(con, suite, metric, group, all_policies_cte)
    return _apply_policy_selector(rows, policy_selector, suite, con)


def _apply_policy_selector(
    rows: List[GroupDataRow], policy_selector: str, suite: str, con: Connection
) -> List[GroupDataRow]:
    """
    Apply the specified policy selection strategy to the rows.
    """
    start_time = time.time()
    logger.info(f"Applying policy selector '{policy_selector}' to {len(rows)} rows")
    
    try:
        if policy_selector == "latest":
            result = _select_latest_policies_per_run(rows)
        elif policy_selector == "best":
            result = _select_best_policies_per_run(rows, suite, con)
        else:
            raise ValueError(f"Invalid policy_selector: {policy_selector}")
        
        selector_time = time.time() - start_time
        logger.info(f"Policy selector '{policy_selector}' completed: {len(result)} rows in {selector_time:.3f}s")
        return result
        
    except Exception as e:
        selector_time = time.time() - start_time
        logger.error(f"Policy selector '{policy_selector}' failed after {selector_time:.3f}s: {e}")
        raise


def _select_latest_policies_per_run(rows: List[GroupDataRow]) -> List[GroupDataRow]:
    """
    Select the latest policy per training run based on end_training_epoch.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    """
    start_time = time.time()
    logger.debug(f"Selecting latest policies from {len(rows)} rows")
    # Group rows by run_id
    run_policies: DefaultDict[str, List[GroupDataRow]] = defaultdict(list)
    no_run_rows: List[GroupDataRow] = []

    for row in rows:
        if row.run_id is None:
            no_run_rows.append(row)
        else:
            run_policies[row.run_id].append(row)

    selected_rows: List[GroupDataRow] = []

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

    latest_time = time.time() - start_time
    logger.debug(f"Latest policy selection completed: {len(selected_rows)} rows in {latest_time:.3f}s")
    return selected_rows


def _select_best_policies_per_run(rows: List[GroupDataRow], suite: str, con: Connection) -> List[GroupDataRow]:
    """
    Select the best policy per training run based on average score across all evaluations.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    Ties are broken by selecting the latest policy (highest end_training_epoch).

    Optimized version that avoids extra database queries and simplifies scoring logic.
    """
    start_time = time.time()
    logger.debug(f"Selecting best policies from {len(rows)} rows")
    
    # Group rows by run_id and policy_uri
    run_policies: DefaultDict[str, DefaultDict[str, List[GroupDataRow]]] = defaultdict(lambda: defaultdict(list))
    no_run_rows: List[GroupDataRow] = []

    for row in rows:
        if row.run_id is None:
            no_run_rows.append(row)
        else:
            run_policies[row.run_id][row.policy_uri].append(row)

    selected_rows: List[GroupDataRow] = []

    # Process each training run
    for _run_id, policies_dict in run_policies.items():
        best_avg_score = float("-inf")
        best_policy_epoch = float("-inf")
        best_policy_rows: List[GroupDataRow] = []

        # Calculate average score for each policy in this run
        for _policy_uri, policy_rows in policies_dict.items():
            policy_epoch = policy_rows[0].end_training_epoch or 0

            # Calculate average score across available evaluations for this policy
            total_score = 0.0
            for row in policy_rows:
                avg_score = row.total_value / row.num_agents if row.num_agents > 0 else 0.0
                total_score += avg_score

            # Simple average across evaluations this policy actually participated in
            avg_score = total_score / len(policy_rows) if policy_rows else 0.0

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

    best_time = time.time() - start_time
    logger.debug(f"Best policy selection completed: {len(selected_rows)} rows in {best_time:.3f}s")
    return selected_rows


def create_dashboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a dashboard router with the given StatsRepo instance."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.get("/suites")
    async def get_suites() -> List[str]:  # type: ignore[reportUnusedFunction]
        return metta_repo.get_suites()

    @router.get("/suites/{suite}/metrics")
    async def get_metrics(suite: str) -> List[str]:  # type: ignore[reportUnusedFunction]
        return metta_repo.get_metrics(suite)

    @router.get("/suites/{suite}/group-ids")
    async def get_group_ids(suite: str) -> List[str]:  # type: ignore[reportUnusedFunction]
        return metta_repo.get_group_ids(suite)

    def _build_heatmap_data(
        con: Connection,
        group_metric: GroupHeatmapMetric,
        data_retriever: GroupDataRetriever,
    ) -> HeatmapData:
        """
        Core heatmap building logic that can be reused for different policy data sources.

        Args:
            con: Database connection for querying evaluation data
            group_metric: Group metric specification (single group or group difference)
            data_retriever: Configured GroupDataRetriever instance with suite, metric, and filter parameters

        Returns:
            HeatmapData containing evaluation cells, policy averages, and evaluation names
        """
        total_start = time.time()
        logger.info(f"Building heatmap data: suite={data_retriever.suite}, metric={data_retriever.metric}")
        
        try:
            # Step 1: Get evaluation names
            eval_start = time.time()
            eval_rows = con.execute(
                "SELECT DISTINCT env_name FROM episodes WHERE eval_category = %s", (data_retriever.suite,)
            )
            all_eval_names: List[str] = [row[0] for row in eval_rows]
            eval_time = time.time() - eval_start
            logger.info(f"Retrieved {len(all_eval_names)} evaluation names in {eval_time:.3f}s")

            # Step 2: Get group data
            groups_start = time.time()
            if isinstance(group_metric.group_metric, GroupDiff):
                logger.info(f"Fetching group difference data: {group_metric.group_metric.group_1} vs {group_metric.group_metric.group_2}")
                group1_rows = data_retriever.get_group_data(con, group_metric.group_metric.group_1)
                group2_rows = data_retriever.get_group_data(con, group_metric.group_metric.group_2)
            else:
                logger.info(f"Fetching single group data: '{group_metric.group_metric}'")
                group1_rows = data_retriever.get_group_data(con, group_metric.group_metric)
                group2_rows: List[GroupDataRow] = []
            groups_time = time.time() - groups_start
            logger.info(f"Retrieved group data: {len(group1_rows)} + {len(group2_rows)} rows in {groups_time:.3f}s")

            # Step 3: Process policy URIs and values
            processing_start = time.time()
            all_policy_uris: Set[str] = set()
            for row in group1_rows:
                all_policy_uris.add(row.policy_uri)
            for row in group2_rows:
                all_policy_uris.add(row.policy_uri)
            logger.info(f"Found {len(all_policy_uris)} unique policies")

            group_1_values: Dict[Tuple[str, str], Tuple[float, str | None]] = {}
            group_2_values: Dict[Tuple[str, str], Tuple[float, str | None]] = {}
            for row in group1_rows:
                group_1_values[(row.policy_uri, row.eval_name)] = (row.total_value / row.num_agents, row.replay_url)
            for row in group2_rows:
                group_2_values[(row.policy_uri, row.eval_name)] = (row.total_value / row.num_agents, row.replay_url)

            # Step 4: Build heatmap cells
            cells_start = time.time()
            cells: Dict[str, Dict[str, HeatmapCell]] = {}
            for policy_uri in all_policy_uris:
                cells[policy_uri] = {}  # Dict[str, HeatmapCell]
                for eval_name in all_eval_names:
                    group_1_value = group_1_values.get((policy_uri, eval_name), (0, None))
                    group_2_value = group_2_values.get((policy_uri, eval_name), (0, None))

                    cells[policy_uri][eval_name] = HeatmapCell(
                        evalName=eval_name,
                        replayUrl=group_1_value[1] if group_1_value[1] is not None else group_2_value[1],
                        value=group_1_value[0] - group_2_value[0],
                    )
            cells_time = time.time() - cells_start
            logger.info(f"Built {len(all_policy_uris) * len(all_eval_names)} heatmap cells in {cells_time:.3f}s")

            # Step 5: Calculate policy averages
            averages_start = time.time()
            policy_average_scores: Dict[str, float] = {}
            for policy_uri in all_policy_uris:
                policy_cells = cells[policy_uri]
                policy_average_scores[policy_uri] = sum(cell.value for cell in policy_cells.values()) / len(policy_cells)
            averages_time = time.time() - averages_start
            logger.info(f"Calculated {len(policy_average_scores)} policy averages in {averages_time:.3f}s")

            processing_time = time.time() - processing_start
            total_time = time.time() - total_start
            logger.info(f"Heatmap data building completed in {total_time:.3f}s (processing: {processing_time:.3f}s)")

            return HeatmapData(
                evalNames=all_eval_names,
                cells=cells,
                policyAverageScores=policy_average_scores,
                evalAverageScores={},
                evalMaxScores={},
            )
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"Heatmap data building failed after {total_time:.3f}s: {e}")
            raise

    @router.post("/suites/{suite}/metrics/{metric}/heatmap")
    async def get_heatmap_data(  # type: ignore[reportUnusedFunction]
        suite: str,
        metric: str,
        group_metric: GroupHeatmapMetric,
    ) -> HeatmapData:
        """Get heatmap data for a given suite, metric, and group metric."""
        request_start = time.time()
        request_id = f"{suite}/{metric}/{group_metric.group_metric}/{group_metric.policy_selector}"
        logger.info(f"HEATMAP REQUEST START: {request_id}")
        
        try:
            # Step 1: Database connection
            conn_start = time.time()
            with metta_repo.connect() as con:
                conn_time = time.time() - conn_start
                logger.info(f"Database connection established in {conn_time:.3f}s")
                
                # Step 2: Data retrieval and processing
                data_retriever = PolicySelectorDataRetriever(suite, metric, group_metric.policy_selector)
                result = _build_heatmap_data(con, group_metric, data_retriever)
                
                # Step 3: Final response
                total_time = time.time() - request_start
                result_size = len(result.cells) * len(result.evalNames) if result.cells else 0
                logger.info(f"HEATMAP REQUEST COMPLETE: {request_id} - {result_size} cells in {total_time:.3f}s")
                
                return result
                
        except Exception as e:
            total_time = time.time() - request_start
            logger.error(f"HEATMAP REQUEST FAILED: {request_id} after {total_time:.3f}s - {e}")
            raise

    @router.get("/saved")
    async def list_saved_dashboards() -> SavedDashboardListResponse:  # type: ignore[reportUnusedFunction]
        """List all saved dashboards."""
        dashboards = metta_repo.list_saved_dashboards()
        return SavedDashboardListResponse(
            dashboards=[
                SavedDashboardResponse(
                    id=dashboard["id"],
                    name=dashboard["name"],
                    description=dashboard["description"],
                    type=dashboard["type"],
                    dashboard_state=dashboard["dashboard_state"],
                    created_at=dashboard["created_at"].isoformat(),
                    updated_at=dashboard["updated_at"].isoformat(),
                    user_id=dashboard["user_id"],
                )
                for dashboard in dashboards
            ]
        )

    @router.get("/saved/{dashboard_id}")
    async def get_saved_dashboard(dashboard_id: str) -> SavedDashboardResponse:  # type: ignore[reportUnusedFunction]
        """Get a specific saved dashboard by ID."""
        dashboard = metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.post("/saved")
    async def create_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_data: SavedDashboardCreate,
        user_or_token: str = user_or_token,
    ) -> SavedDashboardResponse:
        """Create a new saved dashboard (always creates a new row, even if name is duplicate)."""
        dashboard_id = metta_repo.create_saved_dashboard(
            user_id=user_or_token,
            name=dashboard_data.name,
            description=dashboard_data.description,
            dashboard_type=dashboard_data.type,
            dashboard_state=dashboard_data.dashboard_state,
        )

        # Fetch the created dashboard to return
        dashboard = metta_repo.get_saved_dashboard(str(dashboard_id))
        if not dashboard:
            raise HTTPException(status_code=500, detail="Failed to create dashboard")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.put("/saved/{dashboard_id}")
    async def update_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str,
        dashboard_data: SavedDashboardCreate,
        user_or_token: str = user_or_token,
    ) -> SavedDashboardResponse:
        """Update an existing saved dashboard."""
        success = metta_repo.update_saved_dashboard(
            user_id=user_or_token,
            dashboard_id=dashboard_id,
            name=dashboard_data.name,
            description=dashboard_data.description,
            dashboard_type=dashboard_data.type,
            dashboard_state=dashboard_data.dashboard_state,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        # Fetch the updated dashboard to return
        dashboard = metta_repo.get_saved_dashboard(dashboard_id)
        if not dashboard:
            raise HTTPException(status_code=500, detail="Failed to fetch updated dashboard")

        return SavedDashboardResponse(
            id=dashboard["id"],
            name=dashboard["name"],
            description=dashboard["description"],
            type=dashboard["type"],
            dashboard_state=dashboard["dashboard_state"],
            created_at=dashboard["created_at"].isoformat(),
            updated_at=dashboard["updated_at"].isoformat(),
            user_id=dashboard["user_id"],
        )

    @router.delete("/saved/{dashboard_id}")
    async def delete_saved_dashboard(  # type: ignore[reportUnusedFunction]
        dashboard_id: str, user_or_token: str = user_or_token
    ) -> Dict[str, str]:
        """Delete a saved dashboard."""
        success = metta_repo.delete_saved_dashboard(user_or_token, dashboard_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return {"message": "Dashboard deleted successfully"}

    @router.get("/training-runs")
    async def get_training_runs() -> TrainingRunListResponse:  # type: ignore[reportUnusedFunction]
        """Get all training runs."""
        training_runs = metta_repo.get_training_runs()
        return TrainingRunListResponse(
            training_runs=[
                TrainingRun(
                    id=run["id"],
                    name=run["name"],
                    created_at=run["created_at"],
                    user_id=run["user_id"],
                    finished_at=run["finished_at"],
                    status=run["status"],
                    url=run["url"],
                )
                for run in training_runs
            ]
        )

    @router.get("/training-runs/{run_id}")
    async def get_training_run(run_id: str) -> TrainingRun:  # type: ignore[reportUnusedFunction]
        """Get a specific training run by ID."""
        training_run = metta_repo.get_training_run(run_id)
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")

        return TrainingRun(
            id=training_run["id"],
            name=training_run["name"],
            created_at=training_run["created_at"],
            user_id=training_run["user_id"],
            finished_at=training_run["finished_at"],
            status=training_run["status"],
            url=training_run["url"],
        )

    @router.post("/training-runs/{run_id}/suites/{suite}/metrics/{metric}/heatmap")
    async def get_training_run_heatmap_data(  # type: ignore[reportUnusedFunction]
        run_id: str,
        suite: str,
        metric: str,
        group_metric: GroupHeatmapMetric,
    ) -> HeatmapData:
        """Get heatmap data for a specific training run."""
        request_start = time.time()
        request_id = f"run:{run_id}/{suite}/{metric}/{group_metric.group_metric}"
        logger.info(f"TRAINING RUN HEATMAP REQUEST START: {request_id}")
        
        try:
            # Verify training run exists
            training_run = metta_repo.get_training_run(run_id)
            if not training_run:
                raise HTTPException(status_code=404, detail="Training run not found")

            with metta_repo.connect() as con:
                data_retriever = TrainingRunDataRetriever(suite, metric, run_id)
                result = _build_heatmap_data(con, group_metric, data_retriever)
                
                total_time = time.time() - request_start
                result_size = len(result.cells) * len(result.evalNames) if result.cells else 0
                logger.info(f"TRAINING RUN HEATMAP COMPLETE: {request_id} - {result_size} cells in {total_time:.3f}s")
                
                return result
                
        except Exception as e:
            total_time = time.time() - request_start
            logger.error(f"TRAINING RUN HEATMAP FAILED: {request_id} after {total_time:.3f}s - {e}")
            raise

    return router

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


@dataclass
class GroupDataRow:
    policy_uri: str
    eval_name: str
    replay_url: str | None
    num_agents: int
    total_value: float
    run_id: str | None = None
    end_training_epoch: int | None = None


def get_group_data(
    con: Connection, suite: str, metric: str, group: str, policy_selector: str = "latest"
) -> List[GroupDataRow]:
    # Single SQL query that gets all policies with their training run information
    query_template: SQL = SQL("""
        WITH
        episode_agent_metrics_with_group_id AS (
            SELECT
                eam.*,
                CAST ((e.attributes->'agent_groups')[eam.agent_id] AS INTEGER) as group_id
            FROM episode_agent_metrics eam
            JOIN episode_view e ON e.id = eam.episode_id
            WHERE e.simulation_suite = %s AND eam.metric = %s
        ),
        all_policies_with_runs AS (
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

        SELECT
          p.name as policy_uri,
          e.eval_name,
          ANY_VALUE(e.replay_url) as replay_url,
          COUNT(*) AS num_agents,
          SUM(eam.value) AS total_value,
          p.run_id,
          p.end_training_epoch
        FROM episode_agent_metrics_with_group_id eam
        JOIN episode_view e ON e.id = eam.episode_id
        JOIN all_policies_with_runs p ON e.primary_policy_id = p.id
        {}
        GROUP BY p.name, e.eval_name, p.run_id, p.end_training_epoch
        ORDER BY p.run_id, p.end_training_epoch DESC
    """)

    where_clause = SQL("")
    if group != "":
        where_clause = SQL("WHERE eam.group_id = %s")

    query = query_template.format(where_clause)
    params = (suite, metric, group) if group != "" else (suite, metric)

    with con.cursor(row_factory=class_row(GroupDataRow)) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

        return _apply_policy_selector(rows, policy_selector, suite, con)


def _apply_policy_selector(
    rows: List[GroupDataRow], policy_selector: str, suite: str, con: Connection
) -> List[GroupDataRow]:
    """
    Apply the specified policy selection strategy to the rows.
    """
    if policy_selector == "latest":
        return _select_latest_policies_per_run(rows)
    elif policy_selector == "best":
        return _select_best_policies_per_run(rows, suite, con)
    else:
        raise ValueError(f"Invalid policy_selector: {policy_selector}")


def _select_latest_policies_per_run(rows: List[GroupDataRow]) -> List[GroupDataRow]:
    """
    Select the latest policy per training run based on end_training_epoch.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    """
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

    return selected_rows


def _select_best_policies_per_run(rows: List[GroupDataRow], suite: str, con: Connection) -> List[GroupDataRow]:
    """
    Select the best policy per training run based on average score across all evaluations.
    For policies with no run_id (epoch_id is NULL), include them as-is.
    Ties are broken by selecting the latest policy (highest end_training_epoch).
    """
    # Group rows by run_id and policy_uri
    run_policies: DefaultDict[str, DefaultDict[str, List[GroupDataRow]]] = defaultdict(lambda: defaultdict(list))
    no_run_rows: List[GroupDataRow] = []

    for row in rows:
        if row.run_id is None:
            no_run_rows.append(row)
        else:
            run_policies[row.run_id][row.policy_uri].append(row)

    # Get all eval_names for this suite to handle missing evaluations
    eval_rows = con.execute("SELECT DISTINCT eval_name FROM episode_view WHERE simulation_suite = %s", (suite,))
    all_eval_names: Set[str] = {row[0] for row in eval_rows}

    selected_rows: List[GroupDataRow] = []

    # Process each training run
    for _run_id, policies_dict in run_policies.items():
        best_avg_score = float("-inf")
        best_policy_epoch = float("-inf")
        best_policy_rows: List[GroupDataRow] = []

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


def create_dashboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a dashboard router with the given StatsRepo instance."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.get("/suites")
    async def get_suites() -> List[str]:
        return metta_repo.get_suites()

    @router.get("/suites/{suite}/metrics")
    async def get_metrics(suite: str) -> List[str]:
        return metta_repo.get_metrics(suite)

    @router.get("/suites/{suite}/group-ids")
    async def get_group_ids(suite: str) -> List[str]:
        return metta_repo.get_group_ids(suite)

    @router.post("/suites/{suite}/metrics/{metric}/heatmap")
    async def get_heatmap_data(
        suite: str,
        metric: str,
        group_metric: GroupHeatmapMetric,
    ) -> HeatmapData:
        """Get heatmap data for a given suite, metric, and group metric."""
        with metta_repo.connect() as con:
            eval_rows = con.execute("SELECT DISTINCT eval_name FROM episode_view WHERE simulation_suite = %s", (suite,))
            all_eval_names: List[str] = [row[0] for row in eval_rows]

            if isinstance(group_metric.group_metric, GroupDiff):
                group1_rows = get_group_data(
                    con, suite, metric, group_metric.group_metric.group_1, group_metric.policy_selector
                )
                group2_rows = get_group_data(
                    con, suite, metric, group_metric.group_metric.group_2, group_metric.policy_selector
                )
            else:
                group1_rows = get_group_data(
                    con, suite, metric, group_metric.group_metric, group_metric.policy_selector
                )
                group2_rows: List[GroupDataRow] = []

            all_policy_uris: Set[str] = set()
            for row in group1_rows:
                all_policy_uris.add(row.policy_uri)
            for row in group2_rows:
                all_policy_uris.add(row.policy_uri)

            group_1_values: Dict[Tuple[str, str], Tuple[float, str | None]] = {}
            group_2_values: Dict[Tuple[str, str], Tuple[float, str | None]] = {}
            for row in group1_rows:
                group_1_values[(row.policy_uri, row.eval_name)] = (row.total_value / row.num_agents, row.replay_url)
            for row in group2_rows:
                group_2_values[(row.policy_uri, row.eval_name)] = (row.total_value / row.num_agents, row.replay_url)

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

            policy_average_scores: Dict[str, float] = {}
            for policy_uri in all_policy_uris:
                policy_cells = cells[policy_uri]
                policy_average_scores[policy_uri] = sum(cell.value for cell in policy_cells.values()) / len(
                    policy_cells
                )

            return HeatmapData(
                evalNames=all_eval_names,
                cells=cells,
                policyAverageScores=policy_average_scores,
                evalAverageScores={},
                evalMaxScores={},
            )

    @router.get("/saved")
    async def list_saved_dashboards() -> SavedDashboardListResponse:
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
    async def get_saved_dashboard(dashboard_id: str) -> SavedDashboardResponse:
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
    async def create_saved_dashboard(
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
    async def update_saved_dashboard(
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
    async def delete_saved_dashboard(dashboard_id: str, user_or_token: str = user_or_token) -> Dict[str, str]:
        """Delete a saved dashboard."""
        success = metta_repo.delete_saved_dashboard(user_or_token, dashboard_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return {"message": "Dashboard deleted successfully"}

    return router

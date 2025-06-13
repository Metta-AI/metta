from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter
from psycopg import Connection
from psycopg.rows import class_row
from psycopg.sql import SQL
from pydantic import BaseModel

from metta.app.metta_repo import MettaRepo


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


@dataclass
class GroupDataRow:
    policy_uri: str
    eval_name: str
    replay_url: str | None
    num_agents: int
    total_value: float


def get_group_data(con: Connection, suite: str, metric: str, group: str) -> List[GroupDataRow]:
    query_template: SQL = SQL("""
        WITH episode_agent_metrics_with_group_id AS (
            SELECT
                eam.*,
                CAST ((e.attributes->'agent_groups')[eam.agent_id] AS INTEGER) as group_id
            FROM episode_agent_metrics eam
            JOIN episodes e ON e.id = eam.episode_id
            WHERE e.simulation_suite = %s AND eam.metric = %s
        )

        SELECT
          p.name as policy_uri,
          e.eval_name,
          ANY_VALUE(e.replay_url) as replay_url,
          COUNT(*) AS num_agents,
          SUM(eam.value) AS total_value
        FROM episode_agent_metrics_with_group_id eam
        JOIN episodes e ON e.id = eam.episode_id
        JOIN policies p ON e.primary_policy_id = p.id
        {}
        GROUP BY p.name, e.eval_name
    """)

    where_clause = SQL("")
    if group != "":
        where_clause = SQL("WHERE eam.group_id = %s")

    query = query_template.format(where_clause)
    params = (suite, metric, group) if group != "" else (suite, metric)

    with con.cursor(row_factory=class_row(GroupDataRow)) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def create_dashboard_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a dashboard router with the given StatsRepo instance."""
    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

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
            eval_rows = con.execute("SELECT DISTINCT eval_name FROM episodes WHERE simulation_suite = %s", (suite,))
            all_eval_names: list[str] = [row[0] for row in eval_rows]

            if isinstance(group_metric.group_metric, GroupDiff):
                group1_rows = get_group_data(con, suite, metric, group_metric.group_metric.group_1)
                group2_rows = get_group_data(con, suite, metric, group_metric.group_metric.group_2)
            else:
                group1_rows = get_group_data(con, suite, metric, group_metric.group_metric)
                group2_rows: List[GroupDataRow] = []

            all_policy_uris: set[str] = set()
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
                cells[policy_uri] = {}
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

    return router

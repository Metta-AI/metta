from typing import Any, Dict, List, Optional

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mettagrid.postgres_stats_db import PostgresStatsDB

app = fastapi.FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stats_db_uri = "postgres://postgres:password@127.0.0.1/postgres"


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


# Type alias for group metric
GroupHeatmapMetric = Optional[str] | GroupDiff


@app.get("/api/suites")
async def get_suites() -> List[str]:
    """Get all available simulation suites."""
    with PostgresStatsDB(stats_db_uri) as db:
        result = db.query("""
            SELECT DISTINCT simulation_suite
            FROM episodes
            WHERE simulation_suite IS NOT NULL
            ORDER BY simulation_suite
        """)
        return [row[0] for row in result]


@app.get("/api/metrics/{suite}")
async def get_metrics(suite: str) -> List[str]:
    """Get all available metrics for a given suite."""
    with PostgresStatsDB(stats_db_uri) as db:
        result = db.query(
            """
            SELECT DISTINCT eam.metric
            FROM episodes e
            JOIN episode_agent_metrics eam ON e.id = eam.episode_id
            WHERE e.simulation_suite = %s
            ORDER BY eam.metric
        """,
            (suite,),
        )
        return [row[0] for row in result]


@app.get("/api/group-ids/{suite}")
async def get_group_ids(suite: str) -> List[str]:
    """Get all available group IDs for a given suite."""
    with PostgresStatsDB(stats_db_uri) as db:
        result = db.query(
            """
            SELECT DISTINCT jsonb_object_keys(e.attributes->'agent_groups') as group_id
            FROM episodes e
            WHERE e.simulation_suite = %s
            ORDER BY group_id
        """,
            (suite,),
        )
        return [row[0] for row in result]


@app.get("/api/heatmap-data/{suite}/{metric}")
async def get_heatmap_data(suite: str, metric: str, group_metric: Optional[str] = None) -> HeatmapData:
    """Get heatmap data for a given suite, metric, and group metric."""

    with PostgresStatsDB(stats_db_uri) as db:
        # Get all episodes for the suite with their data
        result = db.query(
            """
            SELECT
                p.name as policy_uri,
                e.eval_name,
                e.replay_url,
                e.attributes->'agent_groups' as agent_groups,
                eam.agent_id,
                eam.metric,
                eam.value
            FROM episodes e
            JOIN episode_agent_policies eap ON e.id = eap.episode_id
            JOIN policies p ON eap.policy_id = p.id
            JOIN episode_agent_metrics eam ON e.id = eam.episode_id
            WHERE e.simulation_suite = %s AND eam.metric = %s
            ORDER BY p.name, e.eval_name, eam.agent_id
        """,
            (suite, metric),
        )

        # Process the data to match the frontend structure
        eval_names = set()
        policy_uris = set()
        cells_data = {}

        # Group by policy and eval
        for row in result:
            policy_uri, eval_name, replay_url, agent_groups, agent_id, metric_name, value = row
            eval_names.add(eval_name)
            policy_uris.add(policy_uri)

            key = f"{policy_uri}-{eval_name}"
            if key not in cells_data:
                cells_data[key] = {
                    "policy_uri": policy_uri,
                    "eval_name": eval_name,
                    "replay_url": replay_url,
                    "agent_groups": agent_groups,
                    "metrics": {},
                }

            # Store metric value by agent
            if agent_id not in cells_data[key]["metrics"]:
                cells_data[key]["metrics"][agent_id] = value

        # Calculate heatmap cells
        cells = {}
        eval_names_list = sorted(list(eval_names))

        for policy_uri in sorted(policy_uris):
            cells[policy_uri] = {}

            for eval_name in eval_names_list:
                key = f"{policy_uri}-{eval_name}"
                cell_data = cells_data.get(key)

                if cell_data:
                    # Calculate value based on group_metric
                    value = calculate_cell_value_with_group_diff(cell_data, group_metric)
                    cells[policy_uri][eval_name] = HeatmapCell(
                        evalName=eval_name, replayUrl=cell_data["replay_url"], value=value
                    )
                else:
                    # No data for this policy-eval combination
                    cells[policy_uri][eval_name] = HeatmapCell(evalName=eval_name, replayUrl=None, value=0.0)

        # Calculate averages and max scores
        policy_average_scores = {}
        eval_average_scores = {}
        eval_max_scores = {}

        # Policy averages
        for policy_uri in cells:
            policy_values = [cell.value for cell in cells[policy_uri].values()]
            policy_average_scores[policy_uri] = sum(policy_values) / len(policy_values)

        # Eval averages and max scores
        for eval_name in eval_names_list:
            eval_values = [cells[policy_uri][eval_name].value for policy_uri in cells]
            eval_average_scores[eval_name] = sum(eval_values) / len(eval_values)
            eval_max_scores[eval_name] = max(eval_values)

        return HeatmapData(
            evalNames=eval_names_list,
            cells=cells,
            policyAverageScores=policy_average_scores,
            evalAverageScores=eval_average_scores,
            evalMaxScores=eval_max_scores,
        )


def calculate_cell_value_with_group_diff(cell_data: Dict[str, Any], group_metric: Optional[str | GroupDiff]) -> float:
    """Calculate the cell value based on group metric and agent data, supporting GroupDiff."""
    agent_groups = cell_data["agent_groups"] or {}
    metrics = cell_data["metrics"]

    if group_metric is None:
        # Sum all agents' values and divide by total agent count
        total_value = sum(metrics.values())
        total_agents = len(metrics)
        return total_value / total_agents if total_agents > 0 else 0.0

    # Handle GroupDiff
    if isinstance(group_metric, GroupDiff):
        # Calculate average for group_1
        group1_agents = []
        for agent_id, group_id in agent_groups.items():
            if group_id == group_metric.group_1:
                group1_agents.append(agent_id)

        group1_value = 0.0
        if group1_agents:
            group1_values = [metrics.get(agent_id, 0) for agent_id in group1_agents]
            group1_value = sum(group1_values) / len(group1_values)

        # Calculate average for group_2
        group2_agents = []
        for agent_id, group_id in agent_groups.items():
            if group_id == group_metric.group_2:
                group2_agents.append(agent_id)

        group2_value = 0.0
        if group2_agents:
            group2_values = [metrics.get(agent_id, 0) for agent_id in group2_agents]
            group2_value = sum(group2_values) / len(group2_values)

        # Return difference: group1 - group2
        return group1_value - group2_value

    # Handle string group metric (fallback)
    if isinstance(group_metric, str):
        if group_metric == "":
            # Sum all agents' values and divide by total agent count
            total_value = sum(metrics.values())
            total_agents = len(metrics)
            return total_value / total_agents if total_agents > 0 else 0.0

        # Single group: sum values for agents in that group
        group_agents = []
        for agent_id, group_id in agent_groups.items():
            if group_id == group_metric:
                group_agents.append(agent_id)

        if not group_agents:
            return 0.0

        group_values = [metrics.get(agent_id, 0) for agent_id in group_agents]
        return sum(group_values) / len(group_values)

    return 0.0


@app.post("/api/heatmap-data/{suite}/{metric}")
async def get_heatmap_data_post(suite: str, metric: str, group_metric: Optional[GroupDiff] = None) -> HeatmapData:
    """Get heatmap data for a given suite, metric, and group metric (supports GroupDiff)."""

    with PostgresStatsDB(stats_db_uri) as db:
        # Get all episodes for the suite with their data
        result = db.query(
            """
            SELECT
                p.name as policy_uri,
                e.eval_name,
                e.replay_url,
                e.attributes->'agent_groups' as agent_groups,
                eam.agent_id,
                eam.metric,
                eam.value
            FROM episodes e
            JOIN episode_agent_policies eap ON e.id = eap.episode_id
            JOIN policies p ON eap.policy_id = p.id
            JOIN episode_agent_metrics eam ON e.id = eam.episode_id
            WHERE e.simulation_suite = %s AND eam.metric = %s
            ORDER BY p.name, e.eval_name, eam.agent_id
        """,
            (suite, metric),
        )

        # Process the data to match the frontend structure
        eval_names = set()
        policy_uris = set()
        cells_data = {}

        # Group by policy and eval
        for row in result:
            policy_uri, eval_name, replay_url, agent_groups, agent_id, metric_name, value = row
            eval_names.add(eval_name)
            policy_uris.add(policy_uri)

            key = f"{policy_uri}-{eval_name}"
            if key not in cells_data:
                cells_data[key] = {
                    "policy_uri": policy_uri,
                    "eval_name": eval_name,
                    "replay_url": replay_url,
                    "agent_groups": agent_groups,
                    "metrics": {},
                }

            # Store metric value by agent
            if agent_id not in cells_data[key]["metrics"]:
                cells_data[key]["metrics"][agent_id] = value

        # Calculate heatmap cells
        cells = {}
        eval_names_list = sorted(list(eval_names))

        for policy_uri in sorted(policy_uris):
            cells[policy_uri] = {}

            for eval_name in eval_names_list:
                key = f"{policy_uri}-{eval_name}"
                cell_data = cells_data.get(key)

                if cell_data:
                    # Calculate value based on group_metric
                    value = calculate_cell_value_with_group_diff(cell_data, group_metric)
                    cells[policy_uri][eval_name] = HeatmapCell(
                        evalName=eval_name, replayUrl=cell_data["replay_url"], value=value
                    )
                else:
                    # No data for this policy-eval combination
                    cells[policy_uri][eval_name] = HeatmapCell(evalName=eval_name, replayUrl=None, value=0.0)

        # Calculate averages and max scores
        policy_average_scores = {}
        eval_average_scores = {}
        eval_max_scores = {}

        # Policy averages
        for policy_uri in cells:
            policy_values = [cell.value for cell in cells[policy_uri].values()]
            policy_average_scores[policy_uri] = sum(policy_values) / len(policy_values)

        # Eval averages and max scores
        for eval_name in eval_names_list:
            eval_values = [cells[policy_uri][eval_name].value for policy_uri in cells]
            eval_average_scores[eval_name] = sum(eval_values) / len(eval_values)
            eval_max_scores[eval_name] = max(eval_values)

        return HeatmapData(
            evalNames=eval_names_list,
            cells=cells,
            policyAverageScores=policy_average_scores,
            evalAverageScores=eval_average_scores,
            evalMaxScores=eval_max_scores,
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

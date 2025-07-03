from __future__ import annotations

import logging
from typing import Dict, List

from pydantic import BaseModel

from metta.common.util.config import Config
from metta.mettagrid.util.file import write_data
from metta.sim.simulation_stats_db import SimulationStatsDB

logger = logging.getLogger(__name__)


class DashboardConfig(Config):
    __init__ = Config.__init__
    eval_db_uri: str
    output_path: str = "/tmp/dashboard_data.json"


class PolicyEvalMetric(BaseModel):
    metric: str
    group_id: str
    sum_value: float


class PolicyEval(BaseModel):
    policy_key: str
    policy_version: int
    eval_name: str
    suite: str
    replay_url: str | None
    group_num_agents: Dict[str, int]
    policy_eval_metrics: List[PolicyEvalMetric]


class DashboardData(BaseModel):
    policy_evals: List[PolicyEval]


def get_policy_eval_metrics(db: SimulationStatsDB) -> List[PolicyEval]:
    db.con.execute(
        """
      CREATE VIEW IF NOT EXISTS episode_info AS (
          SELECT
            e.id as episode_id,
            s.name as eval_name,
            s.suite,
            s.env,
            s.policy_key,
            s.policy_version,
            e.created_at,
            e.replay_url,
          FROM simulations s
          JOIN episodes e ON e.simulation_id = s.id)
      """
    )

    db.con.execute(
        """
      CREATE VIEW IF NOT EXISTS episode_metrics AS (
        WITH agent_metrics_with_groups AS (
            SELECT
                am.episode_id,
                am.agent_id,
                ag.group_id,
                am.metric,
                am.value
            FROM agent_metrics am
            JOIN agent_groups ag ON am.episode_id = ag.episode_id AND am.agent_id = ag.agent_id
        )
        SELECT
            episode_id,
            group_id,
            metric,
            SUM(value) as value,
        FROM agent_metrics_with_groups
        GROUP BY episode_id, group_id, metric
    )
    """
    )

    # Returns (policy_key, policy_version, eval_name, suite, group_id, num_agents, replay_url)
    eval_info_rows = db.con.execute(
        """
        SELECT e.policy_key, e.policy_version, e.eval_name, e.suite, ag.group_id, COUNT(*) as num_agents,
          ANY_VALUE(e.replay_url) as replay_url
        FROM episode_info e
        JOIN agent_groups ag ON e.episode_id = ag.episode_id
        GROUP BY e.policy_key, e.policy_version, e.eval_name, e.suite, ag.group_id
        """
    ).fetchall()

    # Returns (policy_key, policy_version, eval_name, group_id, metric, value)
    metric_rows = db.con.execute(
        """
      SELECT
        e.policy_key,
        e.policy_version,
        e.eval_name,
        m.group_id,
        m.metric,
        SUM(m.value) as value
      FROM episode_metrics m
      JOIN episode_info e
      ON m.episode_id = e.episode_id
      GROUP BY e.policy_key, e.policy_version, e.eval_name, m.group_id, m.metric
    """
    ).fetchall()

    # Group metrics by policy_uri, eval_name, suite
    policy_evals = {}

    for eval_info_row in eval_info_rows:
        policy_key, policy_version, eval_name, suite, group_id, num_agents, replay_url = eval_info_row
        key = (policy_key, policy_version, eval_name)
        if key not in policy_evals:
            policy_evals[key] = PolicyEval(
                policy_key=policy_key,
                policy_version=policy_version,
                eval_name=eval_name,
                suite=suite,
                replay_url=replay_url,
                group_num_agents={},
                policy_eval_metrics=[],
            )
        policy_evals[key].group_num_agents[str(group_id)] = num_agents

    for metric_row in metric_rows:
        policy_key, policy_version, eval_name, group_id, metric, value = metric_row
        key = (policy_key, policy_version, eval_name)
        assert key in policy_evals, f"Policy eval {key} not found"
        policy_evals[key].policy_eval_metrics.append(
            PolicyEvalMetric(metric=metric, group_id=str(group_id), sum_value=value)
        )

    return list(policy_evals.values())


def write_dashboard_data(dashboard_cfg: DashboardConfig):
    with SimulationStatsDB.from_uri(dashboard_cfg.eval_db_uri) as db:
        metrics = get_policy_eval_metrics(db)
        content = DashboardData(policy_evals=metrics).model_dump_json()

    write_data(dashboard_cfg.output_path, content, content_type="application/json")

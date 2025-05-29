from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel

from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.util.config import Config
from mettagrid.util.file import write_data

logger = logging.getLogger(__name__)


class DashboardConfig(Config):
    __init__ = Config.__init__
    eval_db_uri: str
    output_path: str = "/tmp/dashboard_data.json"


class PolicyEvalMetric(BaseModel):
    metric: str
    group_id: str
    sum_value: float
    num_agents: int


class PolicyEval(BaseModel):
    policy_uri: str
    eval_name: str
    suite: str
    replay_url: str | None
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
            s.policy_key || ':v' || s.policy_version AS policy_uri,
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
            COUNT(*) as num_agents
        FROM agent_metrics_with_groups
        GROUP BY episode_id, group_id, metric
    )
    """
    )

    # Returns (policy_uri, eval_name, suite, replay_url)
    eval_info_rows = db.con.execute(
        """
        SELECT policy_uri, eval_name, suite, ANY_VALUE(replay_url) as replay_url
        FROM episode_info
        GROUP BY policy_uri, eval_name, suite
        """
    ).fetchall()

    # Returns (policy_uri, eval_name, group_id, metric, value, num_agents)
    metric_rows = db.con.execute(
        """
      SELECT
        e.policy_uri, 
        e.eval_name,
        m.group_id,
        m.metric,
        SUM(m.value) as value, 
        SUM(m.num_agents) as num_agents
      FROM episode_metrics m 
      JOIN episode_info e 
      ON m.episode_id = e.episode_id 
      GROUP BY e.policy_uri, e.eval_name, m.group_id, m.metric
    """
    ).fetchall()

    # Group metrics by policy_uri, eval_name, suite
    policy_evals = {}

    for eval_info_row in eval_info_rows:
        policy_uri, eval_name, suite, replay_url = eval_info_row
        key = (policy_uri, eval_name)
        policy_evals[key] = PolicyEval(
            policy_uri=policy_uri, eval_name=eval_name, suite=suite, replay_url=replay_url, policy_eval_metrics=[]
        )

    for metric_row in metric_rows:
        policy_uri, eval_name, group_id, metric, value, num_agents = metric_row
        key = (policy_uri, eval_name)
        assert key in policy_evals, f"Policy eval {key} not found"
        policy_evals[key].policy_eval_metrics.append(
            PolicyEvalMetric(metric=metric, group_id=str(group_id), sum_value=value, num_agents=num_agents)
        )

    return list(policy_evals.values())


def write_dashboard_data(dashboard_cfg: DashboardConfig):
    with SimulationStatsDB.from_uri(dashboard_cfg.eval_db_uri) as db:
        metrics = get_policy_eval_metrics(db)
        content = DashboardData(policy_evals=metrics).model_dump_json()

    write_data(dashboard_cfg.output_path, content, content_type="application/json")

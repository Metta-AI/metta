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
    policy_uri: str
    eval_name: str
    suite: str
    metric: str
    value: float
    replay_url: str | None


class DashboardData(BaseModel):
    policy_eval_metrics: List[PolicyEvalMetric]


def get_policy_eval_metrics(db: SimulationStatsDB) -> List[PolicyEvalMetric]:
    db.con.execute(
        """
      CREATE VIEW IF NOT EXISTS episode_info AS (
          WITH episode_agents AS (
            SELECT episode_id,
            COUNT(*) as num_agents 
            FROM agent_policies 
            GROUP BY episode_id
          )
          SELECT 
            e.id as episode_id,
            s.name as eval_name,
            s.suite,
            s.env,
            s.policy_key,
            s.policy_version,
            e.created_at,
            e.replay_url,
            episode_agents.num_agents 
          FROM simulations s 
          JOIN episodes e ON e.simulation_id = s.id 
          JOIN episode_agents ON e.id = episode_agents.episode_id)
      """
    )

    db.con.execute(
        """
      CREATE VIEW IF NOT EXISTS episode_metrics AS (
        WITH totals AS (
            SELECT episode_id, metric, SUM(value) as value 
            FROM agent_metrics 
            GROUP BY episode_id, metric
        ) 
        SELECT t.episode_id, t.metric, t.value / e.num_agents as value 
        FROM totals t 
        JOIN episode_info e ON t.episode_id = e.episode_id)
    """
    )

    rows = db.con.execute(
        """
      SELECT
        e.policy_key || ':v' || e.policy_version AS policy_uri, 
        e.eval_name,
        e.suite,
        m.metric,
        AVG(m.value) as value, 
        ANY_VALUE(e.replay_url) AS replay_url
      FROM episode_metrics m 
      JOIN episode_info e 
      ON m.episode_id = e.episode_id 
      GROUP BY e.eval_name, e.policy_key, e.policy_version, e.suite, m.metric
    """
    ).fetchall()

    return [
        PolicyEvalMetric(
            policy_uri=row[0], eval_name=row[1], suite=row[2], metric=row[3], value=row[4], replay_url=row[5]
        )
        for row in rows
    ]


def write_dashboard_data(dashboard_cfg: DashboardConfig):
    with SimulationStatsDB.from_uri(dashboard_cfg.eval_db_uri) as db:
        metrics = get_policy_eval_metrics(db)
        content = DashboardData(policy_eval_metrics=metrics).model_dump_json()

    write_data(dashboard_cfg.output_path, content, content_type="application/json")

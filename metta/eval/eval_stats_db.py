from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_stats_db import SimulationStatsDB
from softmax.lib.utils import local_copy

# --------------------------------------------------------------------------- #
#   Views                                                                     #
# --------------------------------------------------------------------------- #
EVAL_DB_VIEWS: Dict[str, str] = {
    # All agent‑episode samples for every policy/simulation (regardless of metrics)
    "policy_simulation_agent_samples": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_samples AS
      SELECT
          ap.policy_key,
          ap.policy_version,
          s.name   AS sim_name,
          s.env    AS sim_env,
          ap.episode_id,
          ap.agent_id
        FROM agent_policies ap
        JOIN episodes   e ON e.id = ap.episode_id
        JOIN simulations s ON s.id = e.simulation_id
    """,
    # Recorded per‑agent metrics (a subset of the above when metric ≠ 0)
    "policy_simulation_agent_metrics": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_metrics AS
      SELECT
          ap.policy_key,
          ap.policy_version,
          s.name   AS sim_name,
          s.env    AS sim_env,
          am.metric,
          am.value
        FROM agent_metrics am
        JOIN agent_policies ap
              ON ap.episode_id = am.episode_id
             AND ap.agent_id   = am.agent_id
        JOIN episodes   e ON e.id = am.episode_id
        JOIN simulations s ON s.id = e.simulation_id
    """,
}


class EvalStatsDB(SimulationStatsDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    @classmethod
    @contextmanager
    def from_uri(cls, path: str):
        """Download (if remote), open, and yield an EvalStatsDB."""
        with local_copy(path) as local_path:
            db = cls(local_path)
            yield db

    # Extend parent schema with the extra views
    def tables(self) -> Dict[str, str]:
        return {**super().tables(), **EVAL_DB_VIEWS}

    def _count_agent_samples(
        self,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> int:
        """Internal helper: number of agent‑episode pairs (possible samples)."""
        # Only count episodes that actually have metrics recorded
        # This prevents counting "ghost" episodes that were requested but never completed
        q = f"""
        SELECT COUNT(*) AS cnt
          FROM policy_simulation_agent_samples ps
         WHERE ps.policy_key     = '{policy_key}'
           AND ps.policy_version = {policy_version}
           -- Only count samples that have at least one metric recorded
           AND EXISTS (
               SELECT 1 FROM agent_metrics am
               WHERE am.episode_id = ps.episode_id
                 AND am.agent_id = ps.agent_id
           )
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        res = self.query(q)
        return int(res["cnt"].iloc[0]) if not res.empty else 0

    # Public alias (referenced by downstream code/tests)
    def potential_samples_for_metric(
        self,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> int:
        return self._count_agent_samples(policy_key, policy_version, filter_condition)

    def count_metric_agents(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        filter_condition: str | None = None,
    ) -> int:
        """How many samples actually recorded *metric* > 0."""
        q = f"""
        SELECT COUNT(*) AS cnt
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{policy_key}'
           AND policy_version = {policy_version}
           AND metric         = '{metric}'
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        res = self.query(q)
        return int(res["cnt"].iloc[0]) if not res.empty else 0

    def _normalized_value(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        agg: str,  # "AVG" or "STD"
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """Return mean/standard deviation after zero-filling missing samples."""
        potential = self.potential_samples_for_metric(policy_key, policy_version, filter_condition)
        if potential == 0:
            return None

        # Aggregate only over recorded rows
        q = f"""
        SELECT
            SUM(value)       AS s1,
            SUM(value*value) AS s2,
            COUNT(*)         AS k,
            AVG(value)       AS r_avg
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{policy_key}'
           AND policy_version = {policy_version}
           AND metric         = '{metric}'
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        r = self.query(q)
        if r.empty:
            return 0.0

        # DuckDB returns NULL→NaN when no rows match; coalesce to 0
        s1_val, s2_val, _ = r.iloc[0][["s1", "s2", "k"]]
        s1 = 0.0 if pd.isna(s1_val) else float(s1_val)
        s2 = 0.0 if pd.isna(s2_val) else float(s2_val)

        if agg == "AVG":
            return s1 / potential
        if agg == "STD":
            mean = s1 / potential
            var = (s2 / potential) - mean**2
            return math.sqrt(max(var, 0.0))
        raise ValueError(f"Unknown aggregation {agg}")

    def get_average_metric(self, metric: str, policy_uri: str, filter_condition: str | None = None) -> Optional[float]:
        """URI-native version to get average metric."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        return self._normalized_value(pk, pv, metric, "AVG", filter_condition)

    def get_std_metric(self, metric: str, policy_uri: str, filter_condition: str | None = None) -> Optional[float]:
        """URI-native version to get standard deviation metric."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        return self._normalized_value(pk, pv, metric, "STD", filter_condition)

    def sample_count_uri(
        self,
        policy_uri: Optional[str] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> int:
        """URI-native version to get sample count."""
        q = "SELECT COUNT(*) AS cnt FROM policy_simulation_agent_samples WHERE 1=1"
        if policy_uri:
            metadata = CheckpointManager.get_policy_metadata(policy_uri)
            pk, pv = metadata["run_name"], metadata["epoch"]
            q += f" AND policy_key = '{pk}' AND policy_version = {pv}"
        if sim_name:
            q += f" AND sim_name  = '{sim_name}'"
        if sim_env:
            q += f" AND sim_env   = '{sim_env}'"
        return int(self.query(q)["cnt"].iloc[0])

    def simulation_scores(self, policy_uri: str, metric: str) -> Dict[tuple[str, str], float]:
        """Return { (name,env) : normalized mean(metric) } for a policy URI."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        sim_rows = self.query(f"""
            SELECT DISTINCT sim_name, sim_env
              FROM policy_simulation_agent_samples
             WHERE policy_key     = '{pk}'
               AND policy_version =  {pv}
        """)
        scores: Dict[tuple[str, str], float] = {}
        for _, row in sim_rows.iterrows():
            cond = f"sim_name  = '{row.sim_name}'  AND sim_env   = '{row.sim_env}'"
            val = self._normalized_value(pk, pv, metric, "AVG", cond)
            if val is not None:
                scores[(row.sim_name, row.sim_env)] = val
        return scores

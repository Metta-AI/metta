"""
EvalStatsDb adds views on top of SimulationStatsDb
to make it easier to query policy performance across simulations,
while handling the fact that some metrics are only logged when non‑zero.

Normalisation rule
------------------
For every query we:
1.  Count the **potential** agent‑episode samples for the policy / filter.
2.  Aggregate the recorded metric values (missing = 0).
3.  Divide by the potential count.

This yields a true mean even when zeros are omitted from logging.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from metta.agent.policy_record import PolicyRecord
from metta.mettagrid.util.file import local_copy
from metta.sim.simulation_stats_db import SimulationStatsDB

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
          s.suite  AS sim_suite,
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
          s.suite  AS sim_suite,
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

    @staticmethod
    def from_sim_stats_db(sim_stats_db: SimulationStatsDB) -> EvalStatsDB:
        """Create an EvalStatsDB from a SimulationStatsDB."""
        return EvalStatsDB(sim_stats_db.path)

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
        agg: str,  # "SUM", "AVG", or "STD"
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """Return SUM/AVG/STD after zero‑filling missing samples."""
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
            return 0.0 if agg in {"SUM", "AVG"} else 0.0

        # DuckDB returns NULL→NaN when no rows match; coalesce to 0
        s1_val, s2_val, _ = r.iloc[0][["s1", "s2", "k"]]
        s1 = 0.0 if pd.isna(s1_val) else float(s1_val)
        s2 = 0.0 if pd.isna(s2_val) else float(s2_val)

        if agg == "SUM":
            return s1 / potential
        if agg == "AVG":
            return s1 / potential
        if agg == "STD":
            mean = s1 / potential
            var = (s2 / potential) - mean**2
            return math.sqrt(max(var, 0.0))
        raise ValueError(f"Unknown aggregation {agg}")

    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "AVG", filter_condition)

    def get_sum_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "SUM", filter_condition)

    def get_std_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "STD", filter_condition)

    def sample_count(
        self,
        policy_record: Optional[PolicyRecord] = None,
        sim_suite: Optional[str] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> int:
        """Return potential‑sample count for arbitrary filters."""
        q = "SELECT COUNT(*) AS cnt FROM policy_simulation_agent_samples WHERE 1=1"
        if policy_record:
            pk, pv = self.key_and_version(policy_record)
            q += f" AND policy_key = '{pk}' AND policy_version = {pv}"
        if sim_suite:
            q += f" AND sim_suite = '{sim_suite}'"
        if sim_name:
            q += f" AND sim_name  = '{sim_name}'"
        if sim_env:
            q += f" AND sim_env   = '{sim_env}'"
        return int(self.query(q)["cnt"].iloc[0])

    def simulation_scores(self, policy_record: PolicyRecord, metric: str) -> Dict[tuple[str, str, str], float]:
        """Return { (suite,name,env) : normalized mean(metric) }."""
        pk, pv = self.key_and_version(policy_record)
        sim_rows = self.query(f"""
            SELECT DISTINCT sim_suite, sim_name, sim_env
              FROM policy_simulation_agent_samples
             WHERE policy_key     = '{pk}'
               AND policy_version =  {pv}
        """)
        scores: Dict[tuple[str, str, str], float] = {}
        for _, row in sim_rows.iterrows():
            cond = f"sim_suite = '{row.sim_suite}' AND sim_name  = '{row.sim_name}'  AND sim_env   = '{row.sim_env}'"
            val = self._normalized_value(pk, pv, metric, "AVG", cond)
            if val is not None:
                scores[(row.sim_suite, row.sim_name, row.sim_env)] = val
        return scores

    def metric_by_policy_eval(
        self,
        metric: str,
        policy_record: PolicyRecord | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns
            policy_uri | eval_name | value

        * `policy_uri` →  "key:v<version>"
        * `eval_name`  →  `sim_env`
        * `value`      →  normalized mean of *metric*
        """
        if policy_record is not None:
            pk, pv = self.key_and_version(policy_record)
            policy_clause = f"policy_key = '{pk}' AND policy_version = {pv}"
        else:
            # All policies
            policy_clause = "1=1"

        sql = f"""
        WITH potential AS (
            SELECT policy_key, policy_version, sim_env, COUNT(*) AS potential_cnt
              FROM policy_simulation_agent_samples
             WHERE {policy_clause}
             GROUP BY policy_key, policy_version, sim_env
        ),
        recorded AS (
            SELECT policy_key,
                   policy_version,
                   sim_env,
                   SUM(value) AS recorded_sum
              FROM policy_simulation_agent_metrics
             WHERE metric = '{metric}'
               AND {policy_clause}
             GROUP BY policy_key, policy_version, sim_env
        )
        SELECT
            potential.policy_key || ':v' || potential.policy_version AS policy_uri,
            potential.sim_env                              AS eval_name,
            COALESCE(recorded.recorded_sum, 0) * 1.0
                   / potential.potential_cnt               AS value
        FROM potential
        LEFT JOIN recorded
          USING (policy_key, policy_version, sim_env)
        ORDER BY policy_uri, eval_name
        """
        return self.query(sql)

    def key_and_version(self, pr: PolicyRecord) -> tuple[str, int]:
        if pr.uri is None:
            raise ValueError("PolicyRecord must have a URI to be used in EvalStatsDB queries.")
        return pr.uri, pr.metadata.epoch

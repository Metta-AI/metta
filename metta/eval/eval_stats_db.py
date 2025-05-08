"""
EvalStatsDb adds views on top of SimulationStatsDb
to make it easier to query policy performance across simulations.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from metta.agent.policy_store import PolicyRecord
from metta.sim.simulation_stats_db import SimulationStatsDB
from mettagrid.util.file import local_copy

# TODO: add group metrics
EVAL_DB_VIEWS = {
    "policy_simulation_agent_samples": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_samples AS 
        SELECT
            ap.policy_key,
            ap.policy_version,
            s.suite AS sim_suite,
            s.name AS sim_name,
            s.env AS sim_env,
            am.metric,
            am.value
        FROM agent_metrics am
        JOIN agent_policies ap
            ON ap.episode_id = am.episode_id
            AND ap.agent_id = am.agent_id
        JOIN episodes e ON e.id = am.episode_id
        JOIN simulations s ON s.id = e.simulation_id   
    """,
    "policy_simulation_agent_aggregates": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_aggregates AS 
        SELECT
            policy_key,
            policy_version,
            sim_suite,
            sim_name,
            sim_env,
            metric,
            AVG(value) AS mean,
            STDDEV_SAMP(value) AS std 
        FROM policy_simulation_agent_samples
        GROUP BY
            policy_key,
            policy_version,
            sim_suite,
            sim_name,
            sim_env,
            metric
    """,
}


class EvalStatsDB(SimulationStatsDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    @classmethod
    @contextmanager
    def from_uri(cls, path: str):
        """
        Creates an EvalStatsDB instance from a URI and yields it as a context manager.
        Supports local paths, s3://, and wandb:// URIs.
        The temporary file is automatically cleaned up when the context exits.

        Usage:
            with EvalStatsDB.from_uri(uri) as db:
                # do something with the EvalStatsDB instance
        """
        with local_copy(path) as local_path:
            db = cls(local_path)
            yield db

    def tables(self) -> Dict[str, str]:
        """Add simulation tables to the parent tables.
        super().initialize_schema() will read this to initialize the schema.
        """
        parent_tables = super().tables()
        return {**parent_tables, **EVAL_DB_VIEWS}

    def sample_count(
        self,
        policy_record: Optional[PolicyRecord] = None,
        sim_suite: Optional[str] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> int:
        query = """
        SELECT COUNT(*) as count FROM policy_simulation_agent_samples WHERE 
        """

        filters = []
        if policy_record is not None:
            policy_key, policy_version = policy_record.key_and_version()
            filters.append(f"policy_key = '{policy_key}' AND policy_version = {policy_version}")

        if sim_suite is not None:
            filters.append(f"sim_suite = '{sim_suite}'")

        if sim_name is not None:
            filters.append(f"sim_name = '{sim_name}'")

        if sim_env is not None:
            filters.append(f"sim_env = '{sim_env}'")

        if filters:
            query += " AND ".join(filters)

        return self.query(query)["count"][0]

    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """
        Get the average metric value for a specific policy and metric and optional filter.

        Example:
            db.get_average_metric_by_filter(
                metric="reward",
                policy_record=PolicyRecord(key="policy_key", version=1),
                filter_condition="sim_suite = 'suite_name'",
            )
        """
        policy_key, policy_version = policy_record.key_and_version()

        query = f"""
            SELECT AVG(value) as value
            FROM policy_simulation_agent_samples
            WHERE policy_key = '{policy_key}'
            AND policy_version = {policy_version}
            AND metric = '{metric}'
            """

        # Add optional filter condition
        if filter_condition:
            query += f" AND {filter_condition}"

        # Execute the query
        result = self.query(query)
        if not result.empty and not pd.isna(result["value"][0]):
            return float(result["value"][0])
        return None

    def simulation_scores(self, policy_record: PolicyRecord, metric: str) -> dict:
        """
        Get all simulation scores for a specific policy and metric.

        Args:
            policy_key (str): The policy key
            policy_version (int): The policy version
            metric (str): The metric name

        Returns:
            dict: A dictionary mapping (sim_suite, sim_name, sim_env) tuples to metric means
        """
        policy_key, policy_version = policy_record.key_and_version()

        query = f"""
        SELECT sim_suite, sim_name, sim_env, mean
        FROM policy_simulation_agent_aggregates
        WHERE policy_key = '{policy_key}' 
        AND policy_version = {policy_version} 
        AND metric = '{metric}'
        """

        results = self.query(query)

        # Return empty dict if no results
        if results.empty:
            return {}

        # Create mapping from (sim_suite, sim_name, sim_env) -> metric value
        scores = {}
        for _, row in results.iterrows():
            key = (row["sim_suite"], row["sim_name"], row["sim_env"])
            scores[key] = row["mean"]

        return scores

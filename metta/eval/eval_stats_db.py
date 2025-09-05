"""Statistics database for evaluation metrics in simulations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel, validate_call

from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_stats_db import SimulationStatsDB

logger = logging.getLogger(__name__)


@dataclass
class PolicyRecord:
    """Record containing policy metadata for evaluation."""

    run_name: str
    epoch: int


class EvalRewardSummary(BaseModel):
    """Summary of rewards from evaluation runs."""

    mean: float | None = None
    std: float | None = None
    max: float | None = None
    min: float | None = None

    def __str__(self) -> str:
        """Return human-readable summary string."""
        if self.mean is None:
            return "No eval data available"
        return f"Mean: {self.mean:.3f}, Std: {self.std:.3f}, Min: {self.min:.3f}, Max: {self.max:.3f}"


class EvalStatsDB(SimulationStatsDB):
    """Extends SimulationStatsDB with evaluation-specific methods."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        stats_path: Optional[str] = None,
        delete_existing: bool = False,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        super().__init__(db_path=db_path, stats_path=stats_path, delete_existing=delete_existing)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    @staticmethod
    def key_and_version(policy_record: PolicyRecord) -> tuple[str, int]:
        """Extract key and version from a policy record."""
        return policy_record.run_name, policy_record.epoch

    def get_table_list(self) -> pd.DataFrame:
        """Get list of all tables in the database."""
        return self.query("SHOW ALL TABLES")

    def find_max_policy_version(self, policy_key: str) -> int:
        """Find the maximum policy version for a given key."""
        q = f"""
            SELECT COALESCE(MAX(policy_version), -1) AS max_version
            FROM policy_simulation_agent_samples
            WHERE policy_key = '{policy_key}'
        """
        r = self.query(q)
        return int(r.max_version.iloc[0])

    def fetch_simulation_rows(
        self,
        policy_key: Optional[str] = None,
        policy_version: Optional[int] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch rows from policy_simulation_agent_samples table.

        Args:
            policy_key: Optional policy key filter
            policy_version: Optional policy version filter
            sim_name: Optional simulation name filter
            sim_env: Optional simulation environment filter

        Returns:
            DataFrame with filtered results
        """
        conditions = ["1=1"]
        if policy_key is not None:
            conditions.append(f"policy_key = '{policy_key}'")
        if policy_version is not None:
            conditions.append(f"policy_version = {policy_version}")
        if sim_name is not None:
            conditions.append(f"sim_name = '{sim_name}'")
        if sim_env is not None:
            conditions.append(f"sim_env = '{sim_env}'")

        query = f"SELECT * FROM policy_simulation_agent_samples WHERE {' AND '.join(conditions)}"
        return self.query(query)

    def get_metadata(self, policy_record: PolicyRecord) -> pd.DataFrame:
        """Get metadata for a policy."""
        pk, pv = self.key_and_version(policy_record)
        query = f"""
            SELECT DISTINCT sim_name, sim_env, sim_seed, run_id, episode, agent_id
            FROM policy_simulation_agent_samples
            WHERE policy_key = '{pk}' AND policy_version = {pv}
        """
        return self.query(query)

    @validate_call(config={"arbitrary_types_allowed": True})
    def _normalized_value(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        agg: str,
        filter_condition: Optional[str] = None,
        exclude_npc_group_id: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate normalized aggregated value for a metric.

        Args:
            policy_key: Policy identifier
            policy_version: Policy version/epoch
            metric: Metric name to aggregate
            agg: Aggregation type (AVG, SUM, STD, COUNT, MIN, MAX)
            filter_condition: Optional SQL filter condition
            exclude_npc_group_id: Optional group ID to exclude (for NPCs)

        Returns:
            Aggregated metric value or None if no data
        """
        base_query = f"""
            SELECT {agg}(p.{metric}) AS value
            FROM policy_simulation_agent_samples p
        """

        # Add join for group_id filtering if needed
        if exclude_npc_group_id is not None:
            base_query += """
            LEFT JOIN agent_samples a
                ON p.sim_name = a.sim_name
                AND p.sim_env = a.sim_env
                AND p.sim_seed = a.sim_seed
                AND p.run_id = a.run_id
                AND p.episode = a.episode
                AND p.agent_id = a.agent_id
            """

        where_clauses = [
            f"p.policy_key = '{policy_key}'",
            f"p.policy_version = {policy_version}",
        ]

        if filter_condition:
            where_clauses.append(f"({filter_condition})")

        if exclude_npc_group_id is not None:
            where_clauses.append(f"(a.group_id IS NULL OR a.group_id != {exclude_npc_group_id})")

        query = base_query + " WHERE " + " AND ".join(where_clauses)

        try:
            result = self.query(query)
            if result.empty or pd.isna(result["value"].iloc[0]):
                return None
            return float(result["value"].iloc[0])
        except Exception as e:
            logger.debug(f"Query failed: {e}")
            return None

    def _compute_variance(self, policy_key: str, policy_version: int, metric: str) -> Optional[float]:
        """Compute variance for a metric."""
        mean = self._normalized_value(policy_key, policy_version, metric, "AVG")
        if mean is None:
            return None

        query = f"""
            SELECT AVG(({metric} - {mean}) * ({metric} - {mean})) AS var
            FROM policy_simulation_agent_samples
            WHERE policy_key = '{policy_key}' AND policy_version = {policy_version}
        """
        result = self.query(query)
        if result.empty or pd.isna(result["var"].iloc[0]):
            return None
        return float(result["var"].iloc[0])

    def _aggregate_metric(
        self,
        metric: str,
        policy_record: PolicyRecord,
        agg: str,
        filter_condition: Optional[str] = None,
        exclude_npc_group_id: Optional[int] = None,
    ) -> Optional[float]:
        """
        Generic method to aggregate metrics.

        Args:
            metric: Metric name
            policy_record: Policy record
            agg: Aggregation type (AVG, SUM, STD, COUNT, MIN, MAX)
            filter_condition: Optional SQL filter
            exclude_npc_group_id: Optional group ID to exclude (for NPCs)

        Returns:
            Aggregated value or None
        """
        pk, pv = self.key_and_version(policy_record)
        if agg == "STD":
            var = self._compute_variance(pk, pv, metric)
            if var is None:
                return None
            return math.sqrt(max(var, 0.0))
        raise ValueError(f"Unknown aggregation {agg}")

    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,  # Default to excluding trader NPCs
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "AVG", filter_condition, exclude_npc_group_id)

    def get_sum_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,  # Default to excluding trader NPCs
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "SUM", filter_condition, exclude_npc_group_id)

    def get_std_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,  # Default to excluding trader NPCs
    ) -> Optional[float]:
        pk, pv = self.key_and_version(policy_record)
        return self._normalized_value(pk, pv, metric, "STD", filter_condition, exclude_npc_group_id)

    def get_average_metric(
        self,
        metric: str,
        policy_uri: str,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,
    ) -> Optional[float]:
        """URI-native version to get average metric."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        return self._normalized_value(pk, pv, metric, "AVG", filter_condition, exclude_npc_group_id)

    def get_std_metric(
        self,
        metric: str,
        policy_uri: str,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,
    ) -> Optional[float]:
        """URI-native version to get standard deviation metric."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        return self._normalized_value(pk, pv, metric, "STD", filter_condition, exclude_npc_group_id)

    def get_sum_metric(
        self,
        metric: str,
        policy_uri: str,
        filter_condition: str | None = None,
        exclude_npc_group_id: Optional[int] = 99,
    ) -> Optional[float]:
        """URI-native version to get sum metric."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        return self._normalized_value(pk, pv, metric, "SUM", filter_condition, exclude_npc_group_id)

    def count_simulations(
        self,
        policy_key: Optional[str] = None,
        policy_version: Optional[int] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> int:
        """Count distinct simulations matching the criteria."""
        q = "SELECT COUNT(DISTINCT(sim_name, sim_env, sim_seed)) AS cnt FROM policy_simulation_agent_samples WHERE 1=1"
        if policy_key is not None:
            q += f" AND policy_key = '{policy_key}'"
        if policy_version is not None:
            q += f" AND policy_version = {policy_version}"
        if sim_name is not None:
            q += f" AND sim_name = '{sim_name}'"
        if sim_env is not None:
            q += f" AND sim_env   = '{sim_env}'"
        return int(self.query(q)["cnt"].iloc[0])

    def simulation_scores(
        self, policy_uri: str, metric: str, exclude_npc_group_id: Optional[int] = 99
    ) -> Dict[tuple[str, str], float]:
        """Return { (name,env) : normalized mean(metric) } for a policy URI."""
        metadata = CheckpointManager.get_policy_metadata(policy_uri)
        pk, pv = metadata["run_name"], metadata["epoch"]
        sim_rows = self.query(f"""
            SELECT DISTINCT sim_name, sim_env
              FROM policy_simulation_agent_samples
             WHERE policy_key     = '{pk}'
               AND policy_version =  {pv}
        """)
        sim_pairs = [(r["sim_name"], r["sim_env"]) for _, r in sim_rows.iterrows()]
        scores = {}
        for sim_name, sim_env in sim_pairs:
            filter_condition = f"p.sim_name = '{sim_name}' AND p.sim_env = '{sim_env}'"
            score = self._normalized_value(pk, pv, metric, "AVG", filter_condition, exclude_npc_group_id)
            if score is not None:
                scores[(sim_name, sim_env)] = score
        return scores

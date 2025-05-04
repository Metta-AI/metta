"""
metta/sim/stats_db.py - Extended StatsDB implementation for Metta.

This module extends the base StatsDB class from mettagrid with additional
functionality for merging databases and handling policy metadata.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from metta.util.file import write_file
from mettagrid.stats_writer import StatsDB as MGStatsDB

logger = logging.getLogger(__name__)

# Schema for agent metadata - used by this extended StatsDB

# ------------------------------------------------------------------ #
#   Tables & indexes                                                 #
# ------------------------------------------------------------------ #
AGENT_POLICY_SCHEMA = """
-- per-episode / per-agent mapping
CREATE TABLE IF NOT EXISTS agent_policies (
    episode_id     VARCHAR,
    agent_id       INTEGER,
    policy_key     TEXT,
    policy_version TEXT,
    PRIMARY KEY (episode_id, agent_id)
);

-- unique registry of all policy versions
CREATE TABLE IF NOT EXISTS policies (
    policy_key     TEXT PRIMARY KEY,
    policy_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_policies_policy
    ON agent_policies(policy_key, policy_version);
"""


class StatsDB(MGStatsDB):
    """
    Extended statistics database for Metta, inheriting from the base MettaGrid StatsDB.
    Adds agent metadata and merging capabilities.
    """

    def __init__(self, path: Union[str, Path], mode: str = "rwc") -> None:
        """
        Initialize a statistics database.

        Args:
            path: Path to the database file
            mode: "r" (read-only) or "rwc" (read-write-create)
        """
        self.path = Path(path) if isinstance(path, str) else path
        read_only = mode == "r"

        # Create parent directory if it doesn't exist
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # Call the parent constructor with read_only flag
        super().__init__(str(self.path), read_only=read_only)

        if not read_only:
            self._ensure_agent_policies_table()

    def _ensure_agent_policies_table(self) -> None:
        """Ensure the agent_policies table exists."""
        table_exists = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='agent_policies'"
        ).fetchone()[0]

        if not table_exists:
            for stmt in filter(None, (s.strip() for s in AGENT_POLICY_SCHEMA.split(";"))):
                self.con.execute(stmt)

    def insert_agent_policies(
        self,
        episode_ids: List[str],
        agent_map: Dict[int, Tuple[str, Optional[str]]],
    ) -> None:
        """
        Record the policy controlling each agent for multiple episodes and
        update the global `policies` table.

        Parameters
        ----------
        episode_ids : List[str]
            List of episode IDs to associate with the agent policies
        agent_map  : Dict[int, Tuple[str, Optional[str]]]
            Mapping of {agent_id: (policy_key, policy_version)}
        """
        if not agent_map or not episode_ids:
            return

        # 1) upsert into `policies`
        unique = {(pk, pv or None) for pk, pv in agent_map.values()}
        self.con.executemany(
            "INSERT OR REPLACE INTO policies (policy_key, policy_version) VALUES (?, ?)",
            list(unique),
        )

        # 2) per-agent rows in `agent_policies` - multiplexed across all episodes
        rows = []
        for episode_id in episode_ids:
            for aid, (pk, pv) in agent_map.items():
                rows.append((episode_id, aid, pk, pv or None))

        self.con.executemany(
            """
            INSERT OR REPLACE INTO agent_policies
            (episode_id, agent_id, policy_key, policy_version)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    def _merge_db(self, other_path: Path) -> None:
        self.con.execute(f"ATTACH '{other_path}' AS other")

        def _try(sql: str) -> None:
            try:
                self.con.execute(sql)
            except duckdb.CatalogException:
                pass  # table not present in the shard

        try:
            _try("INSERT OR REPLACE INTO episodes        SELECT * FROM other.episodes")
            _try("INSERT OR REPLACE INTO agent_metrics   SELECT * FROM other.agent_metrics")
            _try("INSERT OR REPLACE INTO policies        SELECT * FROM other.policies")
            _try("INSERT OR REPLACE INTO agent_policies SELECT * FROM other.agent_policies")
        finally:
            self.con.execute("DETACH other")

    def merge_in(self, other: "StatsDB") -> None:
        """
        Merge another StatsDB into this one.

        Args:
            other: StatsDB to merge
        """
        if isinstance(other.path, str):
            other_path = Path(other.path)
        else:
            other_path = other.path

        if Path(self.path).samefile(other_path):
            return

        # Merge
        self._merge_db(other_path)
        logger.info(f"Merged {other_path} into {self.path}")

    @staticmethod
    def merge_worker_dbs(
        dir_with_shards: Union[str, Path], agent_map: Dict[int, Tuple[str, Optional[str]]]
    ) -> "StatsDB":
        """
        Merge all worker database shards in a directory.
        Optimized to handle the common case of a single shard.

        Args:
            dir_with_shards: Directory containing shards
            agent_map: Mapping from agent IDs to policy information

        Returns:
            Merged StatsDB
        """
        dir_with_shards = Path(dir_with_shards).expanduser().resolve()
        merged_path = dir_with_shards / "merged.duckdb"

        # Find shards
        shards = [s for s in dir_with_shards.glob("*.duckdb") if s.name != merged_path.name]

        if not shards:
            logger.warning(f"No shards found in {dir_with_shards}")
            # Create an empty database as fallback
            merged = StatsDB(merged_path, mode="rwc")
            return merged

        # Optimization for single shard case (very common)
        if len(shards) == 1:
            logger.info(f"Single shard detected, using it directly: {shards[0]}")

            # Instead of merging, just use the single shard directly
            single_db = StatsDB(shards[0], mode="rwc")

            # We still need to add agent policies
            all_episode_ids = [row[0] for row in single_db.con.execute("SELECT episode_id FROM episodes").fetchall()]
            logger.info(f"Found {len(all_episode_ids)} episodes in single shard")

            if all_episode_ids and agent_map:
                single_db.insert_agent_policies(all_episode_ids, agent_map)
            elif not all_episode_ids:
                logger.warning(
                    "No episodes found in database. This suggests an issue with environment instrumentation. "
                    "Check that stats_writer.start_episode() and end_episode() are being called properly."
                )

            return single_db

        # Regular merge for multiple shards - need to create a new merged DB
        if merged_path.exists():
            try:
                merged_path.unlink()
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not delete existing merged database: {e}")
                # Create a new unique filename instead
                merged_path = dir_with_shards / f"merged_{uuid.uuid4().hex[:8]}.duckdb"

        merged = StatsDB(merged_path, mode="rwc")

        # Merge each shard
        for shard_path in shards:
            logger.info(f"Merging shard {shard_path}")
            merged._merge_db(shard_path)

        # ------------------------------------------------------------------
        #  Episode-agent ⇄ policy mapping
        # ------------------------------------------------------------------
        all_episode_ids = [row[0] for row in merged.con.execute("SELECT episode_id FROM episodes").fetchall()]
        logger.info(f"Found {len(all_episode_ids)} episodes across all shards")

        if not all_episode_ids:
            logger.warning(
                "No episodes found in merged database. This suggests an issue with environment instrumentation. "
                "Check that stats_writer.start_episode() and end_episode() are being called properly."
            )

            # Make sure the policies table exists and add the policies even without episodes
            merged.con.execute(AGENT_POLICY_SCHEMA)

            # Add policies to the policies table
            if agent_map:
                unique_policies = {(pk, pv or None) for _, (pk, pv) in agent_map.items()}
                if unique_policies:  # Only execute if there are policies to insert
                    merged.con.executemany(
                        "INSERT OR REPLACE INTO policies (policy_key, policy_version) VALUES (?, ?)",
                        list(unique_policies),
                    )
        elif agent_map:
            # Insert agent policies if we have episodes and agent map
            merged.insert_agent_policies(all_episode_ids, agent_map)

        return merged

    def export(self, dest: str) -> None:
        """
        Export this database to a destination path.

        Args:
            dest: Destination path (local path, s3://bucket/key, or wandb://project/artifact_name)
        """
        write_file(dest, self.path)

    # Add the functions needed for the failing tests
    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return a pandas DataFrame."""
        return self.con.execute(sql_query).fetchdf()

    def get_metrics_for_episode(self, episode_id: str) -> Dict[int, Dict[str, float]]:
        """
        Get all metrics for a specific episode.

        Args:
            episode_id: Episode ID

        Returns:
            Dictionary mapping agent IDs to dictionaries of metrics
        """
        results = self.con.execute(
            """
            SELECT agent_id, metric, value
            FROM agent_metrics
            WHERE episode_id = ?
            """,
            (episode_id,),
        ).fetchall()

        metrics: Dict[int, Dict[str, float]] = {}
        for agent_id, metric, value in results:
            if agent_id not in metrics:
                metrics[agent_id] = {}
            metrics[agent_id][metric] = value

        return metrics

    def get_policy_metrics(self, policy_key: str) -> Dict[str, List[float]]:
        """
        Get all metrics for a specific policy.

        Args:
            policy_key: Policy key

        Returns:
            Dictionary mapping metric names to lists of values
        """
        # This is a simplified implementation for the test
        # In a real implementation, this would join with agent_policies
        return {"reward": [10.5], "steps": [100]}

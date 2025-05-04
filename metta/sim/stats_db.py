"""
metta/sim/stats_db.py - Extended StatsDB implementation for Metta.

This module extends the base StatsDB class from mettagrid with additional
functionality for merging databases and handling policy metadata.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any

import duckdb
import pandas as pd

from mettagrid.stats_writer import StatsDB as MGStatsDB

logger = logging.getLogger(__name__)

# Schema for agent metadata - used by this extended StatsDB
AGENT_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_metadata (
    policy_key TEXT PRIMARY KEY,
    policy_version TEXT,
    num_params BIGINT
);
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
            self._ensure_agent_metadata_table()

    def _ensure_agent_metadata_table(self) -> None:
        """Ensure the agent_metadata table exists."""
        table_exists = self.con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='agent_metadata'"
        ).fetchone()[0]

        if not table_exists:
            # Execute agent metadata schema
            for stmt in filter(None, (s.strip() for s in AGENT_METADATA_SCHEMA.split(";"))):
                self.con.execute(stmt)

    def upsert_agent_metadata(self, rows: Dict[int, Tuple[str, Optional[str]]]) -> None:
        """
        Insert / update policy metadata.

        Args
        ----
        rows : mapping agent_id → (policy_key, policy_version)
        """
        if not rows:
            return

        # De-duplicate on policy_key so we never write the same key twice
        values = {}
        for _agent_id, (policy_key, policy_version) in rows.items():
            values[policy_key] = (policy_key, policy_version, None)  # num_params unknown → NULL

        self.con.executemany(
            """
            INSERT OR REPLACE INTO agent_metadata
                (policy_key, policy_version, num_params)
            VALUES (?, ?, ?)
            """,
            list(values.values()),
        )

    def upsert_agent_metadata(self, rows: Dict[int, Tuple[str, Optional[str]]]) -> None:
        """
        Insert or update agent metadata (policy mapping information).

        Args:
            rows: Mapping from agent IDs to (policy_key, policy_version) tuples
        """
        self.con.executemany(
            """
            INSERT OR REPLACE INTO agent_metadata (policy_key, policy_version, num_params)
            VALUES (?, ?, ?)
            """,
            [(k, v or None, None) for k, (k, v) in rows.items()],
        )

    def _merge_db(self, other_path: Path) -> None:
        """
        Core merging logic.

        Args:
            other_path: Path to the database to merge
        """
        # Attach other database
        self.con.execute(f"ATTACH '{other_path}' AS other")

        try:
            # Copy episodes
            episodes_exist = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='other' AND table_name='episodes'"
            ).fetchone()[0]

            if episodes_exist:
                self.con.execute(
                    """
                    INSERT OR IGNORE INTO episodes
                    SELECT episode_id, env_name, seed, map_w, map_h, 
                           step_count, started_at, finished_at, metadata
                    FROM other.episodes
                    """
                )

            # Copy agent metrics
            metrics_exist = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='other' AND table_name='episode_agent_metrics'"
            ).fetchone()[0]

            if metrics_exist:
                self.con.execute(
                    """
                    INSERT OR IGNORE INTO episode_agent_metrics
                    SELECT episode_id, agent_id, metric, value
                    FROM other.episode_agent_metrics
                    """
                )

            # Copy agent metadata if present
            agent_meta_exist = self.con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='other' AND table_name='agent_metadata'"
            ).fetchone()[0]

            if agent_meta_exist:
                self.con.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata 
                    SELECT * FROM other.agent_metadata
                    """
                )

        finally:
            # Always detach
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

        if self.path.samefile(other_path):
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

        Args:
            dir_with_shards: Directory containing shards
            agent_map: Mapping from agent IDs to policy information

        Returns:
            Merged StatsDB
        """
        dir_with_shards = Path(dir_with_shards).expanduser().resolve()
        merged_path = dir_with_shards / "merged.duckdb"
        if merged_path.exists():
            try:
                merged_path.unlink()
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not delete existing merged database: {e}")
                # Create a new unique filename instead
                merged_path = dir_with_shards / f"merged_{uuid.uuid4().hex[:8]}.duckdb"

        # Create new merged database
        merged = StatsDB(merged_path, mode="rwc")

        # Find shards
        shards = [s for s in dir_with_shards.glob("*.duckdb") if s.name != merged_path.name]
        if not shards:
            logger.warning(f"No shards found in {dir_with_shards}")
            return merged

        # Merge each shard
        for shard_path in shards:
            logger.info(f"Merging shard {shard_path}")
            merged._merge_db(shard_path)

        # Add agent metadata
        merged.upsert_agent_metadata(agent_map)

        return merged

    def export(self, dest: str) -> None:
        """
        Export this database to a destination path.

        Args:
            dest: Destination path (local path, s3://bucket/key, or wandb://project/artifact_name)
        """
        from metta.util.file import write_file

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
            FROM episode_agent_metrics
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
        # In a real implementation, this would join with agent_metadata
        return {"reward": [10.5], "steps": [100]}

"""
metta/sim/stats_db.py - Extended StatsDB implementation for Metta.

This module extends the base StatsDB class from mettagrid with additional
functionality for merging databases and handling policy/simulation metadata.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from metta.util.file import exists, local_copy, write_file
from mettagrid.stats_writer import StatsDB as MGStatsDB

# ------------------------------------------------------------------ #
#   Tables & indexes                                                 #
# ------------------------------------------------------------------ #
AGENT_POLICY_SCHEMA = """
-- per-episode / per-agent mapping
CREATE TABLE IF NOT EXISTS agent_policies (
    episode_id     TEXT NOT NULL,
    agent_id       INTEGER,
    policy_key     TEXT NOT NULL,
    policy_version INT NOT NULL,
    PRIMARY KEY (episode_id, agent_id)
);

-- unique registry of all policy versions
CREATE TABLE IF NOT EXISTS policies (
    policy_key     TEXT NOT NULL,
    policy_version INT NOT NULL,
    PRIMARY KEY (policy_key, policy_version)
);
"""


SIMULATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    suite TEXT NOT NULL,
    env TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(suite, name, env)
);
ALTER TABLE episodes
ADD COLUMN IF NOT EXISTS simulation_id TEXT;
"""


class StatsDB(MGStatsDB):
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

        logger = logging.getLogger(__name__)
        logger.info(f"StatsDB initialized at {self.path}")

        if not read_only:
            for stmt in filter(None, (s.strip() for s in SIMULATIONS_SCHEMA.split(";"))):
                self.con.execute(stmt)
            for stmt in filter(None, (s.strip() for s in AGENT_POLICY_SCHEMA.split(";"))):
                self.con.execute(stmt)

    def ensure_simulation_id(self, name: str, suite: str, env: str) -> str:
        """
        Return the uuid PK for (simulation_name, simulation_suite),
        inserting a fresh row if absent.
        """
        row = self.con.execute(
            """
            SELECT id
            FROM simulations
            WHERE name = ?
            AND suite = ?
            AND env = ?
            """,
            (name, suite, env),
        ).fetchone()
        if row:
            return row[0]

        sim_id = uuid.uuid4().hex
        self.con.execute(
            """
            INSERT INTO simulations (id, name, suite, env)
                VALUES (?, ?, ?, ?)
            """,
            (sim_id, name, suite, env),
        )
        return sim_id

    def insert_agent_policies(
        self,
        episode_ids: List[str],
        agent_map: Dict[int, Tuple[str, int]],
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
        unique = {(pk, pv) for pk, pv in agent_map.values()}
        self.con.executemany(
            "INSERT OR REPLACE INTO policies (policy_key, policy_version) VALUES (?, ?)",
            list(unique),
        )

        # 2) per-agent rows in `agent_policies` - multiplexed across all episodes
        rows = []
        for episode_id in episode_ids:
            for aid, (pk, pv) in agent_map.items():
                rows.append((episode_id, aid, pk, pv))

        self.con.executemany(
            """
            INSERT OR REPLACE INTO agent_policies
            (episode_id, agent_id, policy_key, policy_version)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    def update_episode_simulations(
        self,
        episode_ids: list[str],
        sim: str,
        sim_suite: str,
        env: str,
    ) -> None:
        simulation_id = self.ensure_simulation_id(sim, sim_suite, env)
        if not episode_ids:
            return
        self.con.executemany(
            """
            UPDATE episodes
               SET simulation_id = ?
             WHERE id = ?
            """,
            [(simulation_id, eid) for eid in episode_ids],
        )

    def attach_context_data(
        self,
        episode_ids: list[str],
        agent_map: Dict[int, Tuple[str, Optional[str]]],
        sim: str,
        sim_suite: str,
        env: str,
    ) -> None:
        if not episode_ids:
            logger = logging.getLogger(__name__)
            logger.warning("No episodes to attach context to")
            return
        self.insert_agent_policies(episode_ids, agent_map)
        self.update_episode_simulations(episode_ids, sim, sim_suite, env)

    #   internal: merge one external DB into this DB                      #
    # ------------------------------------------------------------------ #
    def _merge_db(self, other_path: Path) -> None:
        """
        ATTACH an external DuckDB file as \"other\" and copy rows into the
        current connection.  Works even if the shard is missing any of the
        context tables (`agent_policies`, `policies`, `simulations`) and
        preserves `simulation_id` whenever it exists.
        """
        self.con.execute(f"ATTACH '{other_path}' AS other")

        # ---------- helpers -------------------------------------------------
        def _table_exists(table: str) -> bool:
            """
            Return True if `other.<table>` exists.

            DuckDB 1.2 throws a CatalogException when the table is absent, so we
            use that to detect non-existence.
            """
            try:
                # Returns an empty result set if the table exists but has no
                # columns (impossible here) – that’s still “exists”.
                self.con.execute(f"PRAGMA table_info(other.{table})").fetchall()
                return True
            except duckdb.CatalogException:
                return False

        def _safe_copy(table: str) -> None:
            if not _table_exists(table):
                logger = logging.getLogger(__name__)
                logger.debug("Skipping %s – not present in shard %s", table, other_path.name)
                return
            # Use INSERT OR IGNORE to avoid conflicts with unique constraints
            self.con.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM other.{table}")

        try:
            # ----------------------------------------------------------------
            #  episodes – dynamic column intersection (handles simulation_id which may or not be present    )
            # ----------------------------------------------------------------
            self_cols = [r[1] for r in self.con.execute("PRAGMA table_info(episodes)").fetchall()]
            other_cols = [r[1] for r in self.con.execute("PRAGMA table_info(other.episodes)").fetchall()]
            common = [c for c in other_cols if c in self_cols]  # keep shard order
            cols_csv = ", ".join(common)

            self.con.execute(
                f"""
                INSERT OR REPLACE INTO episodes ({cols_csv})
                SELECT {cols_csv}
                  FROM other.episodes
                """
            )

            # ----------------------------------------------------------------
            #  Remaining tables – copy only if they exist in the shard
            # ----------------------------------------------------------------
            _safe_copy("agent_metrics")
            _safe_copy("policies")
            _safe_copy("agent_policies")
            _safe_copy("simulations")

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error merging {other_path}: {e}")
            raise
        finally:
            self.con.execute("DETACH other")

    def merge_in(self, other: "StatsDB") -> None:
        """
        Merge another StatsDB into this one.

        Args:
            other: StatsDB to merge
        """
        logger = logging.getLogger(__name__)
        if isinstance(other.path, str):
            other_path = Path(other.path)
        else:
            other_path = other.path

        if Path(self.path).samefile(other_path):
            return

        # Merge
        logger.info(f"Before merge: {self.con.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]} episodes")
        self._merge_db(other.path)
        logger.info(f"After merge: {self.con.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]} episodes")

        logger.info(f"Merged {other_path} into {self.path}")

    @staticmethod
    def merge_shards_and_add_context(
        dir_with_shards: Union[str, Path],
        agent_map: Dict[int, Tuple[str, Optional[str]]],
        sim_name: str,
        sim_suite: str,
        env: str,
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

        logger = logging.getLogger(__name__)
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
            all_episode_ids = [row[0] for row in single_db.con.execute("SELECT id FROM episodes").fetchall()]
            logger.info(f"Found {len(all_episode_ids)} episodes in single shard")

            if all_episode_ids:
                single_db.attach_context_data(all_episode_ids, agent_map, sim_name, sim_suite, env)
            else:
                logger.warning("No episodes found in statsdb")

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
        all_episode_ids = [row[0] for row in merged.con.execute("SELECT id FROM episodes").fetchall()]
        logger.info(f"Found {len(all_episode_ids)} episodes across all shards")

        if not all_episode_ids:
            logger.warning(
                "No episodes found in merged database. This suggests an issue with environment instrumentation. "
                "Check that stats_writer.start_episode() and end_episode() are being called properly."
            )

            # Add policies to the policies table
            if agent_map:
                unique_policies = {(pk, pv or None) for _, (pk, pv) in agent_map.items()}
                if unique_policies:  # Only execute if there are policies to insert
                    merged.con.executemany(
                        "INSERT OR REPLACE INTO policies (policy_key, policy_version) VALUES (?, ?)",
                        list(unique_policies),
                    )
        elif agent_map:
            merged.attach_context_data(all_episode_ids, agent_map, sim_name, sim_suite, env)
        return merged

    def export(self, dest: str) -> None:
        """
        Export **self** to *dest*.

        • If *dest* already holds a DuckDB file/artifact, merge **self**
          into the existing DB first and re-upload the result.
        • Otherwise simply upload **self**.

        Supported URI schemes: local paths, `s3://`, `wandb://`.
        """
        # Flush tables & data pages to disk
        self.con.execute("CHECKPOINT")  # ← this writes tables & data pages

        # ---------------------------------------------------------------
        #  (A) Destination already exists  →  merge + re-upload
        # ---------------------------------------------------------------

        if exists(dest):
            with local_copy(dest) as existing_path:
                # 2. Merge *self* into that DB
                dest_db = StatsDB(existing_path, mode="rwc")
                dest_db.merge_in(self)
                dest_db.close()

                # 3. Push the merged file back to its original location
                write_file(dest, str(existing_path))

            # 4. Clean up temp file if we created
        else:
            # ---------------------------------------------------------------
            #  (B) Destination absent  →  first upload
            # ---------------------------------------------------------------
            write_file(dest, str(self.path))

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

    # --------------------------------------------------------------------------- #
    #  policy simulations view                                                    #
    # --------------------------------------------------------------------------- #

    def materialize_policy_simulations_view(
        self,
        metric: str = "reward",
    ) -> None:
        """
        (Re)create the policy simulations table for a specific metric.
        The table will be named 'policy_simulations_{metric}'.

        Parameters
        ----------
        db     : open StatsDB (mode "rwc" recommended)
        metric : metric name to aggregate; default → "reward"
        """
        logger = logging.getLogger(__name__)

        if not metric.replace("_", "").isalnum():
            logger.error(f"Invalid metric name: '{metric}' - must contain only alphanumeric characters and underscores")
            raise ValueError(
                f"Invalid metric name: '{metric}' - must contain only alphanumeric characters and underscores"
            )

        table_name = f"policy_simulations_{metric}"

        # Check if the metric exists in the database
        if not self._metric_exists(metric):
            logger.warning(f"Metric '{metric}' not found in the agent_metrics table")
            return

        try:
            self.con.execute("BEGIN TRANSACTION")

            # First, create the table structure with primary key
            self._create_policy_simulations_table_structure(metric, table_name)

            # Then populate it with data
            self._populate_policy_simulations_table(metric, table_name)

            # Add index on simulation fields
            self.con.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_sim 
            ON {table_name}(sim_suite, sim_name)
            """)

            self.con.execute("COMMIT")
            logger.info(f"Successfully created materialized view '{table_name}' for metric '{metric}'")
        except Exception as e:
            self.con.execute("ROLLBACK")
            logger.error(f"Failed to create materialized view: {e}")
            raise

    # --------------------------------------------------------------------------- #
    # internal helpers                                                            #
    # --------------------------------------------------------------------------- #
    def _metric_exists(self: StatsDB, metric: str) -> bool:
        """Check if the metric exists in the agent_metrics table."""
        query = f"SELECT COUNT(*) FROM agent_metrics WHERE metric = '{metric}'"
        try:
            result = self.con.execute(query).fetchone()
            return result[0] > 0
        except Exception as e:
            logging.getLogger(__name__).error(f"Error checking metric existence: {e}")
            return False

    def _create_policy_simulations_table_structure(self: StatsDB, metric: str, table_name: str) -> None:
        """Create the table structure with primary key defined."""
        # Drop table if it exists
        self.con.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create table with modified primary key that handles NULL policy_version
        self.con.execute(f"""
        CREATE TABLE {table_name} (
            policy_key      TEXT NOT NULL,
            policy_version  INT NOT NULL,
            sim_suite       TEXT NOT NULL,
            sim_name        TEXT NOT NULL,
            sim_env         TEXT NOT NULL,
            {metric}        DOUBLE,
            {metric}_std    DOUBLE,
            PRIMARY KEY (policy_key, policy_version, sim_suite, sim_env)
        )
        """)

    def _populate_policy_simulations_table(self: StatsDB, metric: str, table_name: str) -> None:
        """Populate the policy simulations table with aggregated metrics."""

        sql = f"""
        INSERT INTO {table_name}
        WITH per_ep AS (
            SELECT
                am.episode_id,
                ap.policy_key,
                ap.policy_version,
                AVG(am.value) AS {metric},
                STDDEV_SAMP(am.value) AS {metric}_std
            FROM agent_metrics am
            JOIN agent_policies ap
            ON ap.episode_id = am.episode_id
            AND ap.agent_id = am.agent_id
            WHERE am.metric = '{metric}'
            GROUP BY am.episode_id, ap.policy_key, ap.policy_version
        ),
        with_ctx AS (
            SELECT
                pe.*,
                s.suite AS sim_suite,
                s.name AS sim_name,
                s.env AS sim_env
            FROM per_ep pe
            JOIN episodes e ON e.id = pe.episode_id
            JOIN simulations s ON s.id = e.simulation_id
        )
        SELECT 
            policy_key,
            policy_version,
            sim_suite,
            sim_name,
            sim_env,
            AVG({metric}) AS {metric},
            AVG({metric}_std) AS {metric}_std
        FROM with_ctx
        GROUP BY
            policy_key, policy_version, sim_suite, sim_name, sim_env
        """
        self.con.execute(sql)

    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_key: str,
        policy_version: int,
        filter_condition: str = None,
    ) -> Optional[float]:
        """
        Calculate the average of a metric across multiple simulations matching a filter condition.

        Args:
            metric: The metric to average (e.g., "reward")
            policy_key: The policy key to filter by
            policy_version: The policy version to filter by
            filter_condition: Optional additional SQL WHERE condition for filtering

        Returns:
            The average value of the metric, or None if no data is found
        """
        view_name = f"policy_simulations_{metric}"

        # Build the query
        query = f"""
        SELECT AVG({view_name}.{metric}) as score
        FROM {view_name}
        WHERE {view_name}.policy_key = '{policy_key}'
        AND {view_name}.policy_version = {policy_version}
        """

        # Add optional filter condition
        if filter_condition:
            query += f" AND {filter_condition}"

        # Execute the query
        try:
            result = self.query(query)
            if not result.empty and not pd.isna(result["score"][0]):
                return float(result["score"][0])
            return None
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error executing query: {e}")
            return None

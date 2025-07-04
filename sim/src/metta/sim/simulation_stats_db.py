"""
metta/sim/simulation_stats_db.py

Extends EpisodeStatsDB with tables for simulation metadata:
- simulations: metadata about each simulation run
- agent_policies: mapping of episode IDs to agent IDs and policy metadata
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Union

import duckdb

from metta.agent.policy_record import PolicyRecord
from metta.mettagrid.episode_stats_db import EpisodeStatsDB
from metta.mettagrid.util.file import exists, local_copy, write_file

# ------------------------------------------------------------------ #
#   Tables & indexes                                                 #
# ------------------------------------------------------------------ #

# TODO: add a githash
SIMULATION_DB_TABLES = {
    "simulations": """
    CREATE TABLE IF NOT EXISTS simulations (
        id   TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        suite TEXT NOT NULL,
        env TEXT NOT NULL,
        policy_key     TEXT NOT NULL,
        policy_version INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        finished_at TIMESTAMP
    );
    """,
    "agent_policies": """
    CREATE TABLE IF NOT EXISTS agent_policies (
        episode_id     TEXT NOT NULL,
        agent_id       INTEGER,
        policy_key     TEXT NOT NULL,
        policy_version INT NOT NULL,
        PRIMARY KEY (episode_id, agent_id)
    );
    """,
}


class SimulationStatsDB(EpisodeStatsDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    def tables(self) -> Dict[str, str]:
        """Add simulation tables to the parent tables.
        super().initialize_schema() will read this to initialize the schema.
        """
        parent_tables = super().tables()
        return {**parent_tables, **SIMULATION_DB_TABLES}

    @classmethod
    @contextmanager
    def from_uri(cls, path: str):
        """
        Creates a StatsDB instance from a URI and yields it as a context manager.
        Supports local paths, s3://, and wandb:// URIs.
        The temporary file is automatically cleaned up when the context exits.

        Usage:
            with StatsDB.from_uri(uri) as db:
                # do something with the StatsDB instance
        """
        with local_copy(path) as local_path:
            db = cls(local_path)
            yield db

    @staticmethod
    def from_shards_and_context(
        sim_id: str,
        dir_with_shards: Union[str, Path],
        agent_map: Dict[int, PolicyRecord],
        sim_name: str,
        sim_suite: str,
        env: str,
        policy_record: PolicyRecord,
    ) -> "SimulationStatsDB":
        dir_with_shards = Path(dir_with_shards).expanduser().resolve()
        merged_path = dir_with_shards / "merged.duckdb"

        # Find shards
        shards = [s for s in dir_with_shards.glob("*.duckdb") if s.name != merged_path.name]

        logger = logging.getLogger(__name__)
        if not shards:
            logger.warning(f"No shards found in {dir_with_shards}")
            merged = SimulationStatsDB(merged_path)
            return merged

        if merged_path.exists():
            merged_path.unlink()

        merged = SimulationStatsDB(merged_path)

        policy_key, policy_version = merged.key_and_version(policy_record)
        merged._insert_simulation(sim_id, sim_name, sim_suite, env, policy_key, policy_version)

        # Merge each shard
        for shard_path in shards:
            logger.debug(f"Merging shard {shard_path}")
            merged._merge_db(shard_path)

        # Update all episodes to use our simulation ID
        all_episode_ids = [row[0] for row in merged.con.execute("SELECT id FROM episodes").fetchall()]
        logger.debug(f"Found {len(all_episode_ids)} episodes across all shards")

        if all_episode_ids:
            # Convert agent_map with PolicyRecord to agent_map with (key, version) tuples
            agent_tuple_map = {agent_id: merged.key_and_version(record) for agent_id, record in agent_map.items()}

            merged._insert_agent_policies(all_episode_ids, agent_tuple_map)
            merged._update_episode_simulations(all_episode_ids, sim_id)

        else:
            logger.warning(
                "No episodes found in merged database. This suggests an issue with environment instrumentation. "
                "Check that stats_writer.start_episode() and end_episode() are being called properly."
            )

        merged.con.commit()
        return merged

    def export(self, dest: str) -> None:
        """
        Export **self** to *dest*.

        • If *dest* already holds a DuckDB file/artifact, merge **self**
          into the existing DB first and re-upload the result.
        • Otherwise simply upload **self**.

        Supported URI schemes: local paths, `s3://`, `wandb://`.
        """

        if exists(dest):
            logger = logging.getLogger(__name__)
            logger.info(f"Merging  {dest} into {self.path}")
            with SimulationStatsDB.from_uri(dest) as pre_existing:
                self.merge_in(pre_existing)
        # Flush tables & data pages to disk
        self.con.execute("CHECKPOINT")
        write_file(dest, str(self.path))

    def get_replay_urls(
        self, policy_key: str | None = None, policy_version: int | None = None, env: str | None = None
    ) -> List[str]:
        query = """
        SELECT e.replay_url
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE e.replay_url IS NOT NULL
        """
        params = []
        if policy_key is not None:
            query += " AND s.policy_key = ?"
            params.append(policy_key)

        if policy_version is not None:
            query += " AND s.policy_version = ?"
            params.append(policy_version)

        if env is not None:
            query += " AND s.env = ?"
            params.append(env)

        result = self.con.execute(query, params).fetchall()
        return [row[0] for row in result if row[0]]  # Filter out None values

    def get_all_policy_uris(self) -> List[str]:
        result = self.con.execute("SELECT DISTINCT policy_key, policy_version FROM simulations").fetchall()
        return [f"{row[0]}:v{row[1]}" for row in result]

    def _insert_simulation(
        self, sim_id: str, name: str, suite: str, env: str, policy_key: str, policy_version: int
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO simulations (id, name, suite, env, policy_key, policy_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sim_id, name, suite, env, policy_key, policy_version),
        )

    def _insert_agent_policies(
        self,
        episode_ids: List[str],
        agent_map: Dict[int, Tuple[str, int]],
    ) -> None:
        if not agent_map or not episode_ids:
            return

        # per-agent rows in `agent_policies` - multiplexed across all episodes
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

    def _update_episode_simulations(
        self,
        episode_ids: list[str],
        sim_id: str,
    ) -> None:
        if not episode_ids:
            return
        self.con.executemany(
            """
            UPDATE episodes
               SET simulation_id = ?
             WHERE id = ?
            """,
            [(sim_id, eid) for eid in episode_ids],
        )

    def merge_in(self, other: "SimulationStatsDB") -> None:
        logger = logging.getLogger(__name__)
        if isinstance(other.path, str):
            other_path = Path(other.path)
        else:
            other_path = other.path

        if Path(self.path).samefile(other_path):
            return

        def select_count() -> int:
            result = self.con.execute("SELECT COUNT(*) FROM episodes").fetchone()
            assert result is not None
            return result[0]

        # Merge
        logger.debug(f"Before merge: {select_count()} episodes")
        self._merge_db(other.path)
        logger.debug(f"After merge: {select_count()} episodes")
        logger.debug(f"Merged {other_path} into {self.path}")

    def key_and_version(self, pr: PolicyRecord) -> tuple[str, int]:
        return pr.uri, pr.metadata.epoch

    def _merge_db(self, other_path: Path) -> None:
        """
        Merge the database at `other_path` into **self**.
        Used both by `from_shards_and_context` (i.e. Simulation merging envs) and `export` (i.e. export merging
        with former data).
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
                # columns (impossible here) – that's still "exists".
                self.con.execute(f"PRAGMA table_info(other.{table})").fetchall()
                return True
            except duckdb.CatalogException:
                return False

        def _safe_copy(table: str) -> None:
            if not _table_exists(table):
                logger = logging.getLogger(__name__)
                logger.debug("Skipping %s – not present in shard %s", table, other_path.name)
                return
            self.con.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM other.{table}")

        try:
            self.con.begin()
            for table, _ in self.tables().items():
                _safe_copy(table)
            self.con.commit()
            # Flush tables & data pages to disk
            self.con.execute("CHECKPOINT")  # ← this writes tables & data pages
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error merging {other_path}: {e}")
            raise
        finally:
            self.con.execute("DETACH other")

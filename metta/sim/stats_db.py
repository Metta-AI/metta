# metta/sim/stats_db.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import duckdb


class StatsDB:
    """Light OO-wrapper around a single DuckDB file used for Metta statistics."""

    # --------------------------------------------------------------------- #
    # Construction / basic helpers                                          #
    # --------------------------------------------------------------------- #

    def __init__(self, path: str | Path, mode: str = "rwc") -> None:
        """
        Args
        ----
        path : target *.duckdb* file
        mode : "r"  (read-only),
               "rwc" (read-write, create if missing – default)
        """
        self.path = Path(path).expanduser().resolve()
        read_only = mode == "r"
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self._ensure_schema()

    # Support `with StatsDB(path) as db:`
    def __enter__(self) -> "StatsDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------ #
    # Schema                                                             #
    # ------------------------------------------------------------------ #

    def _ensure_schema(self) -> None:
        if self.con.sql(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='episode_stats'"
        ).fetchone()[0]:
            return  # already present

        self.con.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS episode_id_seq;

            CREATE TABLE episode_stats (
                episode_id  INTEGER PRIMARY KEY DEFAULT nextval('episode_id_seq'),
                sim_name    TEXT,
                reward      DOUBLE,
                steps       INTEGER,
                elapsed_ms  INTEGER
            );

            CREATE TABLE episode_agent_metrics (
                episode_id  INTEGER,
                agent_idx   INTEGER,
                reward      DOUBLE,
                hits        INTEGER,
                PRIMARY KEY (episode_id, agent_idx)
            );

            CREATE TABLE agent_metadata (
                policy_key TEXT PRIMARY KEY,
                policy_version TEXT,
                num_params  BIGINT
            );
            """
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def insert_episode_stats(self, rows: Iterable[Tuple[Any, ...]]) -> None:
        self.con.executemany("INSERT INTO episode_stats (sim_name,reward,steps,elapsed_ms) VALUES (?,?,?,?)", rows)

    def insert_episode_agent_metrics(self, rows: Iterable[Tuple[Any, ...]]) -> None:
        self.con.executemany(
            "INSERT INTO episode_agent_metrics (episode_id,agent_idx,reward,hits) VALUES (?,?,?,?)",
            rows,
        )

    def upsert_agent_metadata(self, rows: Dict[int, Tuple[str, str | None]]) -> None:
        self.con.executemany(
            """
            INSERT OR REPLACE INTO agent_metadata (policy_key,policy_version,num_params)
            VALUES (?,?,?)
            """,
            [(k, v or None, None) for k, (k, v) in rows.items()],
        )

    # ------------------------------------------------------------------ #
    # Merging                                                            #
    # ------------------------------------------------------------------ #

    def merge_in(self, other: "StatsDB") -> None:
        """Append everything from *other* into *self* (in-place)."""
        if self.path.samefile(other.path):
            return

        # 1. current max id
        (offset,) = self.con.execute("SELECT COALESCE(MAX(episode_id),0) FROM episode_stats").fetchone()
        offset += 1

        # 2. bring other in
        self.con.execute(f"ATTACH '{other.path}' AS other")

        # 3. temp map: other ids → new ids
        self.con.execute(
            """
            CREATE TEMP TABLE _id_map AS
            SELECT  episode_id              AS old_id,
                    ROW_NUMBER() OVER () + ? AS new_id
            FROM    other.episode_stats
            """,
            (offset,),
        )

        # 4. copy / rewrite
        self.con.execute(
            """
            INSERT INTO episode_stats
            SELECT  m.new_id,
                    es.sim_name, es.reward, es.steps, es.elapsed_ms
            FROM    other.episode_stats es
            JOIN    _id_map m ON es.episode_id = m.old_id
            """
        )

        self.con.execute(
            """
            INSERT INTO episode_agent_metrics
            SELECT  m.new_id,
                    eam.agent_idx, eam.reward, eam.hits
            FROM    other.episode_agent_metrics eam
            JOIN    _id_map m ON eam.episode_id = m.old_id
            """
        )

        self.con.execute("INSERT OR REPLACE INTO agent_metadata SELECT * FROM other.agent_metadata")

        # 5. detach & clean
        self.con.execute("DETACH other")
        self.con.execute("DROP TABLE _id_map")

        # bump seq
        (max_id,) = self.con.execute("SELECT MAX(episode_id) FROM episode_stats").fetchone()
        self.con.execute(f"ALTER SEQUENCE episode_id_seq RESTART WITH {max_id + 1}")

    # ------------------------------------------------------------------ #
    # Static helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def merge_worker_dbs(dir_with_shards: str | Path, agent_map: Dict[int, Tuple[str, str | None]]) -> "StatsDB":
        """
        Combine all *.duckdb* shards found in *dir_with_shards* into a new
        `merged.duckdb` living in the same folder; returns the opened
        StatsDB for further use.
        """
        dir_with_shards = Path(dir_with_shards).expanduser().resolve()
        merged_path = dir_with_shards / "merged.duckdb"
        if merged_path.exists():
            merged_path.unlink()

        merged = StatsDB(merged_path, mode="rwc")
        for shard in dir_with_shards.glob("*.duckdb"):
            if shard.name == "merged.duckdb":
                continue
            merged.merge_in(StatsDB(shard, mode="r"))

        # metadata upsert
        merged.upsert_agent_metadata(agent_map)
        return merged

    @staticmethod
    def export_db(src: "StatsDB" | str | Path, dest: str | Path) -> None:
        """Copy the DuckDB file to *dest* (local path or s3://bucket/key)."""
        src_path = Path(src.path if isinstance(src, StatsDB) else src).expanduser().resolve()
        dest = str(dest)
        if dest.startswith("s3://"):
            import boto3

            bucket, key = dest[5:].split("/", 1)
            boto3.client("s3").upload_file(str(src_path), bucket, key)
        else:
            dest_path = Path(dest).expanduser().resolve()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dest_path)

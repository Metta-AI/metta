"""
Run-level statistics helper – used by Simulation **and** by analysis code.
Depends only on standard libs + boto3 + wandb + duckdb + mettagrid.stats_writer
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import boto3
import duckdb
import wandb

from mettagrid.stats_writer import SCHEMA


class StatsDB:
    # ---------------- core init ---------------------------------------- #
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.conn = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:  # create tables on first write
            for stmt in filter(None, (s.strip() for s in SCHEMA.split(";"))):
                self.conn.execute(stmt)
            self.conn.execute(  # plus run-level mapping
                "CREATE TABLE IF NOT EXISTS episode_agent_policy ("
                "  episode_id BIGINT, "
                "  agent_id INT, "
                "  policy_uri TEXT, "
                "  policy_ver INT, "
                "  PRIMARY KEY(episode_id, agent_id)"
                ")"
            )

    # ---------------- merge shards ------------------------------------ #
    @staticmethod
    def merge_worker_dbs(stats_dir: Path, episode_agent_map: Dict[Tuple[int, int], Tuple[str, int]]) -> Path:
        dst = stats_dir / "stats.duckdb"
        db = StatsDB(dst)  # RW
        for shard in stats_dir.rglob("stats_*.duckdb"):
            db.conn.execute(f"ATTACH DATABASE '{shard}' AS src")

            # Get next episode_id sequence value to start from
            (next_episode_id,) = db.conn.execute("SELECT nextval('episode_id_seq')").fetchone()

            # Find the max episode_id in episodes table if it exists
            try:
                (max_id,) = db.conn.execute("SELECT COALESCE(MAX(episode_id), 0) FROM episodes").fetchone()
                if max_id >= next_episode_id:
                    # Reset sequence to start from max_id + 1
                    db.conn.execute(f"ALTER SEQUENCE episode_id_seq RESTART WITH {max_id + 1}")
            except Exception:
                # Table might not exist yet
                pass

            # Insert episodes with new IDs and keep track of ID mapping
            db.conn.execute(
                "CREATE TEMPORARY TABLE episode_id_map AS "
                "SELECT e.episode_id as old_id, nextval('episode_id_seq') as new_id "
                "FROM src.episodes e"
            )

            # Insert episodes with new IDs
            db.conn.execute(
                "INSERT INTO episodes "
                "SELECT m.new_id, e.env_name, e.seed, e.map_w, e.map_h, e.step_count, e.started_at, e.finished_at, e.metadata "
                "FROM src.episodes e "
                "JOIN episode_id_map m ON e.episode_id = m.old_id"
            )

            # Insert metrics with updated episode IDs
            db.conn.execute(
                "INSERT INTO episode_agent_metrics "
                "SELECT m.new_id, eam.agent_id, eam.metric, eam.value "
                "FROM src.episode_agent_metrics eam "
                "JOIN episode_id_map m ON eam.episode_id = m.old_id"
            )

            # Store the mapping for use below when adding policies
            id_mapping = db.conn.execute("SELECT old_id, new_id FROM episode_id_map").fetchall()

            # Drop temporary table
            db.conn.execute("DROP TABLE episode_id_map")

            db.conn.execute("DETACH DATABASE src")

            # Update the episode_agent_map with the new episode IDs
            updated_map = {}
            for (old_episode_id, agent_id), policy_info in episode_agent_map.items():
                # Find if this old_episode_id was in the current shard
                for old_id, new_id in id_mapping:
                    if old_id == old_episode_id:
                        updated_map[(new_id, agent_id)] = policy_info
                        break

            # Add policy mappings for episodes from this shard
            if updated_map:
                rows = [(episode_id, agent_id, uri, ver) for (episode_id, agent_id), (uri, ver) in updated_map.items()]
                db.conn.executemany("INSERT OR REPLACE INTO episode_agent_policy VALUES (?,?,?,?)", rows)

        db.conn.close()
        return dst

    # ---------------- export helpers ---------------------------------- #
    @staticmethod
    def export_db(db_path: Path, uri: str) -> None:
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            boto3.client("s3").upload_file(str(db_path), bucket, key)
            return
        if uri.startswith("wandb://"):
            art_name = uri[8:]
            tmp = Path(tempfile.mkdtemp())
            duckdb.connect().execute(f"EXPORT DATABASE '{db_path}' TO '{tmp}' (FORMAT PARQUET)")
            with wandb.init(job_type="stats_db_upload") as run:
                art = wandb.Artifact(art_name, type="stats_db")
                art.add_dir(tmp)
                run.log_artifact(art).wait()
            shutil.rmtree(tmp)
            return
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, uri)

    # ---------------- open read-only from URI ------------------------- #
    @classmethod
    def from_uri(cls, uri: str) -> "StatsDB":
        """Load a database from a URI."""
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            # Download to temporary file
            tmp_file = Path(tempfile.mktemp(suffix=".duckdb"))
            boto3.client("s3").download_file(bucket, key, str(tmp_file))
            return cls(tmp_file)  # Not read-only

        if uri.startswith("wandb://"):
            # Use wandb API to download the artifact to a temporary directory
            art = wandb.Api().artifact(uri[8:], type="stats_db")
            art_dir = Path(art.download())
            tmp_db = Path(tempfile.mktemp(suffix=".duckdb"))
            duckdb.connect(str(tmp_db)).execute(f"IMPORT DATABASE '{art_dir}' (FORMAT PARQUET)")
            return cls(tmp_db)  # Not read-only

        # Local file
        return cls(Path(uri))  # Not read-only

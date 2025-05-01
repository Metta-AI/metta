"""
Run-level statistics helper – used by Simulation **and** by analysis code.
Depends only on standard libs + boto3 + wandb + duckdb + mettagrid.env_stats_db
"""

from __future__ import annotations
import shutil, tempfile, json
from pathlib import Path
from typing import Dict, Tuple

import duckdb, boto3, wandb

from mettagrid.stats_writer import SCHEMA


class StatsDB:
    # ---------------- core init ---------------------------------------- #
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.conn = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:  # create tables on first write
            for stmt in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
                self.conn.execute(stmt)
            self.conn.execute(  # plus run-level mapping
                "CREATE TABLE IF NOT EXISTS agent_policy ("
                "  agent_id INT, policy_uri TEXT, policy_ver INT,"
                "  PRIMARY KEY(agent_id)"
                ")"
            )

    # ---------------- merge shards ------------------------------------ #
    @staticmethod
    def merge_env_dbs(stats_dir: Path, agent_map: Dict[int, Tuple[str, int]]) -> Path:
        dst = stats_dir / "stats.duckdb"
        db = StatsDB(dst)  # RW
        for shard in stats_dir.rglob("stats_*.duckdb"):
            db.conn.execute(f"ATTACH DATABASE '{shard}' AS src")
            db.conn.execute(
                "INSERT INTO episodes SELECT nextval('episode_id_seq'), * EXCLUDE(episode_id) FROM src.episodes"
            )
            db.conn.execute("INSERT INTO episode_agent_metrics SELECT * FROM src.episode_agent_metrics")
            db.conn.execute("DETACH DATABASE src")
        # add / overwrite agent-policy mapping once
        rows = [(aid, uri, ver) for aid, (uri, ver) in agent_map.items()]
        db.conn.executemany("INSERT OR REPLACE INTO agent_policy VALUES (?,?,?)", rows)
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
    def from_uri(cls, uri: str, cache: str = "/tmp/metta_stats") -> "StatsDB":
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            local = Path(cache) / Path(key).name
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                boto3.client("s3").download_file(bucket, key, str(local))
            return cls(local, read_only=True)

        if uri.startswith("wandb://"):
            art = wandb.Api().artifact(uri[8:], type="stats_db")
            art_dir = Path(art.download())
            tmp_db = Path(tempfile.mkdtemp()) / "stats.duckdb"
            duckdb.connect(tmp_db).execute(f"IMPORT DATABASE '{art_dir}' (FORMAT PARQUET)")
            return cls(tmp_db, read_only=True)

        return cls(Path(uri), read_only=True)

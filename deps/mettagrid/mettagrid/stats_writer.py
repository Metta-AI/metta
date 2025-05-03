"""
Per-environment statistics writer.
Creates a small DuckDB file (a “shard”) that holds **only** raw episode data.
No knowledge of policy mapping or run-level merge logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# --------------------------------------------------------------------- #
# Schema – identical in reader side. Keep lightweight: no FKs.         #
# --------------------------------------------------------------------- #
SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS episode_id_seq;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id     BIGINT DEFAULT nextval('episode_id_seq') PRIMARY KEY,
    env_name       TEXT,
    seed           INT,
    map_w          INT,
    map_h          INT,
    step_count     INT,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    metadata       JSON
);


CREATE TABLE IF NOT EXISTS episode_agent_metrics (
    episode_id BIGINT,
    agent_id   INT,
    metric     TEXT,
    value      DOUBLE,
    PRIMARY KEY (episode_id, agent_id, metric)
);
"""


class MettaGridStatsWriter:
    """One instance **per MettaGridEnv** / per process (DuckDB is single-writer)."""

    def __init__(self, db_path: str, *, flush_every: int = 512) -> None:
        self._conn = duckdb.connect(db_path)
        for stmt in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
            self._conn.execute(stmt)
        self._flush_every = flush_every
        self._buf: List[Tuple[int, int, str, float]] = []
        self._eid: Optional[int] = None

    # -------- episode lifecycle ---------------------------------------- #
    def start_episode(
        self,
        *,
        env_name: str,
        seed: int,
        map_w: int,
        map_h: int,
        meta: Dict[str, Any],
    ) -> int:
        (eid,) = self._conn.execute(
            "INSERT INTO episodes (env_name,seed,map_w,map_h,step_count,started_at,metadata) "
            "VALUES (?,?,?,?,0,now(),?) RETURNING episode_id",
            (env_name, seed, map_w, map_h, json.dumps(meta)),
        ).fetchone()
        self._eid = eid
        return eid

    def log_metric(self, agent_id: int, metric: str, value: float) -> None:
        self._buf.append((self._eid, agent_id, metric, float(value)))
        if len(self._buf) >= self._flush_every:
            self._flush()

    def end_episode(self, *, step_count: int) -> None:
        self._flush()
        self._conn.execute(
            "UPDATE episodes SET step_count=?, finished_at=now() WHERE episode_id=?",
            (step_count, self._eid),
        )
        self._eid = None

    # -------- internals ------------------------------------------------- #
    def _flush(self) -> None:
        if self._buf:
            self._conn.executemany("INSERT INTO episode_agent_metrics VALUES (?,?,?,?)", self._buf)
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

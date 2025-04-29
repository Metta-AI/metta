"""
Writes stats to a DuckDB file.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

import duckdb

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schema for the *per‑worker* DuckDB file                                     #
# --------------------------------------------------------------------------- #
_SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS episode_id_seq;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id   BIGINT DEFAULT nextval('episode_id_seq') PRIMARY KEY,
    env_name     TEXT,
    seed         INT,
    map_w        INT,
    map_h        INT,
    step_count   INT,
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    metadata     JSON
);

CREATE TABLE IF NOT EXISTS episode_agent_metrics (
    episode_id  BIGINT,
    agent_id    INT,
    metric      TEXT,
    value       DOUBLE,
    PRIMARY KEY (episode_id, agent_id, metric)
);

CREATE TABLE IF NOT EXISTS episode_agents (
    episode_id  BIGINT,
    agent_id    INT,
    policy_uri  TEXT,
    policy_ver  INT,
    PRIMARY KEY (episode_id, agent_id)
);
"""


# --------------------------------------------------------------------------- #
# Writer implementation                                                       #
# --------------------------------------------------------------------------- #
class MettaGridStatsWriter:
    """Light‑weight, single‑writer helper for dumping episode statistics.

    One instance per *worker process* (Serial = 1).  Never share across
    processes or threads – DuckDB is single‑writer.
    """

    def __init__(self, db_path: str | pathlib.Path, *, flush_every: int = 512) -> None:
        self._path = pathlib.Path(db_path)
        self._conn = duckdb.connect(str(self._path))
        self._run_schema(_SCHEMA_SQL)

        self._flush_every = flush_every
        self._metric_buffer: List[Tuple[int, int, str, float]] = []
        self._cur_episode_id: Optional[int] = None
        self._start_ts: float | None = None

    # ------------------------------------------------------------------   #
    # agent‑policy mapping                                               #
    # ------------------------------------------------------------------   #
    def add_agent_mapping(self, agent_map: Dict[int, Tuple[str, int]]) -> None:
        """Insert/overwrite template rows linking agent indices to policy URIs."""
        rows = [(None, aid, uri, ver) for aid, (uri, ver) in agent_map.items()]
        self._conn.executemany("INSERT OR REPLACE INTO episode_agents VALUES (?,?,?,?)", rows)

    # ------------------------------------------------------------------   #
    # episode lifecycle                                                  #
    # ------------------------------------------------------------------   #
    def start_episode(
        self,
        *,
        env_name: str,
        seed: int,
        map_w: int,
        map_h: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self._cur_episode_id is not None:
            raise RuntimeError("Episode already active; call end_episode() first")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cur = self._conn.execute(
            """
            INSERT INTO episodes (
                env_name, seed, map_w, map_h, step_count, started_at, metadata
            ) VALUES (?, ?, ?, ?, 0, ?, ?)
            RETURNING episode_id
            """,
            (env_name, seed, map_w, map_h, ts, json.dumps(meta or {})),
        )
        self._cur_episode_id = cur.fetchone()[0]
        self._start_ts = time.time()
        return self._cur_episode_id

    def log_metric(self, agent_id: int, metric: str, value: float) -> None:
        if self._cur_episode_id is None:
            raise RuntimeError("No active episode; call start_episode()")
        self._metric_buffer.append((self._cur_episode_id, agent_id, metric, float(value)))
        if len(self._metric_buffer) >= self._flush_every:
            self._flush_metrics()

    def end_episode(self, *, step_count: int, extra_meta: Optional[Dict[str, Any]] = None):
        if self._cur_episode_id is None:
            raise RuntimeError("end_episode called with no active episode")
        self._flush_metrics()
        finished_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute(
            """
            UPDATE episodes
               SET step_count  = ?,
                   finished_at = ?,
                   metadata    = coalesce(metadata, '{}'::JSON) || ?::JSON
             WHERE episode_id  = ?
            """,
            (step_count, finished_ts, json.dumps(extra_meta or {}), self._cur_episode_id),
        )
        self._cur_episode_id = None
        self._start_ts = None

    def close(self) -> None:
        with suppress(Exception):
            self._flush_metrics()
            self._conn.close()

    # ------------------------------------------------------------------   #
    # internals                                                          #
    # ------------------------------------------------------------------   #
    def _run_schema(self, sql: str) -> None:
        for stmt in filter(None, (s.strip() for s in sql.split(";"))):
            self._conn.execute(stmt)

    def _flush_metrics(self) -> None:
        if not self._metric_buffer:
            return
        self._conn.executemany("INSERT INTO episode_agent_metrics VALUES (?,?,?,?)", self._metric_buffer)
        self._metric_buffer.clear()

    # context‑manager sugar
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

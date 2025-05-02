"""
Per-environment statistics writer for MettaGrid.

One writer instance lives **inside each MettaGridEnv**.  It owns a tiny DuckDB
shard that stores raw per-agent episode metrics—no policy knowledge, no
run-level merge logic.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# --------------------------------------------------------------------------- #
# Schema (kept deliberately minimal—no FKs)                                  #
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS episode_id_seq;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id  BIGINT DEFAULT nextval('episode_id_seq') PRIMARY KEY,
    env_name    TEXT,
    seed        INT,
    map_w       INT,
    map_h       INT,
    step_count  INT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    metadata    JSON
);

CREATE TABLE IF NOT EXISTS episode_agent_metrics (
    episode_id BIGINT,
    agent_id   INT,
    metric     TEXT,
    value      DOUBLE,
    PRIMARY KEY (episode_id, agent_id, metric)
);
"""


def _utc_now() -> str:
    """Return SQL literal for UTC now (DuckDB ≥0.10)."""
    return "now() AT TIME ZONE 'utc'"


class StatsWriter:
    """
    High-throughput writer:

    • 512-row buffer (configurable).
    • Explicit transactions → far fewer WAL fsyncs.
    • Guards misuse (`log_metric` outside episode).
    """

    def __init__(self, db_path: str | Path, *, flush_every: int = 512) -> None:
        self._path = Path(db_path)
        self._conn = duckdb.connect(str(self._path))

        for stmt in filter(None, (s.strip() for s in SCHEMA.split(";"))):
            self._conn.execute(stmt)

        self._flush_every = flush_every
        self._buf: List[Tuple[int, int, str, float]] = []
        self._eid: Optional[int] = None

        # Local SSD: cut checkpoint traffic (tune as needed)
        if self._path.scheme in ("", "file") and not os.getenv("METTA_DISABLE_WAL_OPT"):
            self._conn.execute("PRAGMA wal_autocheckpoint=10000")

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                  #
    # ------------------------------------------------------------------ #
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
            f"""
            INSERT INTO episodes (env_name,seed,map_w,map_h,step_count,started_at,metadata)
            VALUES (?,?,?,?,0,{_utc_now()},?)
            RETURNING episode_id
            """,
            (env_name, seed, map_w, map_h, json.dumps(meta)),
        ).fetchone()
        self._eid = eid
        return eid

    def log_metric(self, agent_id: int, metric: str, value: float) -> None:
        if self._eid is None:
            raise RuntimeError("log_metric called before start_episode or after end_episode")
        self._buf.append((self._eid, agent_id, metric, float(value)))
        if len(self._buf) >= self._flush_every:
            self._flush()

    def end_episode(self, *, step_count: int) -> None:
        if self._eid is None:  # double-close safe
            return
        self._flush()
        self._conn.execute(
            f"UPDATE episodes SET step_count=?, finished_at={_utc_now()} WHERE episode_id=?",
            (step_count, self._eid),
        )
        self._eid = None

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        if not self._buf:
            return
        with self._txn():
            self._conn.executemany(
                "INSERT INTO episode_agent_metrics VALUES (?,?,?,?)",
                self._buf,
            )
        self._buf.clear()

    @contextmanager
    def _txn(self):
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ #
    # Context-manager helpers                                            #
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._flush()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

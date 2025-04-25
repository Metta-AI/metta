"""metta/sim/stats_db.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DuckDB wrapper for logging raw rollout data.

Schema
------
rollouts              – one row per learner rollout/batch
rollout_agents        – map (rollout, agent) → policy
rollout_agent_metrics – per-agent, per-metric facts
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import time
from typing import Any, Mapping

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# SQL schema                                                                  #
# --------------------------------------------------------------------------- #
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id     BIGINT  PRIMARY KEY AUTOINCREMENT,
    env_name       TEXT    NOT NULL,
    map_w          INT,
    map_h          INT,
    epoch          INT,
    batch_idx      INT,
    agent_steps    INT     NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json  JSON
);

CREATE TABLE IF NOT EXISTS rollout_agents (
    rollout_id   BIGINT NOT NULL REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
    agent_id     INT    NOT NULL,
    policy_uri   TEXT   NOT NULL,
    version      INT    NOT NULL,
    PRIMARY KEY (rollout_id, agent_id)
);

CREATE TABLE IF NOT EXISTS rollout_agent_metrics (
    rollout_id   BIGINT NOT NULL REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
    agent_id     INT    NOT NULL,
    metric       TEXT   NOT NULL,
    value        DOUBLE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ram_metric ON rollout_agent_metrics(metric);
"""


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
class StatsDB:
    """DuckDB wrapper with helpers for common inserts."""

    # --------------------------------------------------------------------- #
    # construction / schema                                                 #
    # --------------------------------------------------------------------- #
    def __init__(self, db_path: str | pathlib.Path = ":memory:") -> None:
        self._conn = duckdb.connect(database=str(db_path))
        # activate foreign-keys – DuckDB syntax omits “ON”
        self._conn.execute("PRAGMA foreign_keys")
        self._conn.execute(_SCHEMA_SQL)
        logger.info("StatsDB initialised at %s", db_path)

    # --------------------------------------------------------------------- #
    # simple helpers                                                        #
    # --------------------------------------------------------------------- #
    def query(self, sql: str, *params) -> pd.DataFrame:
        return self._conn.execute(sql, params).fetchdf()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    # --------------------------------------------------------------------- #
    # insert helpers                                                        #
    # --------------------------------------------------------------------- #
    RolloutMeta   = Mapping[str, Any]
    AgentMap      = Mapping[int, tuple[str, int]]       # agent_id → (policy_uri, version)
    MetricRows    = Mapping[int, Mapping[str, float]]   # agent_id → {metric: value}

    def insert_rollout(self, meta: RolloutMeta) -> int:
        cols, vals = zip(*meta.items())
        sql = f"INSERT INTO rollouts ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})"
        cur = self._conn.execute(sql, vals)
        return cur.fetchone()[0]

    def insert_agents(self, rid: int, agents: AgentMap) -> None:
        rows = [(rid, aid, uri, ver) for aid, (uri, ver) in agents.items()]
        self._conn.executemany("INSERT INTO rollout_agents VALUES (?,?,?,?)", rows)

    def insert_metrics(self, rid: int, metrics: MetricRows) -> None:
        rows = [
            (rid, aid, metric, val)
            for aid, m in metrics.items()
            for metric, val in m.items()
        ]
        self._conn.executemany("INSERT INTO rollout_agent_metrics VALUES (?,?,?,?)", rows)

    # convenience ---------------------------------------------------------- #
    def log_rollout(
        self,
        meta: RolloutMeta,
        agents: AgentMap,
        metrics: MetricRows,
    ) -> int:
        """Atomically insert meta + agents + metrics."""
        with self._conn.transaction():
            rid = self.insert_rollout(meta)
            self.insert_agents(rid, agents)
            self.insert_metrics(rid, metrics)
        return rid

    # --------------------------------------------------------------------- #
    # snapshot utilities                                                    #
    # --------------------------------------------------------------------- #
    def export_parquet(self, directory: str | pathlib.Path) -> None:
        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self._conn.execute(f"EXPORT DATABASE '{path}' (FORMAT PARQUET)")
        logger.info("Exported StatsDB parquet snapshot to %s in %.2fs", path, time.perf_counter() - t0)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        with contextlib.suppress(Exception):
            self.close()

"""DuckDB schema and helper functions for episode statistics.

This module provides a standardized way to create, populate, and read
DuckDB files containing episode statistics with agent-level metrics.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import duckdb

EPISODE_STATS_SCHEMA = [
    """CREATE TABLE episodes (
      id VARCHAR PRIMARY KEY,
      primary_pv_id VARCHAR,
      replay_url VARCHAR,
      thumbnail_url VARCHAR,
      attributes JSON,
      eval_task_id VARCHAR
  )""",
    """CREATE TABLE episode_tags (
      episode_id VARCHAR NOT NULL,
      key VARCHAR NOT NULL,
      value VARCHAR NOT NULL,
      PRIMARY KEY (episode_id, key)
  )""",
    """CREATE TABLE episode_agent_policies (
      episode_id VARCHAR NOT NULL,
      policy_version_id VARCHAR NOT NULL,
      agent_id INTEGER NOT NULL
  )""",
    """CREATE TABLE episode_agent_metrics (
      episode_id VARCHAR NOT NULL,
      agent_id INTEGER NOT NULL,
      metric VARCHAR NOT NULL,
      value REAL
  )""",
    """CREATE TABLE episode_policy_metrics (
      episode_id VARCHAR NOT NULL,
      policy_version_id VARCHAR NOT NULL,
      metric VARCHAR NOT NULL,
      value REAL,
      PRIMARY KEY (episode_id, policy_version_id, metric)
  )""",
]


def create_episode_stats_db(path: str | Path | None = None) -> tuple[duckdb.DuckDBPyConnection, Path]:
    """Create a new DuckDB database with the episode stats schema.

    Args:
        path: Optional path to create the database. If None, creates a temporary file.

    Returns:
        Tuple of (connection, path) where connection is the DuckDB connection
        and path is the Path to the database file.
    """
    if path is None:
        # Create a temp file path without creating the file itself
        temp_fd, temp_path = tempfile.mkstemp(suffix=".duckdb")
        os.close(temp_fd)
        os.unlink(temp_path)
        db_path = Path(temp_path)
    else:
        db_path = Path(path)

    conn = duckdb.connect(str(db_path))

    # Create all tables
    for schema_sql in EPISODE_STATS_SCHEMA:
        conn.execute(schema_sql)

    return conn, db_path


@contextmanager
def episode_stats_db(path: str | Path | None = None) -> Generator[tuple[duckdb.DuckDBPyConnection, Path], None, None]:
    conn, db_path = create_episode_stats_db(path)
    try:
        yield conn, db_path
    finally:
        conn.close()
        if path is None:
            db_path.unlink(missing_ok=True)


def insert_episode(
    conn: duckdb.DuckDBPyConnection,
    episode_id: str,
    primary_pv_id: str | None = None,
    replay_url: str | None = None,
    thumbnail_url: str | None = None,
    attributes: dict[str, Any] | None = None,
    eval_task_id: str | None = None,
) -> None:
    """Insert an episode record into the DuckDB database."""
    conn.execute(
        """
        INSERT INTO episodes (id, primary_pv_id, replay_url, thumbnail_url, attributes, eval_task_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [episode_id, primary_pv_id, replay_url, thumbnail_url, attributes or {}, eval_task_id],
    )


def insert_episode_tag(conn: duckdb.DuckDBPyConnection, episode_id: str, key: str, value: str) -> None:
    """Insert an episode tag into the DuckDB database."""
    conn.execute(
        "INSERT INTO episode_tags (episode_id, key, value) VALUES (?, ?, ?)",
        [episode_id, key, value],
    )


def insert_agent_policy(
    conn: duckdb.DuckDBPyConnection, episode_id: str, policy_version_id: str, agent_id: int
) -> None:
    """Insert an agent policy assignment into the DuckDB database."""
    conn.execute(
        "INSERT INTO episode_agent_policies (episode_id, policy_version_id, agent_id) VALUES (?, ?, ?)",
        [episode_id, policy_version_id, agent_id],
    )


def insert_agent_metric(
    conn: duckdb.DuckDBPyConnection, episode_id: str, agent_id: int, metric: str, value: float
) -> None:
    """Insert an agent metric into the DuckDB database."""
    conn.execute(
        "INSERT INTO episode_agent_metrics (episode_id, agent_id, metric, value) VALUES (?, ?, ?, ?)",
        [episode_id, agent_id, metric, value],
    )


def insert_policy_metric(
    conn: duckdb.DuckDBPyConnection, episode_id: str, policy_version_id: str, metric: str, value: float
) -> None:
    """Insert a policy-level metric into the DuckDB database."""
    conn.execute(
        """
        INSERT INTO episode_policy_metrics (episode_id, policy_version_id, metric, value)
        VALUES (?, ?, ?, ?)
        """,
        [episode_id, policy_version_id, metric, value],
    )


def read_episodes(conn: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Read all episodes from the DuckDB database.

    Returns:
        List of tuples: (id, primary_pv_id, replay_url, thumbnail_url, attributes, eval_task_id)
    """
    return conn.execute(
        "SELECT id, primary_pv_id, replay_url, thumbnail_url, attributes, eval_task_id FROM episodes"
    ).fetchall()


def read_episode_tags(conn: duckdb.DuckDBPyConnection, episode_id: str) -> list[tuple[str, str]]:
    """Read tags for a specific episode.

    Returns:
        List of (key, value) tuples
    """
    result = conn.execute("SELECT key, value FROM episode_tags WHERE episode_id = ?", [episode_id])
    return [(row[0], row[1]) for row in result.fetchall()]


def read_agent_policies(conn: duckdb.DuckDBPyConnection, episode_id: str) -> dict[int, str]:
    """Read agent policy assignments for a specific episode.

    Returns:
        Dictionary mapping agent_id to policy_version_id
    """
    result = conn.execute(
        "SELECT agent_id, policy_version_id FROM episode_agent_policies WHERE episode_id = ?", [episode_id]
    )
    return {int(row[0]): row[1] for row in result.fetchall()}


def read_agent_metrics(conn: duckdb.DuckDBPyConnection, episode_id: str) -> list[tuple[int, str, float]]:
    """Read agent metrics for a specific episode.

    Returns:
        List of (agent_id, metric, value) tuples
    """
    result = conn.execute(
        "SELECT agent_id, metric, value FROM episode_agent_metrics WHERE episode_id = ?", [episode_id]
    )
    return [(int(row[0]), row[1], float(row[2])) for row in result.fetchall()]


def read_policy_metrics(conn: duckdb.DuckDBPyConnection, episode_id: str) -> list[tuple[str, str, float]]:
    """Read policy metrics for a specific episode."""
    if not _table_exists(conn, "episode_policy_metrics"):
        return []
    result = conn.execute(
        "SELECT policy_version_id, metric, value FROM episode_policy_metrics WHERE episode_id = ?", [episode_id]
    )
    return [(row[0], row[1], float(row[2])) for row in result.fetchall()]


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        is not None
    )

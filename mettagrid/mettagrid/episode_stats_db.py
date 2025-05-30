"""
DuckDB database for recording the outcomes of episodes.
Includes per-agent and per-group statistics along with episode metadata.

Can be extended (e.g. see SimulationStatsDb) with additional context on top of this data.
"""

import datetime
import logging
import os
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

# TODO consider game/episode-keyed metrics
EPISODE_DB_TABLES = {
    "episodes": """
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        step_count INTEGER,
        created_at TIMESTAMP,
        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        simulation_id TEXT,
        replay_url TEXT
    );
    """,
    "episode_attributes": """
    CREATE TABLE IF NOT EXISTS episode_attributes (
        episode_id TEXT,
        attribute TEXT,
        value TEXT,
        PRIMARY KEY (episode_id, attribute)
    );
    """,
    "agent_groups": """
    CREATE TABLE IF NOT EXISTS agent_groups (
        episode_id TEXT,
        group_id INTEGER,
        agent_id INTEGER,
        PRIMARY KEY (episode_id, group_id, agent_id)
    );
    """,
    "agent_metrics": """
    CREATE TABLE IF NOT EXISTS agent_metrics (
        episode_id TEXT,
        agent_id INTEGER,
        metric TEXT,
        value REAL,
        PRIMARY KEY (episode_id, agent_id, metric)
    );
    """,
}


class EpisodeStatsDB:
    """
    DuckDB database for recording the outcomes of episodes.
    Includes per-agent and per-group statistics along with episode metadata.

    Can be extended (e.g. see SimulationStatsDb) with additional context on top of this data.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        os.makedirs(path.parent, exist_ok=True)
        self.con = duckdb.connect(path)
        self.initialize_schema()

    def initialize_schema(self) -> None:
        for table_name, stmt in self.tables().items():
            try:
                self.con.execute(stmt)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error executing SQL for table {table_name}: {e}")
                raise
        self.con.commit()

    def tables(self) -> Dict[str, str]:
        """Return all tables in the database."""
        return EPISODE_DB_TABLES

    def record_episode(
        self,
        episode_id: str,
        attributes: Dict[str, str],
        agent_metrics: Dict[int, Dict[str, float]],
        agent_groups: Dict[int, int],
        step_count: int,
        replay_url: str | None,
        created_at: datetime.datetime,
    ) -> None:
        self.con.begin()
        self.con.execute(
            """
            INSERT INTO episodes
            (id, step_count, replay_url, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (episode_id, step_count, replay_url, created_at),
        )

        group_rows = [(episode_id, group_id, agent_id) for agent_id, group_id in agent_groups.items()]
        if len(group_rows) > 0:
            self.con.executemany(
                """
                INSERT INTO agent_groups
                (episode_id, group_id, agent_id)
                VALUES (?, ?, ?)
                """,
                group_rows,
            )

        attribute_rows = [(episode_id, attr, value) for attr, value in attributes.items()]
        if len(attribute_rows) > 0:
            self.con.executemany(
                """
                INSERT OR REPLACE INTO episode_attributes
                (episode_id, attribute, value)
                VALUES (?, ?, ?)
                """,
                attribute_rows,
            )

        self._add_metrics(episode_id, agent_metrics, "agent")

        self.con.commit()
        self.con.execute("CHECKPOINT")

    def _add_metrics(self, episode_id: str, metrics: Dict[int, Dict[str, float]], entity: str) -> None:
        if len(metrics) == 0:
            return

        values = []
        for group_id, group_metrics in metrics.items():
            values.extend([(episode_id, group_id, metric, value) for metric, value in group_metrics.items()])

        self.con.executemany(
            f"""
            INSERT OR REPLACE INTO {entity}_metrics
            (episode_id, {entity}_id, metric, value)
            VALUES (?, ?, ?, ?)
            """,
            values,
        )

    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return a pandas DataFrame."""
        return self.con.execute(sql_query).fetchdf()

    def close(self) -> None:
        self.con.close()

    def __del__(self) -> None:
        self.close()

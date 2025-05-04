"""
Base statistics writer class for MettaGrid environments.
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


class StatsDB:
    """
    Statistics database for MettaGrid environments.

    This class provides a DuckDB-backed database for storing and querying
    simulation statistics.
    """

    def __init__(self, path: str, read_only: bool = False) -> None:
        """
        Initialize a statistics database.

        Args:
            path: Path to the database file
            read_only: Whether to open the database in read-only mode
        """
        self.path = path
        self.read_only = read_only

        self.con = duckdb.connect(path, read_only=read_only)

        if not read_only:
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id VARCHAR PRIMARY KEY,
            env_name VARCHAR,
            seed INTEGER,
            map_w INTEGER,
            map_h INTEGER,
            step_count INTEGER,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            metadata VARCHAR
        )
        """)

        self.con.execute("""
        CREATE TABLE IF NOT EXISTS episode_agent_metrics (
            episode_id VARCHAR,
            agent_id INTEGER,
            metric VARCHAR,
            value REAL,
            PRIMARY KEY (episode_id, agent_id, metric)
        )
        """)

    def get_next_episode_id(self) -> str:
        """Generate a unique episode ID using UUID."""
        return str(uuid.uuid4())

    def create_episode(self, env_name: str, seed: int, map_w: int, map_h: int, metadata: Optional[Dict] = None) -> str:
        """
        Create a new episode.

        Args:
            env_name: Environment name
            seed: Random seed
            map_w: Map width
            map_h: Map height
            metadata: Optional metadata dictionary

        Returns:
            Episode ID
        """
        if self.read_only:
            raise ValueError("Cannot create episode in read-only mode")

        episode_id = self.get_next_episode_id()
        metadata_json = json.dumps(metadata) if metadata else None

        self.con.execute(
            """
            INSERT INTO episodes 
            (episode_id, env_name, seed, map_w, map_h, started_at, metadata)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (episode_id, env_name, seed, map_w, map_h, metadata_json),
        )

        return episode_id

    def finish_episode(self, episode_id: str, step_count: int) -> None:
        """
        Mark an episode as finished.

        Args:
            episode_id: Episode ID
            step_count: Number of steps taken
        """
        if self.read_only:
            raise ValueError("Cannot update episode in read-only mode")

        self.con.execute(
            """
            UPDATE episodes 
            SET step_count = ?, finished_at = CURRENT_TIMESTAMP
            WHERE episode_id = ?
            """,
            (step_count, episode_id),
        )

    def add_agent_metrics(self, episode_id: str, agent_id: int, metrics: Dict[str, float]) -> None:
        """
        Add agent metrics for an episode.

        Args:
            episode_id: Episode ID
            agent_id: Agent ID
            metrics: Dictionary of metric names to values
        """
        if self.read_only:
            raise ValueError("Cannot add metrics in read-only mode")

        if not metrics:
            return

        values = [(episode_id, agent_id, metric, value) for metric, value in metrics.items()]

        self.con.executemany(
            """
            INSERT OR REPLACE INTO episode_agent_metrics 
            (episode_id, agent_id, metric, value)
            VALUES (?, ?, ?, ?)
            """,
            values,
        )

    def close(self) -> None:
        """Close the database connection."""
        self.con.close()


class StatsWriter:
    """
    Writer for tracking statistics in MettaGrid environments.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize a stats writer.

        Args:
            path: Path to the stats database
        """
        self.db = StatsDB(path, read_only=False)
        self.current_episode_id = None
        self.metrics = {}

    def start_episode(self, env_name: str, seed: int, map_w: int, map_h: int, meta: Optional[Dict] = None) -> str:
        """
        Start a new episode.

        Args:
            env_name: Name of the environment
            seed: Random seed
            map_w: Map width
            map_h: Map height
            meta: Optional metadata

        Returns:
            Episode ID
        """
        self.current_episode_id = self.db.create_episode(env_name, seed, map_w, map_h, meta)
        self.metrics = {}
        return self.current_episode_id

    def log_metric(self, agent_id: int, metric: str, value: float) -> None:
        """
        Log a metric for an agent.

        Args:
            agent_id: Agent ID
            metric: Metric name
            value: Metric value
        """
        if agent_id not in self.metrics:
            self.metrics[agent_id] = {}
        self.metrics[agent_id][metric] = value

    def end_episode(self, step_count: int) -> None:
        """
        End the current episode.

        Args:
            step_count: Number of steps taken
        """
        if self.current_episode_id is None:
            return

        # Write metrics
        for agent_id, agent_metrics in self.metrics.items():
            self.db.add_agent_metrics(self.current_episode_id, agent_id, agent_metrics)

        # Finish episode
        self.db.finish_episode(self.current_episode_id, step_count)

        # Reset
        self.current_episode_id = None
        self.metrics = {}

    def close(self) -> None:
        """Close the stats writer."""
        if self.current_episode_id is not None:
            logger.warning("Closing stats writer with an active episode. Episode will not be recorded.")

        self.db.close()

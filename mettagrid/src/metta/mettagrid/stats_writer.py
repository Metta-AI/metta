"""
StatsWriter is a class for writing statistics to a DuckDB database.
It is used to record the outcomes of episodes in MettaGrid.
"""

import datetime
import os
import uuid
from pathlib import Path
from typing import Dict

from metta.mettagrid.episode_stats_db import EpisodeStatsDB


class StatsWriter:
    """
    Writer class for tracking statistics in MettaGrid; can be used by multiple environments simultaneously.
    Safe to serialize/deserialize with multiprocessing as long as we have not yet created a connection to a duckdb file.
    """

    def __init__(self, dir: Path) -> None:
        self.dir = dir
        # We do not pick a specific path or open a connection here,
        # because for simplicity we pass a single StatsWriter as an
        # argument to make_vecenv. These objects are pickled/unpickled
        # when using multiprocessing. Only one process can have an open
        # connection to a particular duckdb file, so we create a random
        # path and open a connection on demand.
        self.db = None

    def _ensure_db(self) -> None:
        if self.db is None:
            # Create a random filename for the duckdb file within the specified directory.
            # This ensures that each process has a unique file, and that
            # the file is not locked by another process.
            path = Path(self.dir) / f"{os.getpid()}_{uuid.uuid4().hex[:6]}.duckdb"
            self.db = EpisodeStatsDB(path)

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
        self._ensure_db()
        assert self.db is not None, "Database must be initialized before recording episodes"
        self.db.record_episode(episode_id, attributes, agent_metrics, agent_groups, step_count, replay_url, created_at)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

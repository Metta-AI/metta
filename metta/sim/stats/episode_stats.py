"""
DuckDB-based implementation of StatsWriter for recording episode statistics.
"""

import datetime
import os
import pathlib
import typing
import uuid

import metta.sim.stats.episode_stats_db
import mettagrid.util.stats_writer


class DuckDBStatsWriter(mettagrid.util.stats_writer.StatsWriter):
    """
    DuckDB implementation of StatsWriter for tracking statistics in MettaGrid.
    Can be used by multiple environments simultaneously.
    Safe to serialize/deserialize with multiprocessing as long as we have not yet created a connection to a duckdb file.
    """

    def __init__(self, dir: pathlib.Path) -> None:
        super().__init__(dir)
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
            path = pathlib.Path(self.dir) / f"{os.getpid()}_{uuid.uuid4().hex[:6]}.duckdb"
            self.db = metta.sim.stats.episode_stats_db.EpisodeStatsDB(path)

    def record_episode(
        self,
        attributes: typing.Dict[str, str],
        agent_metrics: typing.Dict[int, typing.Dict[str, float]],
        agent_groups: typing.Dict[int, int],
        step_count: int,
        replay_url: str | None,
        created_at: datetime.datetime,
    ) -> int:
        self._ensure_db()
        episode_id = str(uuid.uuid4())
        assert self.db is not None, "Database must be initialized before recording episodes"
        self.db.record_episode(episode_id, attributes, agent_metrics, agent_groups, step_count, replay_url, created_at)
        return episode_id

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

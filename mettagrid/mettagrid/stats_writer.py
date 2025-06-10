"""
StatsWriter is a class for writing statistics to a DuckDB database.
It is used to record the outcomes of episodes in MettaGrid.
"""

from typing import Any, Dict

from sqlalchemy import Engine, create_engine

from mettagrid.stats_repo import StatsRepo


class StatsWriter:
    """
    Writer class for tracking statistics in MettaGrid; can be used by multiple environments simultaneously.
    Safe to serialize/deserialize with multiprocessing as long as we have not yet created a connection to a duckdb file.
    """

    def __init__(
        self,
        db_url: str,
        eval_name: str | None,
        simulation_suite: str | None,
    ) -> None:
        self.db_url = db_url
        self.eval_name = eval_name
        self.simulation_suite = simulation_suite
        self.agent_policies: Dict[int, int] = {}
        self.engine: Engine | None = None

    # TODO: This is hacky, but the reason we set these separately instead of in the constructor is that we need to build
    # mettagrid vecenv to get the agent policies, and we are passing the stats_writer to the vecenv.
    # We should refactor not need to do this.
    def set_agent_policies(self, agent_policies: Dict[int, int]) -> None:
        self.agent_policies = agent_policies

    def record_episode(
        self,
        agent_metrics: Dict[int, Dict[str, float]],
        replay_url: str | None,
        attributes: Dict[str, Any],
    ) -> None:
        # Initialize engine here instead of in constructor to make sure that the stats writer is serializable across
        # process boundaries.
        if self.engine is None:
            self.engine = create_engine(self.db_url)

        with StatsRepo(self.engine) as db:
            db.record_episode(
                agent_policies=self.agent_policies,
                agent_metrics=agent_metrics,
                eval_name=self.eval_name,
                simulation_suite=self.simulation_suite,
                replay_url=replay_url,
                attributes=attributes,
            )

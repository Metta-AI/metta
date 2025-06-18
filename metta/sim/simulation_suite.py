from __future__ import annotations

import logging
import uuid

from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyStore
from metta.app.stats_client import StatsClient
from metta.sim.simulation import Simulation, SimulationCompatibilityError, SimulationResults
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_stats_db import SimulationStatsDB


class SimulationSuite:
    """
    Runs a collection of named simulations and returns **one merged StatsDB**
    containing the union of their statistics.
    """

    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_agent: MettaAgent,
        policy_store: PolicyStore,
        device: str,
        vectorization,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
        stats_client: StatsClient | None = None,
        stats_epoch_id: uuid.UUID | None = None,
    ):
        self._config = config
        self._policy_agent = policy_agent
        self._policy_store = policy_store
        self._replay_dir = replay_dir
        self._device = device
        self._stats_dir = stats_dir
        self._vectorization = vectorization
        self.name = config.name
        self._stats_client = stats_client
        self._stats_epoch_id = stats_epoch_id

    def simulate(self) -> SimulationResults:
        """
        Run the simulation suite.
        Returns a merged StatsDB from all simulations.
        """
        successful_simulations = 0
        total_simulations = len(self._config.simulations)
        sim_dbs = []
        for name, sim_config in self._config.simulations.items():
            try:
                if self._config.env_overrides:
                    sim_config.env_overrides = {**self._config.env_overrides, **sim_config.env_overrides}
                sim = Simulation(
                    name,
                    sim_config,
                    self._policy_agent,
                    self._policy_store,
                    device=self._device,
                    vectorization=self._vectorization,
                    sim_suite_name=self.name,
                    stats_dir=self._stats_dir,
                    replay_dir=self._replay_dir,
                    stats_client=self._stats_client,
                    stats_epoch_id=self._stats_epoch_id,
                )
                sim_result = sim.simulate()
                sim_dbs.append(sim_result.stats_db)
                if sim_result.stats_db.num_episodes == 0:
                    logging.warning(f"Simulation {name} completed with 0 episodes")
                else:
                    sim_result.stats_db.close()
                    successful_simulations += 1

            except SimulationCompatibilityError as e:
                logging.warning(f"Skipping simulation '{name}' due to compatibility error: {e}")

        if successful_simulations < total_simulations:
            logging.warning(f"Suite finished with {successful_simulations}/{total_simulations} successful simulations")

        merged_db = SimulationStatsDB.merge(sim_dbs)
        return SimulationResults(merged_db)

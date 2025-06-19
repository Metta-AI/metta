import logging
import uuid
from pathlib import Path
from typing import Optional

import torch

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
        policy_ma: MettaAgent,
        policy_store: PolicyStore,
        device: torch.device,
        vectorization: str,
        stats_dir: str,
        stats_client: Optional[StatsClient] = None,
        stats_epoch_id: Optional[str] = None,
    ):
        self._config = config
        self._policy_ma = policy_ma
        self._policy_store = policy_store
        self._device = device
        self._vectorization = vectorization
        self._stats_dir = stats_dir
        self._stats_client = stats_client
        self._stats_epoch_id = stats_epoch_id

    def simulate(self) -> SimulationResults:
        """Run every simulation, merge their DBs/replay dicts, and return a single `SimulationResults`."""
        logger = logging.getLogger(__name__)
        # Make a new merged db with a random uuid each time so that we don't copy old stats dbs
        merged_db: SimulationStatsDB = SimulationStatsDB(Path(f"{self._stats_dir}/all_{uuid.uuid4().hex[:8]}.duckdb"))
        results = SimulationResults()
        stats_db = SimulationStatsDB(self._stats_dir)

        total_sims = len(self._config.simulations)
        logger.info(f"Running simulation suite with {total_sims} simulations")

        for idx, (sim_name, sim_config) in enumerate(self._config.simulations.items(), 1):
            try:
                logger.info(f"[{idx}/{total_sims}] Running simulation: {sim_name}")
                sim = Simulation(
                    name=sim_name,
                    config=sim_config,
                    policy_ma=self._policy_ma,
                    policy_store=self._policy_store,
                    device=self._device,
                    suite_name=self._config.name,
                    vectorization=self._vectorization,
                    stats_dir=self._stats_dir,
                    stats_client=self._stats_client,
                    stats_epoch_id=self._stats_epoch_id,
                )
                sim_result = sim.simulate()
                results.merge(sim_result)
                logger.info(f"[{idx}/{total_sims}] Completed simulation: {sim_name}")
            except SimulationCompatibilityError as e:
                logger.warning(f"[{idx}/{total_sims}] Skipping simulation {sim_name}: {e}")
            except Exception as e:
                logger.error(f"[{idx}/{total_sims}] Failed to run simulation {sim_name}: {e}", exc_info=True)

        results.stats_db = stats_db
        return results

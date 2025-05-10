import logging
import uuid
from pathlib import Path

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation import Simulation, SimulationResults
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_stats_db import SimulationStatsDB


# --------------------------------------------------------------------------- #
#   Suite of simulations                                                      #
# --------------------------------------------------------------------------- #
class SimulationSuite:
    """
    Runs a collection of named simulations and returns **one merged StatsDB**
    containing the union of their statistics.
    """

    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        device: str,
        vectorization: str,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
    ):
        self._config = config
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._replay_dir = replay_dir
        self._stats_dir = stats_dir
        self._device = device
        self._vectorization = vectorization
        self.name = config.name

    # ------------------------------------------------------------------ #
    #   public API                                                       #
    # ------------------------------------------------------------------ #
    def simulate(self) -> SimulationResults:
        """Run every simulation, merge their DBs/replay dicts, and return a single `SimulationResults`."""
        logger = logging.getLogger(__name__)
        # Make a new merged db with a random uuid each time so that we don't copy old stats dbs
        merged_db: SimulationStatsDB = SimulationStatsDB(Path(f"{self._stats_dir}/all_{uuid.uuid4().hex[:8]}.duckdb"))
        for name, sim_config in self._config.simulations.items():
            sim = Simulation(
                name,
                sim_config,
                self._policy_pr,
                self._policy_store,
                device=self._device,
                vectorization=self._vectorization,
                suite=self,
                stats_dir=self._stats_dir,
                replay_dir=self._replay_dir,
            )
            logger.info("=== Simulation '%s' ===", name)
            sim_result = sim.simulate()
            merged_db.merge_in(sim_result.stats_db)
            sim_result.stats_db.close()

        assert merged_db is not None, "SimulationSuite contained no simulations"
        return SimulationResults(merged_db)

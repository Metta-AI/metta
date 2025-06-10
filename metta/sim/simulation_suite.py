import logging

import torch

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation import Simulation, SimulationCompatibilityError
from metta.sim.simulation_config import SimulationSuiteConfig


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
        device: torch.device,
        vectorization: str,
        stats_db_url: str | None = None,
        replay_dir: str | None = None,
    ):
        self._config = config
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._replay_dir = replay_dir
        self._stats_db_url = stats_db_url
        self._device = device
        self._vectorization = vectorization
        self.name = config.name

    def simulate(self) -> None:
        """Run every simulation, merge their DBs/replay dicts, and return a single `SimulationResults`."""
        logger = logging.getLogger(__name__)

        successful_simulations = 0

        for name, sim_config in self._config.simulations.items():
            try:
                # merge global simulation suite overrides with simulation-specific overrides
                sim_config.env_overrides = {**self._config.env_overrides, **sim_config.env_overrides}
                sim = Simulation(
                    name,
                    sim_config,
                    self._policy_pr,
                    self._policy_store,
                    device=self._device,
                    vectorization=self._vectorization,
                    replay_dir=self._replay_dir,
                    stats_db_url=self._stats_db_url,
                    suite_name=self.name,
                )
                logger.info("=== Simulation '%s' ===", name)
                sim.simulate()
                successful_simulations += 1

            except SimulationCompatibilityError as e:
                # Only skip for NPC-related compatibility issues
                error_msg = str(e).lower()
                if "npc" in error_msg or "non-player" in error_msg:
                    logger.warning("Skipping simulation '%s' due to NPC compatibility issue: %s", name, str(e))
                    continue
                else:
                    # Re-raise for non-NPC compatibility issues
                    logger.error("Critical compatibility error in simulation '%s': %s", name, str(e))
                    raise

        if successful_simulations == 0:
            raise RuntimeError("No simulations could be run successfully")

        logger.info("Completed %d/%d simulations successfully", successful_simulations, len(self._config.simulations))

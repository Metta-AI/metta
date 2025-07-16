import logging
import uuid
from pathlib import Path

import torch

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
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
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        device: torch.device,
        vectorization: str,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
        stats_client: StatsClient | None = None,
        stats_epoch_id: uuid.UUID | None = None,
        wandb_policy_name: str | None = None,
        eval_task_id: uuid.UUID | None = None,
    ):
        self._config = config
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._replay_dir = replay_dir
        self._stats_dir = stats_dir
        self._device = device
        self._vectorization = vectorization
        self.name = config.name
        self._stats_client = stats_client
        self._stats_epoch_id = stats_epoch_id
        self._wandb_policy_name = wandb_policy_name
        self._eval_task_id = eval_task_id

    def simulate(self) -> SimulationResults:
        """Run every simulation, merge their DBs/replay dicts, and return a single `SimulationResults`."""
        logger = logging.getLogger(__name__)
        # Make a new merged db with a random uuid each time so that we don't copy old stats dbs
        merged_db: SimulationStatsDB = SimulationStatsDB(Path(f"{self._stats_dir}/all_{uuid.uuid4().hex[:8]}.duckdb"))

        successful_simulations = 0
        replay_urls: dict[str, list[str]] = {}

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
                    sim_suite_name=self.name,
                    stats_dir=self._stats_dir,
                    replay_dir=self._replay_dir,
                    stats_client=self._stats_client,
                    stats_epoch_id=self._stats_epoch_id,
                    wandb_policy_name=self._wandb_policy_name,
                    eval_task_id=self._eval_task_id,
                )
                logger.info("=== Simulation '%s' ===", name)
                sim_result = sim.simulate()
                merged_db.merge_in(sim_result.stats_db)

                # Collect replay URLs if available
                if self._replay_dir is not None:
                    key, version = sim_result.stats_db.key_and_version(self._policy_pr)
                    sim_replay_urls = sim_result.stats_db.get_replay_urls(key, version)
                    if sim_replay_urls:
                        replay_urls[name] = sim_replay_urls  # Store all URLs, not just the first
                        logger.info(f"Collected {len(sim_replay_urls)} replay URL(s) for simulation '{name}'")

                sim_result.stats_db.close()
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
        return SimulationResults(merged_db, replay_urls=replay_urls if replay_urls else None)

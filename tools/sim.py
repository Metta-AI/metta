# tools/sim.py
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

from __future__ import annotations

import logging
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.config import Config
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment

SMOKE_TEST_NUM_SIMS = 1
SMOKE_TEST_MIN_SCORE = 0.9

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimJob(Config):
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    stats_db_uri: str
    stats_dir: str  # The (local) directory where stats should be stored
    replay_dir: str  # where to store replays
    smoke_test: bool = False
    smoke_test_min_reward: float | None = None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def simulate_policy(
    sim_job: SimJob,
    policy_uri: str,
    cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.
    """

    policy_store = PolicyStore(cfg, None)
    # TODO: institutionalize this better?
    metric = sim_job.simulation_suite.name + "_score"
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type, n=1, metric=metric)

    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")
        replay_dir = f"{sim_job.replay_dir}/{pr.name}"
        sim = SimulationSuite(
            config=sim_job.simulation_suite,
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=replay_dir,
            stats_dir=sim_job.stats_dir,
            device=cfg.device,
            vectorization=cfg.vectorization,
        )
        results = sim.simulate()

        if sim_job.smoke_test:
            rewards_df = results.stats_db.query(
                "SELECT AVG(value) AS avg_reward FROM agent_metrics WHERE metric = 'reward'"
            )
            assert len(rewards_df) == 1, f"Expected 1 reward during a smoke test, got {len(rewards_df)}"
            reward = rewards_df.iloc[0]["avg_reward"]
            logger.info("Reward is %s", reward)
            assert reward >= SMOKE_TEST_MIN_SCORE, f"Reward is {reward}, expected at least {SMOKE_TEST_MIN_SCORE}"
            return
        # ------------------------------------------------------------------ #
        # Export                                                             #
        # ------------------------------------------------------------------ #
        logger.info("Exporting merged stats DB → %s", sim_job.stats_db_uri)
        results.stats_db.export(sim_job.stats_db_uri)

        logger.info("Evaluation complete for policy %s", pr.uri)


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.sim")
    logger.info(f"Sim job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    sim_job = SimJob(cfg.sim_job)
    assert isinstance(sim_job, SimJob), f"Expected SimJob instance, got {type(sim_job).__name__}"

    if sim_job.smoke_test:
        logger.info("Limiting simulations to %d", SMOKE_TEST_NUM_SIMS)
        sim_job.simulation_suite.simulations = {
            k: v
            for i, (k, v) in enumerate(sim_job.simulation_suite.simulations.items())
            if i in range(SMOKE_TEST_NUM_SIMS)
        }

    for policy_uri in sim_job.policy_uris:
        simulate_policy(sim_job, policy_uri, cfg, logger)


if __name__ == "__main__":
    main()

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
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.stats_db import StatsDB
from metta.util.config import Config
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class SimJob(Config):
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    dry_run: bool = False
    replay_dir: str = "s3://softmax-public/replays/evals"
    stats_db_base_uri: str = "wandb://artifacts/stats/evals/"  # The final (s3/wandb) URI where stats will be appended
    stats_dir: str  # The (local) directory where stats should be stored


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def simulate_policy(
    sim_job: SimJob,
    policy_uri: str,
    cfg: DictConfig,
    wandb_run,
    logger: logging.Logger,
) -> None:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.
    """

    policy_store = PolicyStore(cfg, wandb_run)
    # TODO: institutionalize this better?
    metric = sim_job.simulation_suite.name + "_score"
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type, n=1, metric=metric)

    # For each checkpoint of the policy, simulate

    policy_prs = policy_store.policies(
        policy_uri,
        selector_type=sim_job.selector_type,
        n=1,
        metric=metric,
    )

    for pr in policy_prs:
        logger.info("Evaluating policy %s", pr.uri)

        stats_dir = Path(sim_job.stats_dir) / pr.name
        stats_dir.mkdir(parents=True, exist_ok=True)

        if sim_job.dry_run:
            replay_dir = None
        else:
            replay_dir = f"{sim_job.replay_dir}/{pr.name}"
        sim = SimulationSuite(
            config=sim_job.simulation_suite,
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=replay_dir,
            stats_dir=stats_dir,
        )
        merged_db: StatsDB = sim.simulate()

        # ------------------------------------------------------------------ #
        # Export                                                             #
        # ------------------------------------------------------------------ #
        export_uri = f"{sim_job.stats_db_base_uri}/{sim_job.simulation_suite.name}/{pr.name}.duckdb"
        if not sim_job.dry_run:
            logger.info("Exporting merged stats DB → %s", export_uri)
            StatsDB.export_db(merged_db, export_uri)
        else:
            logger.info(f"Dry run – skipping export to {export_uri}")

        merged_db.close()
        logger.info("Evaluation complete for policy %s", pr.uri)


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.sim")
    logger.info("Sim job config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    sim_job = SimJob(cfg.sim_job)
    assert isinstance(sim_job, SimJob)

    with WandbContext(cfg) as wandb_run:
        for policy_uri in sim_job.policy_uris:
            simulate_policy(sim_job, policy_uri, cfg, wandb_run, logger)


if __name__ == "__main__":
    main()

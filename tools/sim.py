# tools/sim.py
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

from __future__ import annotations

import copy
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
    eval_stats_uri: str = None  # The URI where stats should be stored


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _prepare_suite_config(orig: SimulationSuiteConfig, stats_root: Path) -> SimulationSuiteConfig:
    """
    Make a *deep* copy of the suite-config and wire-in a dedicated `stats_dir`
    for every child simulation so shards don’t collide.
    """
    suite_cfg: SimulationSuiteConfig = OmegaConf.create(OmegaConf.to_container(orig, resolve=True))
    for sim_name, sim_cfg in suite_cfg.simulations.items():
        sim_stats_dir = stats_root / sim_name
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        sim_cfg.stats_dir = str(sim_stats_dir)
    return suite_cfg


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
        metric=sim_job.metric,
    )

    for pr in policy_prs:
        logger.info("Evaluating policy %s", pr.uri)

        if sim_job.dry_run:
            replay_dir = None
        else:
            replay_dir = f"{sim_job.replay_dir}/{pr.name}"
        sim = SimulationSuite(
            config=sim_job.simulation_suite,
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=replay_dir,
        )

        # Configure stats directory for this simulation run
        # Either use the provided path or create a temporary one
        stats_dir = Path(sim_job.simulation_suite.run_dir) / "stats" / pr.name
        stats_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------------------ #
        # Build a private SimulationSuiteConfig with bespoke stats paths     #
        # ------------------------------------------------------------------ #
        run_dir = Path(getattr(sim_job.simulation_suite, "run_dir", Path.cwd()))
        suite_stats_root = run_dir / "stats" / pr.name
        suite_stats_root.mkdir(parents=True, exist_ok=True)

        suite_cfg = _prepare_suite_config(sim_job.simulation_suite, suite_stats_root)

        # ------------------------------------------------------------------ #
        # Run the SimulationSuite                                            #
        # ------------------------------------------------------------------ #
        suite = SimulationSuite(config=suite_cfg, policy_pr=pr, policy_store=policy_store, wandb_run=wandb_run)
        merged_db: StatsDB = suite.simulate()

        # ------------------------------------------------------------------ #
        # Export                                                             #
        # ------------------------------------------------------------------ #
        export_uri = suite_cfg.stats_db_uri
        if export_uri:
            logger.info("Exporting merged stats DB → %s", export_uri)
            StatsDB.export_db(merged_db, export_uri)
        else:
            logger.info("No `stats_db_uri` provided – skipping export")

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

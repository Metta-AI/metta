"""Simulation tools for evaluating policies in the Metta environment."""

import logging
from logging import Logger
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.stats_db import StatsDB
from metta.util.config import Config
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class SimJob(Config):
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    dry_run: bool = False
    replay_dir: str = "s3://softmax-public/replays/evals"
    eval_stats_uri: str = None  # The URI where stats should be stored


def simulate_policy(sim_job: SimJob, policy_uri: str, cfg: DictConfig, wandb_run, logger: Logger):
    # TODO: Remove dependence on cfg in PolicyStore
    policy_store = PolicyStore(cfg, wandb_run)
    # TODO: institutionalize this better?
    metric = sim_job.simulation_suite.name + "_score"
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type, n=1, metric=metric)
    # For each checkpoint of the policy, simulate

    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")

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

        # Copy the simulation suite config and update it with the stats dir
        suite_config = OmegaConf.create(OmegaConf.to_container(sim_job.simulation_suite))

        # Set the stats directory for each simulation in the suite
        for sim_name, sim_config in suite_config.simulations.items():
            sim_specific_stats_dir = stats_dir / sim_name
            sim_specific_stats_dir.mkdir(parents=True, exist_ok=True)
            sim_config.stats_dir = str(sim_specific_stats_dir)

        # Run the simulation
        sim = SimulationSuite(config=suite_config, policy_pr=pr, policy_store=policy_store)
        sim.simulate()

        # Export merged stats if an eval_stats_uri is provided
        if eval_stats_uri:
            # Process each simulation's merged DB
            for sim_name in suite_config.simulations:
                merged_db_path = stats_dir / sim_name / "stats.duckdb"
                if merged_db_path.exists():
                    logger.info(f"Exporting merged stats DB for {sim_name} to {eval_stats_uri}")
                    StatsDB.export_db(merged_db_path, eval_stats_uri)
                    logger.info(f"Stats DB for {sim_name} exported to {eval_stats_uri}")
                else:
                    logger.warning(f"No merged stats DB found at {merged_db_path}")

        logger.info(f"Evaluation complete for policy {pr.uri}")


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.sim")
    logger.info(f"Sim job config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    sim_job = SimJob(cfg.sim_job)
    assert isinstance(sim_job, SimJob)
    with WandbContext(cfg) as wandb_run:
        for policy_uri in sim_job.policy_uris:
            simulate_policy(sim_job, policy_uri, cfg, wandb_run, logger)


if __name__ == "__main__":
    main()

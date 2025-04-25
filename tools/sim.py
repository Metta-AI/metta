"""Simulation tools for evaluating policies in the Metta environment."""

import logging
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class SimJob(Config):
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "latest"


def simulate_policy(sim_job: SimJob, policy_uri: str, cfg: DictConfig, wandb_run):
    logger = logging.getLogger("metta.tools.sim")
    # TODO: Remove dependence on cfg in PolicyStore
    policy_store = PolicyStore(cfg, wandb_run)
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type)
    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")
        sim = SimulationSuite(config=sim_job.simulation_suite, policy_pr=pr, policy_store=policy_store)
        stats = sim.simulate()
        stats_logger = EvalStatsLogger(sim_job.simulation_suite, wandb_run)
        stats_logger.log(stats)
        logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    logger = logging.getLogger("metta.tools.sim")
    logger.info(f"Sim job config: {OmegaConf.to_yaml(cfg, resolve=True)}")
    sim_job = SimJob(cfg.sim_job)
    assert isinstance(sim_job, SimJob)
    with WandbContext(cfg) as wandb_run:
        for policy_uri in sim_job.policy_uris:
            simulate_policy(sim_job, policy_uri, cfg, wandb_run)


if __name__ == "__main__":
    main()

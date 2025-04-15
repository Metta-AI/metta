import logging
from dataclasses import dataclass
from typing import List

import hydra
from omegaconf import MISSING, DictConfig

from metta.agent.policy_store import PolicyStore
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


@dataclass
class SimJob:
    policy_uris: List[str] = MISSING
    eval_stats_db_uri: str = MISSING
    simulation_suite: SimulationSuiteConfig = MISSING
    selector_type: str = "latest"


def simulate_policy(sim_job: SimJob, wandb_run):
    logger = logging.getLogger("metta.tools.sim")
    policy_store = PolicyStore(sim_job.simulation_suite, wandb_run)
    policy_prs = policy_store.policies(sim_job.policy_uris, sim_job.selector_type)
    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info(f"Evaluating policy {pr.uri}")
        sim = Simulation(policy_store, pr, sim_job.simulation_suite)
        stats = sim.simulate()
        stats_logger = EvalStatsLogger(sim_job.simulation_suite, wandb_run)
        stats_logger.log(stats)
        logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")


@hydra.main(version_base=None, config_path="../configs", config_name="sim")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    job = SimJob(**cfg.sim_job)
    with WandbContext(cfg) as wandb_run:
        for policy_uri in sim_job.policy_uris:
            simulate_policy(sim_job, wandb_run)


if __name__ == "__main__":
    main()

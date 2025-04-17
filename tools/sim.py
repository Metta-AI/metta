import logging

import hydra
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.sim.eval_stats_logger import EvalStatsLogger
from metta.sim.simulation import Simulation
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


def simulate(eval: Simulation, cfg: DictConfig, wandb_run):
    stats = eval.simulate()
    stats_logger = EvalStatsLogger(cfg, wandb_run)
    stats_logger.log(stats)


def simulate_policy(cfg: DictConfig, wandb_run):
    logger = logging.getLogger("metta.tools.sim")
    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_prs = policy_store.policies(cfg.eval.policy_uri, cfg.eval.selector_type)
        # For each checkpoint of the policy, simulate
        for pr in policy_prs:
            logger.info(f"Evaluating policy {pr.uri}")

            wandb_run_id = wandb_run and wandb_run.id
            eval = hydra.utils.instantiate(
                cfg.eval, policy_store, pr, cfg.get("run_id", wandb_run_id), cfg_recursive_=False
            )
            simulate(eval, cfg, wandb_run)
            logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")


def simulate_policies(cfg: DictConfig):
    with WandbContext(cfg) as wandb_run:
        for policy_uri in cfg.eval.policy_uris:
            cfg.eval.policy_uri = policy_uri
            simulate_policy(cfg, wandb_run)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_mettagrid_environment(cfg)
    if hasattr(cfg.eval, "policy_uris") and cfg.eval.policy_uris is not None and len(cfg.eval.policy_uris) > 1:
        simulate_policies(cfg)
    else:
        with WandbContext(cfg) as wandb_run:
            simulate_policy(cfg, wandb_run)


if __name__ == "__main__":
    main()

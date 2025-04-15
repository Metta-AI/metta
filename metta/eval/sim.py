"""
Simulates a set of policies in a set of environments.
"""

import logging

import hydra
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.rl.eval.eval_stats_logger import EvalStatsLogger
from metta.rl.pufferlib.eval import Eval
from metta.rl.wandb.wandb_context import WandbContext

def simulate(eval: Eval, cfg: DictConfig, wandb_run):
    stats = eval.evaluate()
    stats_logger = EvalStatsLogger(cfg, wandb_run)
    stats_logger.log(stats)


def simulate_policy(cfg: DictConfig, wandb_run):
    logger = logging.getLogger("metta.tools.eval")
    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_prs = policy_store.policies(cfg.eval.policy_uri, cfg.eval.selector_type)
        # For each checkpoint of the policy, simulate
        for pr in policy_prs:
            logger.info(f"Evaluating policy {pr.uri}")

            eval = hydra.utils.instantiate(
                cfg.eval, policy_store, pr, cfg.get("run_id", wandb_run.id), cfg_recursive_=False
            )
            simulate(eval, cfg, wandb_run)
            logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")


def simulate_policies(cfg: DictConfig):
    with WandbContext(cfg) as wandb_run:
        for policy_uri in cfg.eval.policy_uris:
            cfg.eval.policy_uri = policy_uri
            simulate_policy(cfg, wandb_run)

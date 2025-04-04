import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from util.runtime_configuration import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.eval.eval_stats_logger import EvalStatsLogger
from rl.eval.eval_stats_db import EvalStatsDB
from rl.wandb.wandb_context import WandbContext


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)

    logger = logging.getLogger("metta.tools.eval")
    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_prs = policy_store.policies(cfg.eval.policy_uri, cfg.eval.selector_type)
        for pr in policy_prs:
            logger.info(f"Evaluating policy {pr.uri}")

            eval = hydra.utils.instantiate(
                cfg.eval,
                policy_store,
                pr,
                cfg.get("run_id", wandb_run.id),
                cfg_recursive_=False
            )
            stats = eval.evaluate()
            logger.info(f"Evaluation complete for policy {pr.uri}; logging stats")
            stats_logger = EvalStatsLogger(cfg, wandb_run)

            stats_logger.log(stats)

        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)

if __name__ == "__main__":
    main()

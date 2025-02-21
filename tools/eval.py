import logging

import hydra
from omegaconf import DictConfig
from mettagrid.config.config import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.eval.eval_stats_logger import EvalStatsLogger
from rl.eval.eval_stats_db import EvalStatsDB
from rl.wandb.wandb_context import WandbContext

logger = logging.getLogger("eval.py")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_pr = policy_store.policy(cfg.eval.policy_uri)

        eval = hydra.utils.instantiate(
            cfg.eval,
            policy_store,
            policy_pr,
            cfg.env,
            cfg_recursive_=False
        )
        stats = eval.evaluate()
        stats_logger = EvalStatsLogger(cfg, wandb_run)

        stats_logger.log(stats)

        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        analyzer.analyze()

if __name__ == "__main__":
    main()

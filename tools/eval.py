import logging

import hydra
from omegaconf import DictConfig
from mettagrid.config.config import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.eval_stats_logger import EvalStatsLogger
from rl.wandb.wandb_context import WandbContext

logger = logging.getLogger("eval.py")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        eval = hydra.utils.instantiate(
            cfg.eval,
            policy_store,
            cfg.env,
            cfg_recursive_=False
        )
        stats = eval.evaluate()
        stats_logger = EvalStatsLogger(cfg, wandb_run)
        stats_logger.log(stats, file_name="policy", artifact_name=cfg.eval.eval_artifact_name)




if __name__ == "__main__":
    main()

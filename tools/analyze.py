import os
import signal
import logging
import hydra
from omegaconf import DictConfig
from rl.wandb.wandb_context import WandbContext
from util.runtime_configuration import setup_metta_environment
from rl.eval.eval_stats_db import EvalStatsDB
logger = logging.getLogger("analyze.py")

@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")

def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)
        analyzer =  hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        analyzer.analyze()

if __name__ == "__main__":
    main()

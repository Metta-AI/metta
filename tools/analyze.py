import os
import signal
import logging
import hydra
from omegaconf import DictConfig
from rl.wandb.wandb_context import WandbContext
from mettagrid.config.config import setup_metta_environment
from rl.eval.eval_stats_db import EvalStatsDB
logger = logging.getLogger("analyze.py")

# Aggressively exit on Ctrl+C
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")

def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(cfg.analyzer.db_uri, wandb_run)
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        analyzer.analyze()

if __name__ == "__main__":
    main()

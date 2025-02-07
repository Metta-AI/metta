import os
import signal
import logging
import hydra
from omegaconf import DictConfig
from rl.wandb.wandb_context import WandbContext
from mettagrid.config.config import setup_metta_environment
from rl.wandb.wanduckdb import WandbDuckDB
logger = logging.getLogger("analyze.py")

# Aggressively exit on Ctrl+C
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")

def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        eval_stats_db = WandbDuckDB(
            wandb_run.entity,
            wandb_run.project,
            cfg.analyzer.eval_stats_uri,
            table_name=cfg.analyzer.table_name
        )
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        analyzer.run_all()


if __name__ == "__main__":
    main()

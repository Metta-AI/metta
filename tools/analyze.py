import os
import signal
import logging
import hydra
from omegaconf import DictConfig
from rl.wandb.wandb_context import WandbContext
from mettagrid.config.config import setup_metta_environment
from rl.eval.eval_stats_db import EvalStatsDbWandb, EvalStatsDbFile
logger = logging.getLogger("analyze.py")

# Aggressively exit on Ctrl+C
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")

def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)

    if cfg.analyzer.file_path:
        logger.info(f"Analyzing file: {cfg.analyzer.file_path}")
        eval_stats_db_from_file = EvalStatsDbFile(cfg.analyzer.file_path)
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db_from_file)
        analyzer.run()

    elif cfg.analyzer.artifact_name:
        logger.info(f"Analyzing artifact: {cfg.analyzer.artifact_name}")
        with WandbContext(cfg) as wandb_run:
            eval_stats_db_from_artifact = EvalStatsDbWandb(
                wandb_run.entity,
                wandb_run.project,
                cfg.analyzer.artifact_name,
                cfg.analyzer.version,
                cfg.analyzer.table_name
            )

        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db_from_artifact)
        analyzer.run()


if __name__ == "__main__":
    main()

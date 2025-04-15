import gzip
import json
import logging
import os

import wandb

logger = logging.getLogger("eval_stats_logger.py")


class EvalStatsLogger:
    def __init__(self, cfg, wandb_run):
        self._cfg = cfg
        self._wandb_run = wandb_run
        # We want local stats dir to be the same for train, analysis and eval for a particular run
        save_dir = (cfg.run_dir).replace("analyze", "train").replace("eval", "train")

        artifact_name = None
        if cfg.eval.eval_db_uri is None:
            json_path = os.path.join(save_dir, "eval_stats")
        elif cfg.eval.eval_db_uri.startswith("wandb://"):
            artifact_name = cfg.eval.eval_db_uri.split("/")[-1]
            json_path = os.path.join(save_dir, "eval_stats")
        elif cfg.eval.eval_db_uri.startswith("file://"):
            json_path = cfg.eval.eval_db_uri.split("file://")[1]
        else:
            if "://" in cfg.eval.eval_db_uri:
                raise ValueError(f"Invalid eval_db_uri: {cfg.eval.eval_db_uri}")
            json_path = cfg.eval.eval_db_uri

        self.json_path = json_path if json_path.endswith(".json") else f"{json_path}.json"
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        self.artifact_name = artifact_name

    def _log_to_file(self, eval_stats):
        """
        eval_stats is a dictionary of format {str eval_name : list stats: }"""
        # If file exists, load and merge with existing data
        gzip_path = self.json_path + ".gz"
        if os.path.exists(gzip_path):
            try:
                logger.info(f"Loading existing eval stats from {gzip_path}")
                with gzip.open(gzip_path, "rt", encoding="utf-8") as f:
                    existing_stats = json.load(f)
                for eval_name, stats in eval_stats.items():
                    existing_stats[eval_name].extend(stats)
                eval_stats = existing_stats
            except Exception as e:
                logger.error(f"Error loading existing eval stats from {gzip_path}: {e}, will overwrite")

        with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
            json.dump(eval_stats, f, indent=4)
        logger.info(f"Saved eval stats to {gzip_path}")

    def _log_to_wandb(self, artifact_name: str, eval_stats):
        artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
        zip_file_path = self.json_path + ".gz"
        with gzip.open(zip_file_path, "wt", encoding="utf-8") as f:
            json.dump(eval_stats, f)
        artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
        artifact.add_file(zip_file_path)
        artifact.save()
        artifact.wait()
        self._wandb_run.log_artifact(artifact)
        logger.info(f"Logged artifact {artifact_name} to wandb")

    def log(self, eval_stats):
        self._log_to_file(eval_stats)
        if self.artifact_name is not None:
            self._log_to_wandb(self.artifact_name, eval_stats)

        return eval_stats

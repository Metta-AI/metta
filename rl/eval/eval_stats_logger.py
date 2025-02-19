import os
import json
from datetime import datetime
import wandb
from util.datastruct import flatten_config
import logging
from pathlib import Path

logger = logging.getLogger("eval_stats_logger.py")

class EvalStatsLogger:
    def __init__(self, cfg, wandb_run):
        self._cfg = cfg
        self._wandb_run = wandb_run

        artifact_name = None
        if cfg.eval.eval_db_uri is None:
            json_path = os.path.join(cfg.run_dir, "eval_stats")
        elif cfg.eval.eval_db_uri.startswith("wandb://"):
            artifact_name = cfg.eval.eval_db_uri.split("/")[-1]
            json_path = os.path.join(cfg.run_dir, "eval_stats")
        elif cfg.eval.eval_db_uri.startswith("file://"):
            json_path = cfg.eval.eval_db_uri.split("file://")[1]
        else:
            if "://" in cfg.eval.eval_db_uri:
                raise ValueError(f"Invalid eval_db_uri: {cfg.eval.eval_db_uri}")
            json_path = cfg.eval.eval_db_uri

        self._json_path = json_path if json_path.endswith('.json') else json_path + '.json'
        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        self.artifact_name = artifact_name

    def _add_additional_fields(self, eval_stats, eval_name):
        additional_fields = {}
        additional_fields['run_id'] = self._cfg.get("run_id", self._wandb_run.id)
        additional_fields['eval_name'] = self._cfg.eval.get("name", eval_name)
        if self._cfg.eval.npc_policy_uri is not None:
            additional_fields['npc'] = self._cfg.eval.npc_policy_uri
        additional_fields['timestamp'] = datetime.now().isoformat()

        # Convert the environment configuration to a dictionary and flatten it.
        flattened_env = flatten_config(self._cfg.env.game, parent_key = "env.game")
        additional_fields.update(flattened_env)

        for episode in eval_stats:
            for record in episode:
                record.update(additional_fields)

        return eval_stats

    def _log_to_file(self, eval_stats):
        # If file exists, load and merge with existing data
        if os.path.exists(self._json_path):
            with open(self._json_path, "r") as f:
                existing_stats = json.load(f)
            eval_stats.extend(existing_stats)
        with open(self._json_path, "w") as f:
            json.dump(eval_stats, f, indent=4)
        logger.info(f"Saved eval stats to {self._json_path}")

    def _log_to_wandb(self, artifact_name: str):
        artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
        artifact.add_file(self._json_path)
        artifact.save()
        artifact.wait()
        self._wandb_run.log_artifact(artifact)
        logger.info(f"Logged artifact {artifact_name} to wandb")

    def log(self, eval_stats):

        # If we are running eval suite, we need to add additional fields for each eval
        if isinstance(eval_stats, dict):
            all_eval_stats = []
            for eval_name, stats in eval_stats.items():
                self._add_additional_fields(stats, eval_name = eval_name)
                all_eval_stats.extend(stats)
            eval_stats = all_eval_stats
        else:
            self._add_additional_fields(eval_stats)

        self._log_to_file(eval_stats)

        if self.artifact_name is not None:
            self._log_to_wandb(self.artifact_name)

        return eval_stats

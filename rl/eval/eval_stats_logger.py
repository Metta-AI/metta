import os
import json
from datetime import datetime
import wandb
from util.datastruct import flatten_dict
import logging
from pathlib import Path
from typing import List
logger = logging.getLogger("eval.py")

class EvalStatsLogger:
    def __init__(self, cfg, wandb_run):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.log_dir = os.path.join(cfg.run_dir, "eval_stats")
        os.makedirs(self.log_dir, exist_ok=True)

    def add_additional_fields(self, eval_stats):
        additional_fields = {}
        additional_fields['run_id'] = self.cfg.get("run_id", self.wandb_run.id)
        additional_fields['eval_name'] = self.cfg.eval.get("name", "eval")
        if self.cfg.eval.npc_policy_uri is not None:
            additional_fields['npc'] = self.cfg.eval.npc_policy_uri
        additional_fields['timestamp'] = datetime.now().isoformat()

        # Convert the environment configuration to a dictionary and flatten it.
        flattened_env = flatten_dict(self.cfg.env.game, parent_key = "env.game")
        additional_fields.update(flattened_env)

        for episode in eval_stats:
            for record in episode:
                record.update(additional_fields)

        return eval_stats

    def log_to_file(self, eval_stats, file_name: str):
        json_path = os.path.join(self.log_dir, f"{file_name}.json")
        with open(json_path, "w") as f:
            json.dump(eval_stats, f, indent=4)
        logger.info(f"Saved eval stats to {json_path}")
        return json_path

    def log_to_wandb(self, artifact_name: str, json_path: str):
        artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
        artifact.add_file(json_path)
        artifact.save()
        artifact.wait()
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Logged artifact {artifact_name} to wandb")

    def log(self, eval_stats, file_name: str, artifact_name: str = None):

        if isinstance(eval_stats, dict):
            for eval_name, stats in eval_stats.items():
                self.log(stats, file_name=f"{file_name}_{eval_name}", artifact_name=artifact_name)
            return

        eval_stats = self.add_additional_fields(eval_stats)

        json_path = self.log_to_file(eval_stats, file_name)

        if artifact_name is not None:
            self.log_to_wandb(artifact_name, json_path)

        return eval_stats

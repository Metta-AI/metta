import os
import json
from datetime import datetime
import wandb
from util.datastruct import flatten_dict
import logging
logger = logging.getLogger("eval.py")

class EvalStatsLogger:
    def __init__(self, cfg, wandb_run):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.log_dir = os.path.join(cfg.run_dir, "eval_stats")
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, eval_stats, file_name: str, artifact_name: str = None):
        # Build additional fields that you want to inject into each record.
        additional_fields = {}
        additional_fields['run_id'] = self.cfg.get("run_id", self.wandb_run.id)
        additional_fields['eval_name'] = self.cfg.eval.get("name", "eval")
        if self.cfg.eval.npc_policy_uri is not None:
            additional_fields['npc'] = self.cfg.eval.npc_policy_uri
        additional_fields['timestamp'] = datetime.now().isoformat()

        # Convert the environment configuration to a dictionary and flatten it.
        #env_dict = OmegaConf.to_container(cfg.env, resolve=True)
        flattened_env = flatten_dict(self.cfg.env.game, parent_key = "env.game")
        additional_fields.update(flattened_env)

        for episode in eval_stats:
            for record in episode:
                # Update each record with the additional fields.
                record.update(additional_fields)

        # Write game_stats to JSON file
        json_path = os.path.join(self.log_dir, f"{file_name}.json")
        with open(json_path, "w") as f:
            json.dump(eval_stats, f, indent=4)
        logger.info(f"Saved eval stats to {json_path}")

        # Optionally log the JSON file as an artifact:
        if artifact_name is not None:
            artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
            artifact.add_file(json_path)
            self.wandb_run.log_artifact(artifact)
            logger.info(f"Logged artifact {artifact_name} to wandb")

        return eval_stats


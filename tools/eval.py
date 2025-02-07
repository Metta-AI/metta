import logging

import hydra
from omegaconf import DictConfig
from mettagrid.config.config import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.wandb.wandb_context import WandbContext
import os
import wandb
import json
from datetime import datetime
from util.stats_library import (
    EloTest,
    Glicko2Test,
    MannWhitneyUTest,
    StatisticalTest,
    get_test_results
)

logger = logging.getLogger("eval.py")
def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flatten a nested dictionary using dot notation.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key (used in recursion).
        sep (str): Separator between keys.

    Returns:
        dict: A flattened dictionary with keys in dot notation.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.update(flatten_dict(dict(v), new_key, sep=sep))
            continue
        else:
            items[new_key] = v
    return items
# class Eval:

#     def __init__(self, cfg: DictConfig, metrics = ["mann_whitney_u", "elo", "glicko2", "time_to_targets"], env_name = None):
#         """
#         Store config for later use.
#         """
#         self.cfg = cfg
#         self.metrics = metrics
#         self.env_name = env_name

#     def log_metrics(self, stats):
#         """
#         Logs the various metrics using the stats gathered.
#         """
#         logger.setLevel(logging.INFO)

#         _, mean_altar_use = get_test_results(MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['action.use.altar'], mode = 'mean', label = self.env_name)
#             )
#         logger.info("\n" + mean_altar_use)

#         if "time_to_targets" in self.metrics:
#             _, time_to_targets = get_test_results(
#                 MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['time_to'], mode = 'mean', label = self.env_name)
#             )
#             logger.info("\n" + time_to_targets)


#         if "mann_whitney_u" in self.metrics:
#             # Generic Mann-Whitney on "all" stats
#             _, fr = get_test_results(
#                 MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['all'])
#             )
#             logger.info("\n" + fr)

#         if "elo" in self.metrics:
#             # Elo test on "altar" category
#             _, fr = get_test_results(
#                 EloTest(stats, self.cfg.evaluator.stat_categories['altar']),
#                 self.cfg.evaluator.baselines.elo_scores_path
#             )
#             logger.info("\n" + fr)

#         if "glicko2" in self.metrics:
#             # Glicko2 test on "altar" category
#             _, fr = get_test_results(
#                 Glicko2Test(stats, self.cfg.evaluator.stat_categories['altar']),
#                 self.cfg.evaluator.baselines.glicko_scores_path
#             )
#             logger.info("\n" + fr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        eval = hydra.utils.instantiate(
            cfg.eval,
            policy_store,
            cfg.env,
            cfg.run_dir,
            _recursive_=False
        )
        stats = eval.evaluate()

        # Build additional fields that you want to inject into each record.
        additional_fields = {}
        additional_fields['run_id'] = cfg.get("run_id", wandb_run.id)
        additional_fields['eval_name'] = cfg.eval.get("name", "eval")
        if cfg.eval.npc_policy_uri is not None:
            additional_fields['npc'] = cfg.eval.npc_policy_uri
        additional_fields['timestamp'] = datetime.now().isoformat()

        # Convert the environment configuration to a dictionary and flatten it.
        #env_dict = OmegaConf.to_container(cfg.env, resolve=True)
        flattened_env = flatten_dict(cfg.env.game, parent_key = "env.game")
        additional_fields.update(flattened_env)

        for episode in stats:
            for record in episode:
                # Update each record with the additional fields.
                record.update(additional_fields)

        # Write game_stats to JSON file
        run_dir = cfg.run_dir
        artifact_name = cfg.eval.eval_artifact_name
        json_path = os.path.join(run_dir, f"{artifact_name}.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved eval stats to {json_path}")

        # Optionally log the JSON file as an artifact:
        if cfg.eval.log_eval_artifact:
            run_dir = cfg.run_dir
            json_path = os.path.join(run_dir, f"{artifact_name}.json")
            artifact = wandb.Artifact(name=artifact_name, type=artifact_name)
            artifact.add_file(json_path)
            wandb_run.log_artifact(artifact)
            logger.info(f"Logged artifact {artifact_name} to wandb")

            #analyze stats to print some metrics
           # wandb_db = WandbDuckDB(entity, project, artifact_name, alias="latest", table_name="eval_stats")
           #run elo analysis
        return stats


if __name__ == "__main__":
    main()

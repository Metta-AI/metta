import os
import time
import logging

import wandb
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rl.carbs.metta_carbs import MettaCarbs
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri, upload_policy_to_wandb
from rl.wandb.sweep import generate_run_id_for_sweep
from util.eval_analyzer import Analysis
import json
logger = logging.getLogger("sweep_rollout")


class CarbsSweepRollout:
    def __init__(self, cfg: OmegaConf, wandb_run):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.sweep_id = wandb_run.sweep_id

        self.run_id = generate_run_id_for_sweep(self.sweep_id, self.cfg.run_dir)
        self.run_dir = os.path.join(self.cfg.run_dir, "runs", self.run_id)
        os.makedirs(self.run_dir)
        wandb_run.name = self.run_id

        self._log_file("run_cfg.yaml", self.cfg)

        self.carbs = MettaCarbs(cfg, wandb_run)
        self._log_file("carbs_state.yaml", {
            "generation": self.carbs.generation,
            "observations": self.carbs._observations,
            "params": str(self.carbs._carbs.params)
        })

        self.suggestion = self.carbs.suggest()
        self._log_file("carbs_suggestion.yaml", self.suggestion)

        logger.info("Generated CARBS suggestion: ")
        logger.info(yaml.dump(self.suggestion, default_flow_style=False))
        self._log_file("carbs_suggestion.yaml", self.suggestion)

    def run(self):
        try:
            self._run()
        except Exception as e:
            logger.error(f"Error running suggested rollout: {e}")
            Console().print_exception()
            self.carbs.record_failure()
            return False
        return True

    def _run(self):
        wandb_run = self.wandb_run
        sweep_stats = {}
        start_time = time.time()
        train_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        train_cfg.sweep = {}

        train_cfg.run = self.run_id
        train_cfg.data_dir = os.path.join(self.cfg.run_dir, "runs")
        train_cfg.wandb.group = self.cfg.run

        eval_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        eval_cfg.evaluator = OmegaConf.create(OmegaConf.to_container(self.cfg.sweep.evaluator))

        self._apply_carbs_suggestion(train_cfg, self.suggestion)
        if self.cfg.sweep.generation.enabled:
            train_cfg.agent.policy_selector.generation = self.carbs.generation - 1

        self._log_file("config.yaml", self.cfg)
        self._log_file("train_config.yaml", train_cfg)
        self._log_file("eval_config.yaml", eval_cfg)
        self._log_file("carbs_suggestion.yaml", self.suggestion)


        train_start_time = time.time()
        trainer = hydra.utils.instantiate(train_cfg.trainer, train_cfg, wandb_run)

        initial_policy_uri = trainer.uncompiled_policy.uri
        logger.info(f"Loading initial policy from {initial_policy_uri}")
        initial_policy = load_policy_from_uri(initial_policy_uri, eval_cfg, wandb_run)
        initial_metadata = {}
        if hasattr(initial_policy, "metadata"):
            initial_metadata = initial_policy.metadata

        sweep_stats.update({
            "score.metric": eval_cfg.sweep.metric,
            "initial.uri": initial_policy_uri,
            "generation": initial_metadata.get("generation", 0) + 1
        })
        wandb_run.summary.update(sweep_stats)

        trainer.train()
        trainer.close()
        train_time = time.time() - train_start_time

        sweep_stats.update({
            "train.agent_steps": trainer.policy_checkpoint.agent_steps,
            "train.epoch": trainer.policy_checkpoint.epoch,
            "time.train": train_time,
        })
        wandb_run.summary.update(sweep_stats)

        eval_start_time = time.time()
        policy_uri = trainer.policy_checkpoint.model_path
        logger.info(f"Loading final policy from {trainer.policy_checkpoint.model_path}")
        trained_policy = load_policy_from_uri(policy_uri, eval_cfg, wandb_run)

        logger.info(f"Evaluating policy {trained_policy.name} against {initial_policy.name}")
        evaluator = hydra.utils.instantiate(eval_cfg.evaluator, eval_cfg, trained_policy, [initial_policy])

        stats = evaluator.evaluate()
        evaluator.close()
        eval_time = time.time() - eval_start_time
        self._log_file("eval_stats.yaml", stats)

        elo_analysis = Analysis(stats, eval_method='elo_1v1', stat_category=eval_cfg.sweep.metric)
        elo_results = elo_analysis.get_results()
        logger.info("\n" + elo_analysis.get_display_results())
        self._log_file("elo_results.yaml", elo_results)

        p_analysis = Analysis(stats, eval_method='1v1', stat_category='all')
        logger.info("\n" + p_analysis.get_display_results())
        self._log_file("p_results.yaml", p_analysis.get_results())

        altar_analysis = Analysis(stats, eval_method='1v1', stat_category=eval_cfg.sweep.metric)
        logger.info("\n" + altar_analysis.get_display_results())
        altar_results = altar_analysis.get_results()
        self._log_file("altar_results.yaml", altar_results)

        final_score = altar_results[eval_cfg.sweep.metric]['policy_stats'][trained_policy.name]['mean']
        initial_score = altar_results[eval_cfg.sweep.metric]['policy_stats'][initial_policy.name]['mean']
        eval_metric = final_score

        sweep_stats.update({
            "initial.uri": initial_policy_uri,
            "initial.score": initial_score,
            "initial.elo": elo_results[initial_policy.name],

            "final.uri": trained_policy.uri,
            "final.score": final_score,
            "final.elo": elo_results[trained_policy.name],

            "time.eval": eval_time,
            "time.total": train_time + eval_time,

            "delta.elo": elo_results[trained_policy.name] - elo_results[initial_policy.name],
            "delta.score": final_score - initial_score,
        })

        for stat in ["train.agent_steps", "train.epoch", "time.train", "time.eval", "time.total"]:
            sweep_stats["lineage." + stat] = sweep_stats[stat] + initial_metadata.get("lineage." + stat, 0)

        wandb_run.summary.update(sweep_stats)
        logger.info(
            "Sweep Stats: \n" +
            json.dumps({ k: str(v) for k, v in sweep_stats.items() }, indent=4))

        final_model_artifact = upload_policy_to_wandb(
            wandb_run,
            trainer.policy_checkpoint.model_path,
            f"{self.run_id}.model",
            metadata={
                **sweep_stats,
                "training_run": train_cfg.run,
            },
            artifact_type="sweep_model",
        )

        final_model_artifact.link(
            self.cfg.run, [self.run_id]
        )

        total_time = time.time() - start_time
        logger.info(f"Carbs Observation: {eval_metric}, {total_time}")
        self.carbs.record_observation(eval_metric, total_time)
        wandb_run.summary.update({"run_time": total_time})

    def _log_file(self, name: str, data):
        path = os.path.join(self.run_dir, name)
        with open(path, "w") as f:
            if isinstance(data, DictConfig):
                data = OmegaConf.to_container(data, resolve=True)
            json.dump(data, f, indent=4)

        wandb.run.save(path, base_path=self.run_dir)

    def _apply_carbs_suggestion(self, config: OmegaConf, suggestion: DictConfig):
        for key, value in suggestion.items():
            if key == "suggestion_uuid":
                continue
            new_cfg_param = config
            key_parts = key.split(".")
            for k in key_parts[:-1]:
                new_cfg_param = new_cfg_param[k]
            param_name = key_parts[-1]
            new_cfg_param[param_name] = value

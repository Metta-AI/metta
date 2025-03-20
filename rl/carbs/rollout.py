import os
import time
import logging
import json
import wandb
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig
from rich.console import Console

from rl.carbs.metta_carbs import MettaCarbs
from rl.wandb.sweep import generate_run_id_for_sweep
from agent.policy_store import PolicyStore
from rl.eval.eval_stats_logger import EvalStatsLogger
from rl.eval.eval_stats_db import EvalStatsDB
from pathlib import Path

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
        self.eval_stats_logger = EvalStatsLogger(cfg, wandb_run)

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

        policy_store = PolicyStore(train_cfg, wandb_run)

        train_cfg.run = self.run_id
        train_cfg.run_dir = os.path.join(self.cfg.run_dir, "runs", self.run_id)
        train_cfg.wandb.group = self.cfg.run

        eval_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        eval_cfg.eval.update(self.cfg.sweep.eval)

        self._apply_carbs_suggestion(train_cfg, self.suggestion)

        # if self.cfg.sweep.generation.enabled:
        #     train_cfg.agent.policy_selector.generation = self.carbs.generation - 1

        self._log_file("config.yaml", self.cfg)
        self._log_file("train_config.yaml", train_cfg)
        self._log_file("carbs_suggestion.yaml", self.suggestion)

        train_start_time = time.time()
        trainer = hydra.utils.instantiate(train_cfg.trainer, train_cfg, wandb_run, policy_store)
        if train_cfg.trainer.initial_policy.uri is not None:
            initial_pr = policy_store.policy(train_cfg.trainer.initial_policy)
        else:
            initial_pr = policy_store.policy(trainer.initial_pr_uri())

        sweep_stats.update({
            "score.metric": self.cfg.sweep.metric,
            "initial.uri": initial_pr.uri,
            "generation": initial_pr.metadata["generation"]
        })
        wandb_run.summary.update(sweep_stats)

        trainer.train()
        trainer.close()
        train_time = time.time() - train_start_time

        sweep_stats.update({
            "train.agent_step": trainer.agent_step,
            "train.epoch": trainer.epoch,
            "time.train": train_time,
        })
        wandb_run.summary.update(sweep_stats)

        eval_start_time = time.time()

        final_pr = policy_store.policy(trainer.last_pr_uri())

        policy_store.add_to_wandb_run(wandb_run.name, final_pr)
        logger.info(f"Final policy saved to {final_pr.uri}")

        logger.info(f"Evaluating policy {final_pr.name}")
        self._log_file("eval_config.yaml", eval_cfg)

        eval_cfg.eval.policy_uri = final_pr.uri
        eval_cfg.analyzer.policy_uri = final_pr.uri
        eval = hydra.utils.instantiate(eval_cfg.eval, policy_store, final_pr, eval_cfg.env, _recursive_ = False)
        stats = eval.evaluate()
        eval_time = time.time() - eval_start_time

        self.eval_stats_logger.log(stats)

        eval_stats_db = EvalStatsDB.from_uri(self.eval_stats_logger.json_path, self.run_dir, wandb_run)
        analyzer = hydra.utils.instantiate(eval_cfg.analyzer, eval_stats_db)

        metric_idxs = [i for i, m in enumerate(analyzer.analysis.metrics) if m == eval_cfg.sweep.metric]
        if len(metric_idxs) == 0:
            raise ValueError(f"Metric {eval_cfg.sweep.metric} not found in analyzer metrics: {analyzer.analysis.metrics}")
        elif len(metric_idxs) > 1:
            raise ValueError(f"Multiple metrics found for {eval_cfg.sweep.metric} in analyzer")
        sweep_metric_index = metric_idxs[0]

        results, _ = analyzer.analyze()
        # Filter by policy name and sum up the mean values over evals
        filtered_results = results[sweep_metric_index][results[sweep_metric_index]['policy_name'] == final_pr.name]
        eval_metric = filtered_results['mean'].sum()

        stats_update = {
            "time.eval": eval_time,
            "time.total": train_time + eval_time,
            "uri": final_pr.uri,
            "score": eval_metric,
        }

        sweep_stats.update(stats_update)

        for stat in ["train.agent_step", "train.epoch", "time.train", "time.eval", "time.total"]:
            sweep_stats["lineage." + stat] = sweep_stats[stat] + initial_pr.metadata.get("lineage." + stat, 0)

        wandb_run.summary.update(sweep_stats)
        logger.info(
            "Sweep Stats: \n" +
            json.dumps({ k: str(v) for k, v in sweep_stats.items() }, indent=4))

        final_pr.metadata.update({
            **sweep_stats,
            "training_run": train_cfg.run,
        })

        policy_store.add_to_wandb_sweep(self.cfg.run, final_pr)

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

import os
import time
import logging

import wandb
import yaml
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rl.carbs.spaces import carbs_from_cfg
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri, upload_policy_to_wandb
from rl.pufferlib.trainer import PufferTrainer
from rl.wandb.sweep import generate_run_id_for_sweep
from util.eval_analyzer import analyze_policy_stats

logger = logging.getLogger("sweep_rollout")

class CarbsSweepRollout:
    def __init__(self, cfg: OmegaConf, wandb_run):
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.sweep_id = wandb_run.sweep_id

        self.run_id = generate_run_id_for_sweep(self.sweep_id)
        self.run_dir = os.path.join(self.cfg.run_dir, "runs", self.run_id)
        os.makedirs(self.run_dir)
        wandb_run.name = self.run_id

        self.wandb_carbs = carbs_from_cfg(cfg, wandb_run)
        self.suggestion = self.wandb_carbs.suggest()

        self._log_file("sweep_config.yaml", self.cfg)

        logger.info("Generated CARBS suggestion: ")
        logger.info(yaml.dump(self.suggestion, default_flow_style=False))
        self._log_file("carbs_suggestion.yaml", self.suggestion)


    def run(self):
        try:
            self._run()
        except Exception as e:
            logger.error(f"Error running suggested rollout: {e}")
            Console().print_exception()
            self.wandb_carbs.record_failure()
            return False
        return True

    def _run(self):
        wandb_run = self.wandb_run
        start_time = time.time()
        train_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        train_cfg.sweep = {}

        train_cfg.run = self.run_id
        train_cfg.data_dir = os.path.join(self.cfg.run_dir, "runs")
        train_cfg.wandb.group = self.cfg.run

        eval_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        eval_cfg.eval = OmegaConf.create(OmegaConf.to_container(self.cfg.sweep.eval))

        self._apply_carbs_suggestion(train_cfg, self.suggestion)

        self._log_file("config.yaml", self.cfg)
        self._log_file("train_config.yaml", train_cfg)
        self._log_file("eval_config.yaml", eval_cfg)
        self._log_file("carbs_suggestion.yaml", self.suggestion)

        train_start_time = time.time()
        trainer = PufferTrainer(train_cfg, wandb_run)
        initial_policy_uri = trainer.uncompiled_policy.uri
        trainer.train()
        trainer.close()
        train_time = time.time() - train_start_time

        eval_start_time = time.time()
        policy_uri = trainer.policy_checkpoint.model_path
        logger.info(f"Loading final policy from {trainer.policy_checkpoint.model_path}")
        trained_policy = load_policy_from_uri(policy_uri, eval_cfg, wandb_run)
        logger.info(f"Loading initial policy from {initial_policy_uri}")
        initial_policy = load_policy_from_uri(initial_policy_uri, eval_cfg, wandb_run)

        logger.info(f"Evaluating policy {trained_policy.name} against {initial_policy.name}")
        evaluator = PufferEvaluator(eval_cfg, trained_policy, [initial_policy])
        stats = evaluator.evaluate()
        evaluator.close()
        eval_time = time.time() - eval_start_time

        policy_stats, policy_stats_table = analyze_policy_stats(stats, '1v1', 'all')
        logger.info("\n" + policy_stats_table)
        elo, elo_table = analyze_policy_stats(stats, 'elo_1v1', 'altar')
        logger.info("\n" + elo_table)

        train_mean = 0
        init_mean = 0
        eval_metric = 0

        stat_items = list(filter(lambda x: x['stat_name'] == self.cfg.sweep.metric, policy_stats))
        if len(stat_items) > 0:
            for stat in stat_items[0]['policy_stats']:
                if stat['policy_name'] == trained_policy.name:
                    train_mean = stat['mean']
                elif stat['policy_name'] == initial_policy.name:
                    init_mean = stat['mean']
                else:
                    raise ValueError(f"Policy {stat['policy_name']} not found in stats")

            eval_metric = train_mean - init_mean

        wandb_run.log(
            {"eval_metric": eval_metric},
            step=trainer.policy_checkpoint.agent_steps)

        total_lineage_time = time.time() - start_time
        policy_generation = 0
        if hasattr(initial_policy, "metadata"):
            total_lineage_time += initial_policy.metadata.get("total_lineage_time", 0)
            policy_generation = initial_policy.metadata.get("policy_generation", 0) + 1

        wandb_run.summary.update({
            "training_time": train_time,
            "eval_time": eval_time,
            "eval_metric": eval_metric,
            "train_policy_mean": train_mean,
            "init_policy_mean": init_mean,
            "agent_step": trainer.policy_checkpoint.agent_steps,
            "epoch": trainer.policy_checkpoint.epoch,
            "total_lineage_time": total_lineage_time,
            "policy_generation": policy_generation,
            "trained_policy_uri": trained_policy.uri,
            "init_policy_uri": initial_policy_uri,
            "trained_policy_elo": elo[0],
            "init_policy_elo": elo[1],
            "elo_delta": elo[0] - elo[1],
        })

        logger.info(f"Sweep Objective: {eval_metric}")
        logger.info(f"Sweep Train Time: {train_time}")
        logger.info(f"Sweep Eval Time: {eval_time}")
        logger.info(f"Sweep Total Lineage Time: {total_lineage_time}")
        logger.info(f"Sweep Policy Generation: {policy_generation}")

        self._log_file("eval_stats.yaml", stats)
        self._log_file("eval_config.yaml", eval_cfg)
        self._log_file("eval_stats.txt", stats)

        final_model_artifact = upload_policy_to_wandb(
            wandb_run,
            trainer.policy_checkpoint.model_path,
            f"{self.run_id}.model",
            metadata={
                "training_run": train_cfg.run,
                "agent_step": trainer.policy_checkpoint.agent_steps,
                "epoch": trainer.policy_checkpoint.epoch,
                "train_time": train_time,
                "eval_time": eval_time,
                "eval_objective": eval_metric,
                "train_policy_mean": train_mean,
                "init_policy_mean": init_mean,
                "total_lineage_time": total_lineage_time,
                "policy_generation": policy_generation,
                "trained_policy_elo": elo[0],
                "init_policy_elo": elo[1],
                "elo_delta": elo[0] - elo[1],
            },
            artifact_type="sweep_model",
        )
        final_model_artifact.link(
            self.cfg.run, [self.run_id]
        )

        total_time = time.time() - start_time
        logger.info(f"Carbs Observation: {eval_metric}, {total_time}")
        self.wandb_carbs.record_observation(eval_metric, total_time)
        wandb_run.summary.update({"run_time": total_time})

    def _compute_objective(self, stats, trained_policy_name, baseline_policy_names):
        sum = 0
        count = 0
        for game in stats:
            for agent in game:
                if agent["policy_name"] == trained_policy_name:
                    sum += agent.get(self.cfg.sweep.metric, 0)
                    count += 1
        logger.info(f"Sweep Metric: {self.cfg.sweep.metric} = {sum} / {count}")
        return sum / count

    def _log_file(self, name: str, data):
        path = os.path.join(self.run_dir, name)
        with open(path, "w") as f:
            if isinstance(data, DictConfig):
                data = OmegaConf.to_container(data, resolve=True)
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
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

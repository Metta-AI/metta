import os
import time
import logging

import wandb
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rl.carbs.metta_carbs import MettaCarbs
from rl.wandb.sweep import generate_run_id_for_sweep
from agent.policy_store import PolicyStore
from util.stats_library import EloTest, Glicko2Test, MannWhitneyUTest, get_test_results
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

        policy_store = PolicyStore(train_cfg, wandb_run)

        train_cfg.run = self.run_id
        train_cfg.run_dir = os.path.join(self.cfg.run_dir, "runs", self.run_id)
        train_cfg.wandb.group = self.cfg.run

        eval_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
        eval_cfg.evaluator.update(self.cfg.sweep.evaluator)

        self._apply_carbs_suggestion(train_cfg, self.suggestion)

        # if self.cfg.sweep.generation.enabled:
        #     train_cfg.agent.policy_selector.generation = self.carbs.generation - 1

        self._log_file("config.yaml", self.cfg)
        self._log_file("train_config.yaml", train_cfg)
        self._log_file("carbs_suggestion.yaml", self.suggestion)

        train_start_time = time.time()
        trainer = hydra.utils.instantiate(train_cfg.trainer, train_cfg, wandb_run, policy_store)
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

        logger.info(f"Evaluating policy {final_pr.name} against {initial_pr.name}")
        self._log_file("eval_config.yaml", eval_cfg)

        evaluator = hydra.utils.instantiate(eval_cfg.evaluator, eval_cfg, final_pr, [initial_pr])
        stats = evaluator.evaluate()
        evaluator.close()
        eval_time = time.time() - eval_start_time
        self._log_file("eval_stats.yaml", stats)

        all_stats_results, fr = get_test_results(MannWhitneyUTest(stats, eval_cfg.evaluator.stat_categories['all']))
        logger.info("\n" + fr)
        self._log_file("all_stats.yaml", all_stats_results)


        sweep_glicko, fr = get_test_results(
            Glicko2Test(stats, eval_cfg.evaluator.stat_categories[eval_cfg.sweep.metric]),
            eval_cfg.evaluator.baselines.glicko_scores_path)
        logger.info("\n" + fr)
        self._log_file("sweep_glicko.yaml", sweep_glicko)

        sweep_elo, fr = get_test_results(
            EloTest(stats, eval_cfg.evaluator.stat_categories[eval_cfg.sweep.metric]),
            eval_cfg.evaluator.baselines.elo_scores_path)
        logger.info("\n" + fr)
        self._log_file("sweep_elo.yaml", sweep_elo)

        final_score = sweep_glicko[final_pr.name]["rating"]
        initial_score = sweep_glicko[initial_pr.name]["rating"]
        eval_metric = final_score

        sweep_stats.update({
            "initial.uri": initial_pr.uri,
            "initial.score": initial_score,
            "initial.elo": sweep_elo[initial_pr.name]["score"],

            "final.uri": final_pr.uri,
            "final.score": final_score,
            "final.elo": sweep_elo[final_pr.name]["score"],
            "final.glicko": sweep_glicko[final_pr.name]["rating"],

            "time.eval": eval_time,
            "time.total": train_time + eval_time,

            "delta.elo": sweep_elo[final_pr.name]["score"] - sweep_elo[initial_pr.name]["score"],
            "delta.glicko": sweep_glicko[final_pr.name]["rating"] - sweep_glicko[initial_pr.name]["rating"],
            "delta.score": final_score - initial_score,
        })

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

        # final_model_artifact.link(
        #     self.sweep_id, [self.run_id]
        # )

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

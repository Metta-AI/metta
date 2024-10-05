import os
from copy import deepcopy
import time
import yaml
from carbs import ObservationInParam
from omegaconf import OmegaConf
from rich.console import Console
from rl.carbs.sweep import CarbsSweep
from rl.carbs.sweep import apply_carbs_suggestion, pow2_suggestion
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri, upload_policy_to_wandb
from rl.pufferlib.trainer import PufferTrainer
from rl.wandb.wandb_context import WandbContext
from util.eval_analyzer import print_policy_stats

global _consecutive_failures

class CarbsSweepRollout:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.run_id = None
        self.suggestion = None
        self.run_dir = None

        self._init_run()

    def _init_run(self):
        with WandbContext(self.cfg, name=self.cfg.run + ".init") as wandb_run:
            with open(os.path.join(self.cfg.run_dir, "sweep_config.yaml"), "w") as f:
                OmegaConf.save(self.cfg, f)
                wandb_run.save(os.path.join(self.cfg.run_dir, "*.yaml"), base_path=self.cfg.run_dir)

            with CarbsSweep(self.cfg.run_dir) as sweep_state:
                self.suggestion = sweep_state.carbs.suggest().suggestion

                sweep_state.num_suggestions += 1
                wandb_run.summary["num_suggestions"] = sweep_state.num_suggestions
                wandb_run.summary["num_failures"] = sweep_state.num_failures
                wandb_run.summary["num_observations"] = sweep_state.num_observations
                wandb_run.summary["run_id"] = self.run_id
                self.run_id = self.cfg.run + ".r." + str(sweep_state.num_suggestions)
                wandb_run.name = self.run_id

        self.run_dir = os.path.join(self.cfg.run_dir, "runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        with WandbContext(self.cfg, name=self.run_id) as wandb_run:
            print("Generated CARBS suggestion: ")
            print(yaml.dump(self.suggestion, default_flow_style=False))
            with open(os.path.join(self.cfg.run_dir, "carbs_suggestion.yaml"), "w") as f:
                    yaml.dump(self.suggestion, f)
                    wandb_run.save(os.path.join(self.cfg.run_dir, "*.yaml"), base_path=self.cfg.run_dir)

    def run(self):
        try:
            self._run()
        except Exception as e:
            print(f"Error running suggested rollout: {e}")
            Console().print_exception()
            with CarbsSweep(self.cfg.run_dir) as sweep_state:
                sweep_state.carbs.observe(
                    ObservationInParam(
                        input=self.suggestion,
                        output=0,
                        cost=0,
                        is_failure=True,
                    )
                )
                sweep_state.num_failures += 1


    def _run(self):
        start_time = time.time()
        train_cfg = deepcopy(self.cfg)
        train_cfg.sweep = {}

        train_cfg.run = self.run_id
        train_cfg.data_dir = os.path.join(self.cfg.run_dir, "runs")
        train_cfg.wandb.group = self.cfg.run

        eval_cfg = deepcopy(train_cfg)
        eval_cfg.eval = self.cfg.sweep.eval

        apply_carbs_suggestion(train_cfg, pow2_suggestion(self.cfg, self.suggestion))

        self._log_file("config.yaml", self.cfg)
        self._log_file("train_config.yaml", train_cfg)
        self._log_file("eval_config.yaml", eval_cfg)
        self._log_file("carbs_suggestion.yaml", self.suggestion)

        with WandbContext(train_cfg) as wandb_run:
            self._update_wandb_config(wandb_run)

            train_start_time = time.time()
            trainer = PufferTrainer(train_cfg, wandb_run)
            trainer.train()
            trainer.close()
            train_time = time.time() - train_start_time

            eval_start_time = time.time()
            policy_uri = trainer.policy_checkpoint.model_path
            policy = load_policy_from_uri(policy_uri, eval_cfg, wandb_run)
            policy.name = "final"
            initial_policy = load_policy_from_uri(trainer.checkpoints[0], eval_cfg, wandb_run)
            initial_policy.name = "initial"
            evaluator = PufferEvaluator(eval_cfg, policy, [initial_policy])
            stats = evaluator.evaluate()
            evaluator.close()
            eval_time = time.time() - eval_start_time

            print_policy_stats(stats, '1v1', 'all')
            print_policy_stats(stats, 'elo_1v1', 'altar')

            eval_metric = self._compute_objective(stats)

            wandb_run.log(
                {"eval_metric": eval_metric},
                step=trainer.policy_checkpoint.agent_steps)

            wandb_run.summary["training_time"] = train_time
            wandb_run.summary["eval_time"] = eval_time
            wandb_run.summary["eval_metric"] = eval_metric
            wandb_run.summary["agent_step"] = trainer.policy_checkpoint.agent_steps
            wandb_run.summary["epoch"] = trainer.policy_checkpoint.epoch

            print(f"Sweep Objective: {eval_metric}")
            print(f"Sweep Train Time: {trainer.train_time}")

            self._log_file("eval_stats.yaml", stats)
            self._log_file("eval_config.yaml", eval_cfg)
            self._log_file("eval_stats.txt", stats)
            print_policy_stats(stats, '1v1', 'all')
            print_policy_stats(stats, 'elo_1v1', 'altar')

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
                },
                artifact_type="sweep_model",
                additional_files=[
                    os.path.join(self.run_dir, f) for f in [
                        "train_config.yaml",
                        "eval_config.yaml",
                        "eval_stats.txt",
                        "eval_stats.yaml",
                        "carbs_suggestion.yaml",
                        "eval_config.yaml",
                    ]
                ]
            )
            final_model_artifact.link(
                self.cfg.run, [self.run_id]
            )

            total_time = time.time() - start_time
            with CarbsSweep(self.cfg.run_dir) as sweep_state:
                sweep_state.carbs.observe(
                    ObservationInParam(
                        input=self.suggestion,
                        output=eval_metric,
                        cost=total_time,
                        is_failure=False))
                sweep_state.num_observations += 1
            wandb_run.summary["total_time"] = total_time

    def _compute_objective(self, stats):
        sum = 0
        count = 0
        for game in stats:
            for agent in game:
                if agent["policy_name"] == "final":
                    sum += agent.get(self.cfg.sweep.metric, 0)
                    count += 1
        print(f"Sweep Metric: {self.cfg.sweep.metric} = {sum} / {count}")
        return sum / count

    def _log_file(self, name: str, data):
        with open(os.path.join(self.run_dir, name), "w") as f:
            if isinstance(data, OmegaConf):
                yaml.dump(OmegaConf.to_container(data, resolve=True), f)
            else:
                yaml.dump(data, f)

    def _update_wandb_config(self, wandb_run):
        wandb_run.config.__dict__["_locked"] = {}
        wandb_run.config.update(
            pow2_suggestion(self.cfg, self.suggestion),
            allow_val_change=True
        )

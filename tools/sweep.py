import os
import signal  # Aggressively exit on ctrl+c
from copy import deepcopy

import hydra
import wandb
import yaml
from omegaconf import OmegaConf
from rich import traceback
from rich.console import Console

from rl.carbs.util import (
    apply_carbs_suggestion,
    create_sweep_state,
    load_sweep_state,
    pow2_suggestion,
    save_sweep_state,
)
from carbs import ObservationInParam
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri
from rl.pufferlib.trainer import PufferTrainer
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from util.stats import print_policy_stats
from rl.pufferlib.policy import upload_policy_to_wandb

import base64

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

global _cfg
global _consecutive_failures
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    global _cfg
    _cfg = cfg

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)
    os.makedirs(cfg.run_dir, exist_ok=True)

    if os.path.exists(os.path.join(cfg.run_dir, "sweep.yaml")):
        sweep_state = load_sweep_state(cfg.run_dir)
    else:
        sweep_state = create_sweep_state(cfg)

    global _consecutive_failures
    _consecutive_failures = 0

    wandb.agent(sweep_state.wandb_sweep_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                function=run_carb_sweep_rollout,
                count=999999)


def run_carb_sweep_rollout():
    print("Running carb sweep rollout")
    global _cfg
    cfg = _cfg

    global _consecutive_failures
    if _consecutive_failures > 10:
        print("Too many consecutive failures, exiting")
        os._exit(0)

    sweep_state = load_sweep_state(cfg.run_dir)
    try:
        suggestion = sweep_state.carbs.suggest().suggestion
    except Exception as e:
        print(f"Error suggesting CARBS: {e}")
        Console().print_exception()
        os._exit(1)

    sweep_state.num_suggestions += 1
    save_sweep_state(cfg.run_dir, sweep_state)

    print("Generated CARBS suggestion: ")
    print(yaml.dump(suggestion, default_flow_style=False))
    with open(os.path.join(cfg.run_dir, "carbs_suggestion.yaml"), "w") as f:
        yaml.dump(suggestion, f)

    try:
        run_suggested_rollout(cfg, suggestion, sweep_state)
    except Exception as e:
        print(f"Error running suggested rollout: {e}")
        Console().print_exception()
        _consecutive_failures += 1
        sweep_state.carbs.observe(
            ObservationInParam(
                input=suggestion,
                output=0,
                cost=0,
                is_failure=True,
            )
        )
        sweep_state.num_failures += 1
        save_sweep_state(cfg.run_dir, sweep_state)
        _consecutive_failures = 0

def run_suggested_rollout(cfg, suggestion, sweep_state):
    train_cfg = deepcopy(cfg)
    train_cfg.sweep = {}
    run_id = sweep_state.num_suggestions
    train_cfg.run = cfg.run + ".r." + str(run_id)
    train_cfg.data_dir = os.path.join(cfg.run_dir, "runs")
    train_cfg.wandb.group = cfg.run
    apply_carbs_suggestion(train_cfg, pow2_suggestion(cfg, suggestion))

    os.makedirs(train_cfg.run_dir, exist_ok=True)

    eval_cfg = deepcopy(train_cfg)
    eval_cfg.eval = cfg.sweep.eval

    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(train_cfg, f)
    with open(os.path.join(train_cfg.run_dir, "train_config.yaml"), "w") as f:
        OmegaConf.save(train_cfg, f)
    with open(os.path.join(train_cfg.run_dir, "eval_config.yaml"), "w") as f:
        OmegaConf.save(eval_cfg, f)
    with open(os.path.join(train_cfg.run_dir, "carbs_suggestion.yaml"), "w") as f:
        yaml.dump(suggestion, f)

    with WandbContext(train_cfg) as wandb_run:
        wandb_run.config.__dict__["_locked"] = {}
        wandb_run.config.update(pow2_suggestion(cfg, suggestion), allow_val_change=True)
        trainer = PufferTrainer(train_cfg, wandb_run)
        trainer.train()
        trainer.close()

        policy_uri = trainer.policy_checkpoint.model_path
        policy = load_policy_from_uri(policy_uri, eval_cfg, wandb_run)
        evaluator = PufferEvaluator(eval_cfg, policy, [])
        stats = evaluator.evaluate()
        evaluator.close()

        print_policy_stats(stats)
        metric = stats[0].get(cfg.sweep.metric, None)
        print(f"Sweep Metric: {cfg.sweep.metric} = {metric}")
        objective = 0

        if metric is not None:
            objective = metric["sum"] / metric["count"]
            wandb_run.log(
                {"eval_metric": objective},
                step=trainer.policy_checkpoint.agent_steps)

        wandb_run.summary["num_suggestions"] = sweep_state.num_suggestions
        wandb_run.summary["num_failures"] = sweep_state.num_failures
        wandb_run.summary["num_observations"] = sweep_state.num_observations
        wandb_run.summary["training_time"] = trainer.train_time
        wandb_run.summary["eval_objective"] = objective
        wandb_run.summary["agent_step"] = trainer.policy_checkpoint.agent_steps
        wandb_run.summary["epoch"] = trainer.policy_checkpoint.epoch
        wandb_run.summary["run_id"] = run_id

        print(f"Sweep Objective: {objective}")
        print(f"Sweep Train Time: {trainer.train_time}")

        with open(os.path.join(train_cfg.run_dir, "eval_stats.yaml"), "w") as f:
            yaml.dump(stats, f)
        with open(os.path.join(train_cfg.run_dir, "eval_config.yaml"), "w") as f:
            OmegaConf.save(eval_cfg, f)
        with open(os.path.join(train_cfg.run_dir, "eval_stats.txt"), "w") as f:
            print_policy_stats(stats, file=f)

        final_model_artifact = upload_policy_to_wandb(
            wandb_run,
            trainer.policy_checkpoint.model_path,
            f"{train_cfg.run}.model",
            metadata={
                "training_run": train_cfg.run,
                "agent_step": trainer.policy_checkpoint.agent_steps,
                "epoch": trainer.policy_checkpoint.epoch,
                "training_time": trainer.train_time,
                "eval_objective": objective,
            },
            artifact_type="sweep_model",
            additional_files=[
                os.path.join(train_cfg.run_dir, f) for f in [
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
            cfg.run, [train_cfg.run]
        )

        sweep_state.carbs.observe(
            ObservationInParam(
                input=suggestion,
                output=objective,
                cost=trainer.train_time,
                is_failure=False,
            )
        )
        sweep_state.num_observations += 1
        save_sweep_state(cfg.run_dir, sweep_state)
if __name__ == "__main__":
    main()

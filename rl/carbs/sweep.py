import os
import signal  # Aggressively exit on ctrl+c
from copy import deepcopy

import hydra
import wandb
import yaml
from omegaconf import OmegaConf
from rich import traceback
from rl.carbs.util import (
    apply_carbs_suggestion,
    create_sweep_state_if_needed,
    CarbsSweep,
    pow2_suggestion,
    save_sweep_state,
)
from carbs import ObservationInParam
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri
from rl.pufferlib.trainer import PufferTrainer
from rl.wandb.wandb_context import WandbContext
from util.rng import init_rng
from util.stats import print_policy_stats
from rich.console import Console

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

global _cfg
@hydra.main(version_base=None, config_path="../../configs", config_name="sweep/sweep")
def main(cfg):
    global _cfg
    _cfg = cfg

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    init_rng(cfg.rng)
    os.makedirs(cfg.run_dir, exist_ok=True)

    if os.path.exists(os.path.join(cfg.run_dir, "sweep.yaml")):
        sweep_state = CarbsSweep(cfg.run_dir)
    else:
        sweep_state = create_sweep_state_if_needed(cfg)

    wandb.agent(sweep_state.wandb_sweep_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                function=run_carb_sweep_rollout,
                count=999999)


def run_carb_sweep_rollout():
    print("Running carb sweep rollout")
    global _cfg
    cfg = _cfg

    sweep_state = CarbsSweep(cfg.run_dir)
    suggestion = sweep_state.carbs.suggest().suggestion
    sweep_state.num_suggestions += 1
    save_sweep_state(cfg.run_dir, sweep_state)

    print("Generated CARBS suggestion: ")
    print(yaml.dump(suggestion, default_flow_style=False))

    try:
        run_suggested_rollout(cfg, suggestion, sweep_state)
    except Exception as e:
        print(f"Error running suggested rollout: {e}")
        Console().print_exception()
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

def run_suggested_rollout(cfg, suggestion, sweep_state):
    train_cfg = deepcopy(cfg)
    train_cfg.run = cfg.run + ".r." + str(sweep_state.num_suggestions)
    train_cfg.data_dir = os.path.join(cfg.run_dir, "runs")
    apply_carbs_suggestion(train_cfg, pow2_suggestion(cfg, suggestion))
    print("Generated train config: ")
    print(OmegaConf.to_yaml(train_cfg))

    os.makedirs(train_cfg.run_dir, exist_ok=True)

    with open(os.path.join(train_cfg.run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(train_cfg, f)

    with WandbContext(train_cfg) as wandb_run:
        wandb_run.config.__dict__["_locked"] = {}
        wandb_run.config.update(pow2_suggestion(cfg, suggestion), allow_val_change=True)
        trainer = PufferTrainer(train_cfg, wandb_run)
        trainer.train()

        eval_cfg = deepcopy(train_cfg)
        eval_cfg.eval = cfg.sweep.eval
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
            wandb.log({"eval_metric": objective})

    print(f"Sweep Objective: {objective}")
    print(f"Sweep Train Time: {trainer.train_time}")
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

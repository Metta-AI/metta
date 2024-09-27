
from json import load
import math
import time
import traceback
from math import ceil, floor, log
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.dashboard.dashboard import Dashboard
from rl.pufferlib.train import PufferTrainer
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf

import wandb
from rl.wandb.wandb import init_wandb
from wandb.errors import CommError

global _cfg
global _dashboard
def run_sweep(cfg: OmegaConf, dashboard: Dashboard):

    sweep_id = None
    if cfg.sweep.resume:
        try:
            print(f"Loading previous sweep {cfg.experiment}...")
            artifact = wandb.use_artifact(cfg.experiment + ":latest", type="sweep")
            sweep_id = artifact.metadata["sweep_id"]
        except CommError:
            print(f"No previous sweep {cfg.experiment} found, creating...")

    if sweep_id is None:
        print(f"Creating new sweep {cfg.experiment}...")
        sweep_id = wandb.sweep(
            sweep=_wandb_sweep_cfg(cfg),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
        )
        _save_carbs_state(_init_carbs(cfg), cfg.experiment, sweep_id)

    global _cfg
    _cfg = cfg
    wandb.agent(sweep_id, function=run_carb_sweep_rollout,
                entity=cfg.wandb.entity, project=cfg.wandb.project, count=10000)

def run_carb_sweep_rollout():
    global _cfg
    global _dashboard
    init_wandb(_cfg)
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    carbs_controller = _load_carbs_state(_cfg, _dashboard)

    print(f"CARBS: obs: {carbs_controller.observation_count}")
    orig_suggestion = carbs_controller.suggest().suggestion
    carbs_controller.num_suggestions += 1

    suggestion = orig_suggestion.copy()
    del suggestion["suggestion_uuid"]
    print(f"Carbs Suggestion: {suggestion}")
    wandb.config.__dict__["_locked"] = {}

    new_cfg = _cfg.copy()
    for key, value in suggestion.items():
        new_cfg_param = new_cfg
        sweep_param = _cfg.sweep.parameters
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            new_cfg_param = new_cfg_param[k]
            sweep_param = sweep_param[k]
        param_name = key_parts[-1]
        if sweep_param[param_name].space == "pow2":
            value = 2**value
            suggestion[key] = value
        new_cfg_param[param_name] = value
    wandb.config.update(suggestion, allow_val_change=True)
    print(f"Sweep Params: {suggestion}")
    _save_carbs_state(carbs_controller, _cfg.experiment, wandb.run.sweep_id)
    wandb.finish(quiet=True)

    sweep_experiment = f"{new_cfg.experiment}-{carbs_controller.num_suggestions}"
    new_cfg.experiment = sweep_experiment
    new_cfg.train.resume = False
    init_wandb(new_cfg)
    objective = 0
    train_time = 0
    is_failure = False
    trainer = None
    try:
        trainer = PufferTrainer(new_cfg, _dashboard)
        trainer.train()
        trainer.close()
        train_time = trainer.train_time
        model_artifact_name = trainer._upload_model_to_wandb()
        model_uri = f"wandb://{model_artifact_name}"
        eval_cfg = new_cfg.copy()
        eval_cfg.eval = _cfg.sweep.eval.copy()
        eval_cfg.eval.policy_uri = model_uri
        eval_cfg.wandb.track = False
        stats = evaluate(eval_cfg, _dashboard)

        metric = stats[0].get(_cfg.sweep.metric, {"sum": 0, "count": 1})
        objective = metric["sum"] / metric["count"]

    except Exception:
        if trainer is not None:
            trainer.close()
        is_failure = True
        print(traceback.format_exc())
        traceback.print_exc()
    wandb.finish(quiet=True)


    init_wandb(_cfg)
    carbs_controller = _load_carbs_state(_cfg, _dashboard)
    carbs_controller.observe(
        ObservationInParam(
            input=orig_suggestion,
            output=objective,
            cost=train_time,
            is_failure=is_failure,
        )
    )
    _save_carbs_state(carbs_controller, _cfg.experiment, wandb.run.sweep_id)
    _dashboard.update_carbs(
        num_observations=carbs_controller.observation_count,
        num_suggestions=carbs_controller.num_suggestions,
        last_metric=objective,
        last_run_time=train_time,
        last_run_success=not is_failure,
        num_failures=len(carbs_controller.failure_observations),
    )
    wandb.finish(quiet=True)

def _save_carbs_state(carbs_controller, experiment, sweep_id):
    artifact = wandb.Artifact(
        experiment,
        type="sweep",
        metadata={
            "sweep_id": sweep_id,
            "num_observations": carbs_controller.observation_count,
            "num_suggestions": carbs_controller.num_suggestions,
        })
    with artifact.new_file("carbs_state") as f:
        f.write(carbs_controller.serialize())
    artifact.save()

def _load_carbs_state(cfg: OmegaConf, dashboard: Dashboard):
    init_wandb(cfg)
    artifact = wandb.use_artifact(cfg.experiment + ":latest", type="sweep")
    carbs_state = artifact.file(wandb.run.dir + "/carbs_state")
    carbs = None
    with open(carbs_state, "rb") as f:
        carbs = CARBS.load_from_string(f.read())
        carbs.num_suggestions = artifact.metadata.get("num_suggestions", carbs.observation_count)
        dashboard.update_carbs(
            carbs.observation_count,
            carbs.num_suggestions,
            0,
            0,
            0,
            len(carbs.failure_observations),
        )
    return carbs


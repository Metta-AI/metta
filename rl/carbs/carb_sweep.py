
from json import load
import math
import time
import traceback
from math import ceil, floor, log
from rl.pufferlib.evaluate import evaluate
from rl.pufferlib.dashboard import Dashboard
from rl.pufferlib.train import PufferTrainer
import numpy as np
import torch
from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    Param,
)
from omegaconf import DictConfig, OmegaConf

import wandb
from rl.wandb.wandb import init_wandb
from wandb.errors import CommError

global _cfg
global _dashboard
def run_sweep(cfg: OmegaConf, dashboard: Dashboard):
    init_wandb(cfg)
    global _dashboard
    _dashboard = dashboard

    sweep_id = None
    if cfg.sweep.resume:
        try:
            _dashboard.log(f"Loading previous sweep {cfg.experiment}...")
            artifact = wandb.use_artifact(cfg.experiment + ":latest", type="sweep")
            sweep_id = artifact.metadata["sweep_id"]
        except CommError:
            _dashboard.log(f"No previous sweep {cfg.experiment} found, creating...")

    if sweep_id is None:
        _dashboard.log(f"Creating new sweep {cfg.experiment}...")
        sweep_id = wandb.sweep(
            sweep=_wandb_sweep_cfg(cfg),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
        )
        _save_carbs_state(_init_carbs(cfg), cfg.experiment, sweep_id)

    global _cfg
    _cfg = cfg
    wandb.finish(quiet=True)
    wandb.agent(sweep_id, function=run_carb_sweep_rollout,
                entity=cfg.wandb.entity, project=cfg.wandb.project, count=10000)

def run_carb_sweep_rollout():
    global _cfg
    global _dashboard
    init_wandb(_cfg)
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    carbs_controller = _load_carbs_state(_cfg)
    carbs_controller._set_seed(int(time.time()))

    _dashboard.log(f"CARBS: obs: {carbs_controller.observation_count}")
    orig_suggestion = carbs_controller.suggest().suggestion
    carbs_controller.num_suggestions += 1

    suggestion = orig_suggestion.copy()
    del suggestion["suggestion_uuid"]
    _dashboard.log(f"Carbs Suggestion: {suggestion}")
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
    _dashboard.log(f"Sweep Params: {suggestion}")
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
        stats = evaluate(eval_cfg)

        metric = stats[0].get(_cfg.sweep.metric, {"sum": 0, "count": 1})
        objective = metric["sum"] / metric["count"]

    except Exception:
        if trainer is not None:
            trainer.close()
        is_failure = True
        _dashboard.log(traceback.format_exc())
        traceback.print_exc()
    wandb.finish(quiet=True)


    init_wandb(_cfg)
    carbs_controller = _load_carbs_state(_cfg)
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

def _wandb_distribution(param):
    if param.space == "log":
        return "log_uniform_values"
    elif param.space == "linear":
        return "uniform"
    elif param.space == "logit":
        return "uniform"
    elif param.space == "pow2":
        return "int_uniform"
    elif param.space == "linear":
        if param.is_int:
            return "int_uniform"
        else:
            return "uniform"

_carbs_space = {
    "log": LogSpace,
    "linear": LinearSpace,
    "pow2": LinearSpace,
    "logit": LogitSpace,
}

def _carbs_params_spaces(cfg: OmegaConf):
    param_spaces = []
    params = _fully_qualified_parameters(cfg.sweep.parameters)
    for param_name, param in params.items():
        train_cfg_param = cfg
        if "search_center" not in param:
            for k in param_name.split("."):
                train_cfg_param = train_cfg_param[k]
            OmegaConf.set_struct(param, False)
            param.search_center = train_cfg_param
            OmegaConf.set_struct(param, True)

        if param.space == "pow2":
            param.min = int(math.log2(param.min))
            param.max = int(math.log2(param.max))
            param.search_center = int(math.log2(param.search_center))

        scale = param.get("scale", 1)
        if param.space == "pow2" or param.get("is_int", False):
            scale = 4

        if param.search_center < param.min or param.search_center > param.max:
            raise ValueError(f"Search center for {param_name}: {param.search_center} is not in range [{param.min}, {param.max}]")

        param_spaces.append(
            Param(
                name=param_name,
                space=_carbs_space[param.space](
                    min=param.min,
                    max=param.max,
                    is_integer=param.get("is_int", False) or param.space == "pow2",
                    rounding_factor=param.get("rounding_factor", 1),
                    scale=scale,
                ),
                search_center=param.search_center,                )
            )
    return param_spaces


def _fully_qualified_parameters(nested_dict, prefix=''):
    qualified_params = {}
    if "space" in nested_dict:
        return {prefix: nested_dict}
    for key, value in nested_dict.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, DictConfig):
            qualified_params.update(_fully_qualified_parameters(value, new_prefix))
    return qualified_params

def _wandb_sweep_cfg(cfg: OmegaConf):
    params = _fully_qualified_parameters(cfg.sweep.parameters)
    wandb_sweep_cfg = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "environment/" + cfg.sweep.metric,
        },
        "parameters": {},
        "name": cfg.wandb.name,
    }
    for param_name, param in params.items():
        wandb_sweep_cfg["parameters"][param_name] = {
            "min": param.min,
            "max": param.max,
            "distribution": _wandb_distribution(param),
        }
    return wandb_sweep_cfg

def _init_carbs(cfg: OmegaConf):
        param_spaces = _carbs_params_spaces(cfg)
        carbs_params = CARBSParams(
            better_direction_sign=1,
            resample_frequency=5,
            num_random_samples=len(param_spaces),
            checkpoint_dir=f"{cfg.data_dir}/{cfg.experiment}/carbs/",
            is_wandb_logging_enabled=False,
        )
        carbs = CARBS(carbs_params, param_spaces)
        carbs.num_suggestions = 0
        return carbs

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

def _load_carbs_state(cfg: OmegaConf):
    init_wandb(cfg)
    artifact = wandb.use_artifact(cfg.experiment + ":latest", type="sweep")
    carbs_state = artifact.file(wandb.run.dir + "/carbs_state")
    with open(carbs_state, "rb") as f:
        carbs = CARBS.load_from_string(f.read())
        carbs.num_suggestions = artifact.metadata.get("num_suggestions", carbs.observation_count)
        return carbs

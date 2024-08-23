
from omegaconf import OmegaConf, DictConfig

from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import Param

import numpy as np
import math

import torch
import time
from math import log, ceil, floor

from carbs import CARBS
from carbs import CARBSParams
from carbs import ObservationInParam

from rl.wandb.wandb import init_wandb
import traceback
import hydra
import wandb

_carbs_controller = None
_cfg = None
_sweep_id = None
_sweep_run_id = 0

def run_sweep(cfg: OmegaConf):
    global _cfg
    _cfg = cfg

    sweep_id = wandb.sweep(
        sweep=_wandb_sweep_cfg(cfg),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
    )
    global _sweep_id
    _sweep_id = sweep_id

    param_spaces = _carbs_params_spaces(cfg)
    print("Param Spaces:", param_spaces)
    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=5,
        num_random_samples=len(param_spaces),
    )
    global _carbs_controller
    _carbs_controller = CARBS(carbs_params, param_spaces)
    wandb.agent(sweep_id, run_carb_sweep_rollout, count=100)

def run_carb_sweep_rollout():
    global _carbs_controller
    global _cfg
    global _sweep_run_id

    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    init_wandb(_cfg)
    wandb.run.name = f"{wandb.run.name}.r_{_sweep_run_id:04d}"
    _sweep_run_id += 1

    wandb.config.__dict__["_locked"] = {}

    orig_suggestion = _carbs_controller.suggest().suggestion
    suggestion = orig_suggestion.copy()
    print("Carbs Suggestion:", suggestion)

    new_cfg = _cfg.copy()
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        new_cfg_param = new_cfg
        sweep_param = _cfg.sweep.parameters
        key_parts = key.split(".")
        for k in key_parts[:-1]:
            new_cfg_param = new_cfg_param[k]
            sweep_param = sweep_param[k]
        param_name = key_parts[-1]
        if sweep_param[param_name].space == "pow2":
            value = 2**value
        new_cfg_param[param_name] = value

    print(OmegaConf.to_yaml(new_cfg))

    observed_value = 0
    train_time = 0

    is_failure = False
    try:
        rl_controller = hydra.utils.instantiate(new_cfg.framework, new_cfg, _recursive_=False)
        rl_controller.train()
        observed_value = rl_controller.last_stats[_cfg.sweep.metric]
        train_time = rl_controller.train_time
    except Exception:
        is_failure = True
        traceback.print_exc()

    try:
        rl_controller.close()
    except Exception:
        print("Failed to close controller")

    print("Observed Value:", observed_value)
    print("Train Time:", train_time)
    print("Is Failure:", is_failure)

    _carbs_controller.observe(
        ObservationInParam(
            input=orig_suggestion,
            output=observed_value,
            cost=train_time,
            is_failure=is_failure,
        )
    )

def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return int(2**min(possible_results, key= lambda z: abs(x-2**z)))

def _wandb_distribution(param):
    if param.space == "log":
        return "log_uniform_values"
    elif param.space == "linear":
        return "uniform"
    elif param.space == "logit":
        return "uniform"
    elif param.space == "pow2":
        return "log_uniform_values"
    elif param.space == "linear":
        if param.is_int:
            return "int_uniform"
        else:
            return "uniform"

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
        if param.space == "pow2":
            param.min = int(math.log2(param.min))
            param.max = int(math.log2(param.max))
            if "search_center" in param:
                param.search_center = int(math.log2(param.search_center))

        param_spaces.append(
            Param(
                name=param_name,
                space=_carbs_space[param.space](
                    min=param.min,
                    max=param.max,
                    is_integer=param.get("is_int", False) or param.space == "pow2",
                    rounding_factor=param.get("rounding_factor", 1),
                    scale=param.get("scale", 1),
                ),
                search_center=param.get(
                    "search_center",
                    param.min + (param.max - param.min) / 2,
                )
            ))
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

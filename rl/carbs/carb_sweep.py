
from click import group
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
from carbs import ObservationInParam, WandbLoggingParams

from rl.pufferlib import checkpoint
from rl.wandb.wandb import init_wandb
import traceback
import hydra
import wandb

def run_sweep(cfg: OmegaConf):
    param_spaces = _carbs_params_spaces(cfg)
    print("Param Spaces:", param_spaces)
    carbs_params = CARBSParams(
        better_direction_sign=1,
        resample_frequency=5,
        num_random_samples=len(param_spaces),
        checkpoint_dir=f"{cfg.data_dir}/{cfg.experiment}/carbs/",
        is_wandb_logging_enabled=cfg.wandb.track,
        wandb_params=WandbLoggingParams(
            project_name = cfg.wandb.project,
            group_name = cfg.wandb.group,
            run_id = cfg.experiment or wandb.util.generate_id(),
            run_name = cfg.wandb.name,
            root_dir = "wandb",
        ),
    )
    carbs_controller = CARBS(carbs_params, param_spaces)

    while True:
        run_carb_sweep_rollout(carbs_controller, cfg)

def run_carb_sweep_rollout(carbs_controller: CARBS, cfg: OmegaConf):
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    orig_suggestion = carbs_controller.suggest().suggestion
    suggestion = orig_suggestion.copy()
    print("Carbs Suggestion:", suggestion)

    new_cfg = cfg.copy()
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue
        new_cfg_param = new_cfg
        sweep_param = cfg.sweep.parameters
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
        observed_value = rl_controller.last_stats[cfg.sweep.metric]
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

    carbs_controller.observe(
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

        scale = param.get("scale", 1)
        if param.space == "pow2" or param.get("is_int", False):
            scale = 4

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


from colorama import init
from omegaconf import OmegaConf


import numpy as np

from carbs import LinearSpace
from carbs import Param

import torch
import time
from math import log, ceil, floor

from carbs import CARBS
from carbs import CARBSParams
from carbs import ObservationInParam

from rl.wandb.wandb import init_wandb
from rl.rl_framework import RLFramework
import traceback
import hydra
import wandb

_carbs_controller = None

def run_sweep(cfg: OmegaConf):
    sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
    init_wandb(cfg)

    sweep_id = wandb.sweep(
        sweep=sweep,
        project=cfg.wandb.project,
    )
    param_spaces = []
    for param_name, param in cfg.sweep.parameters.items():
        param_spaces.append(
            Param(
                name=param_name,
                space=LinearSpace(
                    min=param.min,
                    max=param.max,
                    is_integer=False,
                    rounding_factor=1,
                    scale=1,
                ),
                search_center=param.min + (param.max - param.min) / 2,
            ))

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

    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))

    wandb.config.__dict__['_locked'] = {}
    cfg = OmegaConf.create(dict(wandb.config))

    orig_suggestion = _carbs_controller.suggest().suggestion
    suggestion = orig_suggestion.copy()
    print('Suggestion:', suggestion)
    if "batch_size" in suggestion:
        suggestion["batch_size"] = closest_power(suggestion["batch_size"])
    if "minibatch_size" in suggestion:
        suggestion["minibatch_size"] = closest_power(suggestion["minibatch_size"])
    if "bptt_horizon" in suggestion:
        suggestion["bptt_horizon"] = closest_power(suggestion["bptt_horizon"])
    if "forward_pass_minibatch_target_size" in suggestion:
        suggestion["forward_pass_minibatch_target_size"] = closest_power(suggestion["forward_pass_minibatch_target_size"])

    cfg["framework"]["pufferlib"]["train"].update(suggestion)
    observed_value = 0
    train_time = 0

    try:
        rl_controller = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)
        rl_controller.train()
        observed_value = rl_controller.stats.get('environment/episode/reward.mean', 0)
        train_time = rl_controller.train_time
    except Exception:
        is_failure = True
        traceback.print_exc()

    print('Observed Value:', observed_value)
    print('Train Time:', train_time)
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


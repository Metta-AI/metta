import logging
import math
import time
from collections import defaultdict

import numpy as np
from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    Param,
)
from omegaconf import DictConfig, OmegaConf
from wandb_carbs import Pow2WandbCarbs

logger = logging.getLogger("sweep_rollout")

_carbs_space = {
    "log": LogSpace,
    "linear": LinearSpace,
    "pow2": LinearSpace,
    "logit": LogitSpace,
}


def carbs_params_from_cfg(cfg: OmegaConf):
    param_spaces = []
    pow2_params = set()
    params = _fully_qualified_parameters(cfg.sweep)
    for param_name, param in params.items():
        train_cfg_param = cfg
        if param.search_center is None:
            for k in param_name.split("."):
                train_cfg_param = train_cfg_param[k]
            OmegaConf.set_struct(param, False)
            param.search_center = train_cfg_param
            OmegaConf.set_struct(param, True)

        if param.space == "pow2":
            try:
                param.min = int(math.log2(param.min))
                param.max = int(math.log2(param.max))
                param.search_center = int(math.log2(param.search_center))
            except Exception as e:
                print(
                    f"Error setting pow2 params for {param_name}=({param.min}, {param.max}, {param.search_center}): {e}"
                )
                raise e
            pow2_params.add(param_name)
        scale = param.get("scale", 1)

        if param.space == "pow2" or param.get("is_int", False):
            scale = 4
        if param.search_center < param.min or param.search_center > param.max:
            raise ValueError(
                f"Search center for {param_name}: {param.search_center} is not in range [{param.min}, {param.max}]"
            )

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
                search_center=param.search_center,
            )
        )
    return param_spaces, pow2_params


def _fully_qualified_parameters(nested_dict, prefix=""):
    qualified_params = {}
    if "space" in nested_dict:
        return {prefix: nested_dict}
    for key, value in nested_dict.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, DictConfig):
            qualified_params.update(_fully_qualified_parameters(value, new_prefix))
    return qualified_params


class MettaCarbs(Pow2WandbCarbs):
    def __init__(self, cfg: OmegaConf, run):
        self.cfg = cfg
        carbs_params, pow2_params = carbs_params_from_cfg(cfg)
        self.generation = 0
        self.runs = []

        super().__init__(
            CARBS(
                CARBSParams(
                    better_direction_sign=1,
                    resample_frequency=5,
                    num_random_samples=cfg.num_random_samples,
                    is_wandb_logging_enabled=False,
                    seed=int(time.time()),
                    is_saved_on_every_observation=False,
                    checkpoint_dir=None,
                ),
                carbs_params,
            ),
            pow2_params,
            run,
        )

    def _get_runs_from_wandb(self):
        runs = super()._get_runs_from_wandb()
        if not hasattr(self.cfg, "generation") or not self.cfg.generation.enabled:
            return runs

        generations = defaultdict(list)
        for run in runs:
            if run.summary["carbs.state"] == "success":
                generation = run.summary.get("generation", 0)
                generations[generation].append(run)
        max_gen = 0
        if len(generations) > 0:
            max_gen = max(generations.keys())
        if len(generations[max_gen]) < self.cfg.generation.min_samples:
            self.generation = max_gen
            logger.info(
                f"Updating newest generation: {self.generation} with {len(generations[self.generation])} samples"
            )
        elif np.random.random() >= self.cfg.generation.regen_pct:
            self.generation = max_gen + 1
            logger.info(f"New creating a new generation: {self.generation}")
        else:
            self.generation = np.random.randint(max_gen + 1)
            logger.info(f"Updating generation: {self.generation} with {len(generations[self.generation])} samples")

        self.runs = [run for run in runs if run.summary.get("generation", 0) == self.generation]
        return self.runs

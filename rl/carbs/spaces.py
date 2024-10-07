import math

from carbs import (
    LinearSpace,
    LogitSpace,
    LogSpace,
    Param,
)
from carbs import (
    CARBS,
    CARBSParams,
)

from wandb_carbs import WandbCarbs
from typing import Set
from omegaconf import DictConfig, OmegaConf
import time
_carbs_space = {
    "log": LogSpace,
    "linear": LinearSpace,
    "pow2": LinearSpace,
    "logit": LogitSpace,
}

def carbs_params_from_cfg(cfg: OmegaConf):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    param_spaces = []
    pow2_params = set()
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
            try:
                param.min = int(math.log2(param.min))
                param.max = int(math.log2(param.max))
                param.search_center = int(math.log2(param.search_center))
            except Exception as e:
                print(f"Error setting pow2 params for {param_name}=({param.min}, {param.max}, {param.search_center}): {e}")
                raise e
            pow2_params.add(param_name)
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
                search_center=param.search_center,
            )
        )
    return param_spaces, pow2_params


def _fully_qualified_parameters(nested_dict, prefix=''):
    qualified_params = {}
    if "space" in nested_dict:
        return {prefix: nested_dict}
    for key, value in nested_dict.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, DictConfig):
            qualified_params.update(_fully_qualified_parameters(value, new_prefix))
    return qualified_params


class Pow2WandbCarbs(WandbCarbs):
    def __init__(self, cfg: OmegaConf, wandb_run, pow2_params: Set[str]):
        self.pow2_params = pow2_params
        super().__init__(cfg, wandb_run)

    def suggest(self):
        suggestion = super().suggest()
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                suggestion[param.name] = 2 ** suggestion[param.name]
        return suggestion

    def _suggestion_from_run(self, run):
        suggestion = super()._suggestion_from_run(run)
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                suggestion[param.name] = int(math.log2(suggestion[param.name]))
        return suggestion

def carbs_from_cfg(cfg: OmegaConf, run) -> Pow2WandbCarbs:
    carbs_params, pow2_params = carbs_params_from_cfg(cfg)
    return Pow2WandbCarbs(
        CARBS(
            CARBSParams(
                better_direction_sign=1,
                resample_frequency=5,
                num_random_samples=cfg.sweep.num_random_samples,
                checkpoint_dir=f"{cfg.run_dir}/carbs/",
                is_wandb_logging_enabled=False,
                seed=int(time.time()),
            ),
            carbs_params
        ),
        run,
        pow2_params
    )

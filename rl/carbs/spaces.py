import math

from carbs import (
    LinearSpace,
    LogitSpace,
    LogSpace,
    Param,
)
from omegaconf import DictConfig, OmegaConf

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
                search_center=param.search_center,
            )
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

def wandb_sweep_cfg(cfg: OmegaConf):
    params = _fully_qualified_parameters(cfg.sweep.parameters)
    wandb_sweep_cfg = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "eval_metric",
        },
        "parameters": {},
        "name": cfg.run,
    }
    for param_name, param in params.items():
        wandb_sweep_cfg["parameters"][param_name] = {
            "min": param.min,
            "max": param.max,
            "distribution": _wandb_distribution(param),
        }
    return wandb_sweep_cfg

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


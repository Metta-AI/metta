import json
import os
from typing import Any

from pydantic import ValidationError

from metta.rl.trainer_config import TrainerConfig


def save_train_job_override_config(cfg: Any, overrides: dict[str, Any]) -> str:
    """Save a train job config overrides file.

    Args:
        cfg: The config to save the overrides for (TrainToolConfig or dict)
        overrides: The overrides to save

    Returns:
        The path to the saved overrides file
    """
    # Get run_dir from config
    if hasattr(cfg, "run_dir"):
        run_dir = cfg.run_dir
    elif isinstance(cfg, dict):
        run_dir = cfg.get("run_dir", "./train_dir")
    else:
        run_dir = "./train_dir"

    save_path = os.path.join(run_dir, "train_config_overrides.json")
    with open(save_path, "w") as f:
        json.dump(overrides, f, indent=2)
    return save_path


def merge_train_job_config_overrides(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge two train job configs. Note: the overrides take precedence over the base config.

    Args:
        base_cfg: The base config to merge into
        overrides: The overrides to merge into the base config

    Returns:
        The merged config
    """
    import copy

    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base_cfg, overrides)


def validate_train_job_config(trainer_cfg: TrainerConfig) -> TrainerConfig:
    """Validate a train job config.

    Args:
        trainer_cfg: The trainer config to validate

    Returns:
        The validated config
    """
    try:
        # The TrainerConfig is already a pydantic model, so validation happens automatically
        # Just ensure it's valid by accessing a field
        _ = trainer_cfg.total_timesteps
        return trainer_cfg
    except (ValueError, TypeError, ValidationError) as e:
        raise ValueError("Invalid trainer config") from e


def load_train_job_config_with_overrides(cfg: Any) -> Any:
    """
    Load a train job config with overrides.
    Overrides are expected to be in the run_dir as `train_config_overrides.json`.

    Args:
        cfg: The base config to load (TrainToolConfig or dict)

    Returns:
        The loaded config with overrides applied
    """
    # Get run_dir from config
    if hasattr(cfg, "run_dir"):
        run_dir = cfg.run_dir
    elif isinstance(cfg, dict):
        run_dir = cfg.get("run_dir", "./train_dir")
    else:
        run_dir = "./train_dir"

    overrides_path = os.path.join(run_dir, "train_config_overrides.json")
    if os.path.exists(overrides_path):
        with open(overrides_path, "r") as f:
            override_dict = json.load(f)

        if isinstance(cfg, dict):
            # It's already a dict
            return merge_train_job_config_overrides(cfg, override_dict)
        elif hasattr(cfg, "model_dump"):
            # It's a pydantic model
            base_dict = cfg.model_dump()
            merged_dict = merge_train_job_config_overrides(base_dict, override_dict)
            # Create a new instance with merged config
            return type(cfg).model_validate(merged_dict)

    return cfg

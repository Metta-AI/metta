import os
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ValidationError

from metta.common.util.config import copy_omegaconf_config
from metta.rl.trainer_config import create_trainer_config


def save_train_job_override_config(
    cfg: DictConfig | ListConfig, overrides: DictConfig | ListConfig | dict[str, Any]
) -> str:
    """Save a train job config overrides file.

    Args:
        cfg: The config to save the overrides for
        overrides: The overrides to save

    Returns:
        The path to the saved overrides file
    """
    save_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    OmegaConf.save(overrides, save_path)
    return save_path


def merge_train_job_config_overrides(
    base_cfg: DictConfig | ListConfig, overrides: DictConfig | ListConfig | dict[str, Any]
) -> DictConfig | ListConfig:
    """Merge two train job configs. Note: the overrides take precedence over the base config.

    Args:
        base_cfg: The base config to merge into
        overrides: The overrides to merge into the base config

    Returns:
        The merged config
    """
    cfg_copy = copy_omegaconf_config(base_cfg)

    OmegaConf.set_struct(cfg_copy, False)
    merged_cfg: DictConfig | ListConfig = OmegaConf.merge(cfg_copy, overrides)
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


def validate_train_job_config(cfg: DictConfig) -> DictConfig:
    """Validate a train job config.

    Args:
        cfg: The config to validate

    Returns:
        The validated config
    """
    try:
        create_trainer_config(cfg)  # type: ignore[arg-type]
    except (ValueError, TypeError, ValidationError) as e:
        raise ValueError("Invalid trainer config") from e
    return cfg


def load_train_job_config_with_overrides(cfg: DictConfig) -> DictConfig:
    """
    Load a train job config with overrides.
    Overrides are expected to be in the run_dir as `train_config_overrides.yaml`.

    Args:
        cfg: The base config to load

    Returns:
        The loaded config
    """
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        override_cfg = OmegaConf.load(overrides_path)

        # Since sweep_job mimics train_job.yaml, just merge them directly
        cfg = merge_train_job_config_overrides(cfg, override_cfg)
    return cfg

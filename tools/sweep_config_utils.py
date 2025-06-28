import os
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ValidationError

from metta.common.util.config import copy_omegaconf_config
from metta.rl.trainer_config import parse_trainer_config


def validate_merged_config(
    base_cfg: DictConfig | ListConfig, overrides: DictConfig | ListConfig | dict[str, Any]
) -> DictConfig | ListConfig:
    """Validates that an input train_job config is valid after applying overrides."""
    cfg_copy = copy_omegaconf_config(base_cfg)

    OmegaConf.set_struct(cfg_copy, False)
    merged_cfg: DictConfig | ListConfig = OmegaConf.merge(cfg_copy, overrides)
    OmegaConf.set_struct(merged_cfg, True)

    if "trainer" in merged_cfg:
        try:
            _ = parse_trainer_config(merged_cfg.trainer)
        except (ValueError, TypeError, ValidationError) as e:
            raise ValueError("Invalid trainer config after applying overrides") from e

    return merged_cfg


def save_train_job_override_config(
    cfg: DictConfig | ListConfig, overrides: DictConfig | ListConfig | dict[str, Any]
) -> str:
    save_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    OmegaConf.save(overrides, save_path)
    return save_path


def apply_carbs_suggestion(config: DictConfig | ListConfig, suggestion: DictConfig) -> None:
    """Apply suggestions to a configuration object using dotted path notation.

    Args:
        config: The configuration object to modify
        suggestion: The suggestions to apply
    """
    for key, value in suggestion.items():
        if key == "suggestion_uuid":
            continue

        # Convert key to string if it's not already
        str_key = str(key) if not isinstance(key, str) else key

        # Use OmegaConf.update with the string key
        OmegaConf.update(config, str_key, value)


def load_train_job_config_with_overrides(cfg: DictConfig | ListConfig) -> DictConfig | ListConfig:
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        override_cfg = OmegaConf.load(overrides_path)
        cfg = validate_merged_config(cfg, override_cfg)
    return cfg

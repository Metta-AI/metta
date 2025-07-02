import os
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ValidationError

from metta.common.util.config import copy_omegaconf_config
from metta.common.util.logging import setup_mettagrid_logger
from metta.rl.trainer_config import parse_trainer_config

# ===== Legacy functions kept for backwards compatibility and tests =====


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


# ===== Main function used by the sweep process =====


def load_train_job_config_with_overrides(cfg: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Load training config with sweep overrides applied.

    Priority order (highest to lowest):
    1. Raw Protein suggestions (protein_suggestions.yaml)
    2. Sweep job overrides (train_config_overrides.yaml)
    3. Base config
    """
    logger = setup_mettagrid_logger("sweep_config")

    # Step 1: Load sweep job overrides if they exist
    overrides_path = os.path.join(cfg.run_dir, "train_config_overrides.yaml")
    if os.path.exists(overrides_path):
        override_cfg = OmegaConf.load(overrides_path)

        # Handle sweep_job structure (extract trainer, agent, evals)
        if "trainer" in override_cfg and "agent" in override_cfg and "evals" in override_cfg:
            train_overrides = {
                "trainer": override_cfg.trainer,
                "agent": override_cfg.agent,
                "train_job": {"evals": override_cfg.evals},
            }
            if "device" in override_cfg:
                train_overrides["device"] = override_cfg.device

            # Apply overrides
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, train_overrides)
            OmegaConf.set_struct(cfg, True)

    # Step 2: Apply raw Protein suggestions with highest priority
    protein_suggestions_path = os.path.join(cfg.run_dir, "protein_suggestions.yaml")
    if os.path.exists(protein_suggestions_path):
        protein_suggestions = OmegaConf.load(protein_suggestions_path)
        logger.info(f"Applying Protein suggestions from {protein_suggestions_path}")

        # Apply with force - Protein always wins
        OmegaConf.set_struct(cfg, False)
        for key, value in protein_suggestions.items():
            if key == "suggestion_uuid":
                continue
            if key in cfg and isinstance(value, dict):
                cfg[key] = OmegaConf.unsafe_merge(cfg[key], value)
            else:
                cfg[key] = value
        OmegaConf.set_struct(cfg, True)

        # Log what was applied for transparency
        if "trainer" in protein_suggestions:
            logger.info(f"Applied Protein trainer suggestions: {protein_suggestions['trainer']}")

    return cfg

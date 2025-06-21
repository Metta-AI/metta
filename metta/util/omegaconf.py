from __future__ import annotations

from typing import Any, Dict, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

from .fs import get_repo_root  # Import from fs module


def convert_to_dict(config, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert OmegaConf config to a standard Python dictionary with string keys.
    Raises ValueError if the config structure resolves to a list or other non-dict type.

    Args:
        config: The OmegaConf config to convert
        resolve: Whether to resolve interpolations

    Returns:
        Dict[str, Any]: A standard Python dictionary with string keys

    Raises:
        ValueError: If the config doesn't resolve to a dictionary
    """
    result = OmegaConf.to_container(config, resolve=resolve, enum_to_str=True, structured_config_mode=SCMode.DICT)

    if not isinstance(result, dict):
        raise ValueError(f"Expected dictionary configuration, got {type(result).__name__}")

    # Convert all keys to strings to ensure consistent typing
    string_keyed_dict: Dict[str, Any] = {}
    for key, value in result.items():
        string_keyed_dict[str(key)] = value

    return string_keyed_dict


def load_from_path(config_path: str, overrides: Optional[DictConfig] = None) -> DictConfig:
    """
    Load configuration from a path with validation.

    Args:
        config_path: Path to the configuration relative to configs/ directory
                    (e.g., "model/bert" or "training/default.yaml")
        overrides: Optional overrides to apply to the configuration

    Returns:
        The loaded configuration

    Raises:
        ValueError: If the config_path is invalid or if the configuration could not be loaded
    """
    if not config_path:
        raise ValueError("Config path cannot be empty")

    # Remove leading slash if present
    if config_path.startswith("/"):
        config_path = config_path[1:]

    # Get repo root and configs directory
    repo_root = get_repo_root()
    configs_dir = repo_root / "configs"

    if not configs_dir.exists():
        raise ValueError(f"Configs directory not found at {configs_dir}")

    # Add .yaml extension if not present
    if not config_path.endswith((".yaml", ".yml")):
        config_path = f"{config_path}.yaml"

    # Construct full path and check if it exists
    full_path = configs_dir / config_path

    if not full_path.exists():
        # Try without extension in case it was double-added
        alt_path = configs_dir / config_path.replace(".yaml", "")
        if alt_path.exists():
            full_path = alt_path
        else:
            raise ValueError(f"Config file not found: {full_path}")

    # Convert to relative path from configs dir for Hydra
    relative_path = full_path.relative_to(configs_dir)

    # Remove extension for Hydra config name
    config_name = str(relative_path).replace(".yaml", "").replace(".yml", "")

    try:
        # Initialize Hydra with the configs directory
        with hydra.initialize_config_dir(config_dir=str(configs_dir), version_base=None):
            cfg = hydra.compose(config_name=config_name)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {str(e)}") from e

    # Apply overrides if provided
    if overrides:
        # Allow overrides that are not in the config
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.set_struct(cfg, True)

    return cast(DictConfig, cfg)

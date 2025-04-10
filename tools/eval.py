from typing import Any, List, Optional, Union

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


def config_from_path(
    config_path: str, overrides: Optional[Union[DictConfig, dict[str, Any]]] = None
) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration from a specified path, optionally with overrides.

    Args:
        config_path: Path to the configuration file
        overrides: Optional configuration overrides to merge with the loaded config

    Returns:
        The loaded and potentially merged configuration (either DictConfig or ListConfig)
    """
    # Start with a DictConfig from hydra.compose
    env_cfg: Union[DictConfig, ListConfig] = hydra.compose(config_name=config_path)

    if config_path.startswith("/"):
        config_path = config_path[1:]

    path_components: List[str] = config_path.split("/")

    # Navigate through nested config structure
    for p in path_components[:-1]:
        # Type checking for safe access
        if isinstance(env_cfg, DictConfig):
            env_cfg = env_cfg[p]
        elif isinstance(env_cfg, ListConfig):
            # Try to convert p to int for list access, or raise a meaningful error
            try:
                idx = int(p)
                env_cfg = env_cfg[idx]
            except ValueError as err:
                raise TypeError(
                    f"Cannot use string key '{p}' with ListConfig - must be an integer index"
                ) from err
        else:
            raise TypeError(f"Unexpected config type: {type(env_cfg)}")

    # Apply overrides if provided
    if overrides is not None:
        env_cfg = OmegaConf.merge(env_cfg, overrides)

    return env_cfg

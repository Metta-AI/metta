import os
from pathlib import Path
from typing import Optional, cast

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

def config_from_path(
    config_path: str, overrides: Optional[DictConfig | ListConfig | dict] = None
) -> DictConfig | ListConfig:
    """
    Load configuration from a path, with better error handling

    Args:
        config_path: Path to the configuration
        overrides: Optional overrides to apply to the configuration

    Returns:
        The loaded configuration

    Raises:
        ValueError: If the config_path is None or if the configuration could not be loaded
    """
    if config_path is None:
        raise ValueError("Config path cannot be None")

    cfg = hydra.compose(config_name=config_path)

    # when hydra loads a config, it "prefixes" the keys with the path of the config file.
    # We don't want that prefix, so we remove it.
    if config_path.startswith("/"):
        config_path = config_path[1:]

    for p in config_path.split("/")[:-1]:
        cfg = cfg[p]

    if overrides not in [None, {}]:
        # Allow overrides that are not in the config.
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.set_struct(cfg, True)
    return cast(DictConfig, cfg)


def get_test_basic_cfg() -> DictConfig:
    return get_cfg("test_basic")


def get_cfg(config_name: str) -> DictConfig:
    # Save current directory to restore later if needed
    original_cwd = Path.cwd()

    try:
        # Lazy import to avoid circular dependencies during build
        from metta.common.util.fs import cd_repo_root
        
        # Change to repository root
        cd_repo_root()

        # Now we can use a consistent path from repo root
        mettagrid_configs_root = Path("mettagrid/configs")

        config_path = mettagrid_configs_root / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig), f"Config {config_name} did not load into a DictConfig"

        return cfg

    finally:
        # Optionally restore original directory
        os.chdir(original_cwd)

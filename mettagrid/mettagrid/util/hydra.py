from pathlib import Path
from typing import Optional, cast

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


def config_from_path(config_path: str, overrides: Optional[DictConfig | ListConfig] = None) -> DictConfig | ListConfig:
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
    # Get the directory containing the current file
    config_dir = Path(__file__).parent

    # Navigate up two levels to the package root (repo root / mettagrid)
    package_root = config_dir.parent.parent

    # Create paths to the specific directories
    mettagrid_configs_root = package_root / "configs"

    cfg = OmegaConf.load(f"{mettagrid_configs_root}/{config_name}.yaml")
    assert isinstance(cfg, DictConfig), f"Config {config_name} did not load into a DictConfig"
    return cfg

from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


# proxy to hydra.utils.instantiate
# mettagrid doesn't load configs through hydra anymore, but it still needs this function
def simple_instantiate(cfg: DictConfig, recursive: bool = False):
    return hydra.utils.instantiate(cfg, _recursive_=recursive)


def get_test_basic_cfg():
    return get_cfg("test_basic")


def get_cfg(config_name: str):
    # Get the directory containing the current file
    config_dir = Path(__file__).parent

    # Navigate up two levels to the package root (repo root / mettagrid)
    package_root = config_dir.parent.parent

    # Create paths to the specific directories
    mettagrid_configs_root = package_root / "configs"

    cfg = OmegaConf.load(f"{mettagrid_configs_root}/{config_name}.yaml")
    assert isinstance(cfg, DictConfig), f"Config {config_name} did not load into a DictConfig"
    return cfg

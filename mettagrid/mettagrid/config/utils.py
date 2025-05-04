from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

# mettagrid/configs dir, with omegaconf configs
mettagrid_configs_root = Path(__file__).parent.resolve() / "../../configs"

# dir with scenes configs, used in tests and in mettagrid.map.utils.dcss
scenes_root = mettagrid_configs_root / "scenes"


# proxy to hydra.utils.instantiate
# mettagrid doesn't load configs through hydra anymore, but it still needs this function
def simple_instantiate(cfg: DictConfig, recursive: bool = False):
    return hydra.utils.instantiate(cfg, _recursive_=recursive)


def get_test_basic_cfg():
    return get_cfg("test_basic")


def get_cfg(config_name: str):
    cfg = OmegaConf.load(f"{mettagrid_configs_root}/{config_name}.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg

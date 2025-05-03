from typing import Optional

from omegaconf import DictConfig, OmegaConf

from mettagrid.map.mapgen import MapGen


def with_boundaries(pattern: str):
    return "\n".join([f"|{line}|" for line in pattern.split("\n")])


def make_convchain_config_from_pattern(pattern: str) -> DictConfig:
    pattern = with_boundaries(pattern)
    config = OmegaConf.create(
        {
            "_target_": "mettagrid.map.scenes.convchain.ConvChain",
            "pattern_size": 3,
            "iterations": 10,
            "temperature": 1,
            "pattern": pattern,
        }
    )
    return config


def make_wfc_config_from_pattern(pattern: str) -> Optional[DictConfig]:
    pattern = with_boundaries(pattern)
    config = OmegaConf.create(
        {
            "_target_": "mettagrid.map.scenes.wfc.WFC",
            "pattern_size": 3,
            "pattern": pattern,
        }
    )

    # Some WFC patterns are invalid, so we need to check that they are valid.
    # This is the slowest part of import, 2-20 seconds per map.
    mapgen = MapGen(100, 100, config)
    try:
        mapgen.build()
    except Exception:
        return None

    return config

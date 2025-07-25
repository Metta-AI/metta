from typing import Optional

from metta.map.mapgen import MapGen


def with_boundaries(pattern: str):
    return "\n".join([line for line in pattern.split("\n")])


def make_convchain_config_from_pattern(pattern: str) -> dict:
    pattern = with_boundaries(pattern)
    config = {
        "type": "metta.map.scenes.convchain.ConvChain",
        "params": {
            "pattern_size": 3,
            "iterations": 10,
            "temperature": 1,
            "pattern": pattern,
        },
    }

    return config


def make_wfc_config_from_pattern(pattern: str) -> Optional[dict]:
    pattern = with_boundaries(pattern)
    config = {
        "type": "metta.map.scenes.wfc.WFC",
        "params": {
            "pattern_size": 3,
            "pattern": pattern,
        },
    }

    # Some WFC patterns are invalid, so we need to check that they are valid.
    # This is the slowest part of import, 2-20 seconds per map.
    mapgen = MapGen(width=100, height=100, root=config)
    try:
        mapgen.build()
    except Exception:
        return None

    return config

from typing import Optional

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import SceneConfig
from mettagrid.mapgen.scenes.convchain import ConvChain
from mettagrid.mapgen.scenes.wfc import WFC


def make_convchain_config_from_pattern(pattern: str) -> SceneConfig:
    return ConvChain.Config(
        pattern_size=3,
        iterations=10,
        temperature=1,
        pattern=pattern,
    )


def make_wfc_config_from_pattern(pattern: str) -> Optional[SceneConfig]:
    scene_config = WFC.Config(
        pattern_size=3,
        pattern=pattern,
    )

    # Some WFC patterns are invalid, so we need to check that they are valid.
    # This is the slowest part of import, 2-20 seconds per map.
    mapgen = MapGen.Config(width=100, height=100, instance=scene_config).create()
    try:
        mapgen.build()
    except Exception:
        return None

    return scene_config

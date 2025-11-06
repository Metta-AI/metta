import typing

import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.convchain
import mettagrid.mapgen.scenes.wfc


def make_convchain_config_from_pattern(pattern: str) -> mettagrid.mapgen.scene.SceneConfig:
    return mettagrid.mapgen.scenes.convchain.ConvChain.Config(
        pattern_size=3,
        iterations=10,
        temperature=1,
        pattern=pattern,
    )


def make_wfc_config_from_pattern(pattern: str) -> typing.Optional[mettagrid.mapgen.scene.SceneConfig]:
    scene_config = mettagrid.mapgen.scenes.wfc.WFC.Config(
        pattern_size=3,
        pattern=pattern,
    )

    # Some WFC patterns are invalid, so we need to check that they are valid.
    # This is the slowest part of import, 2-20 seconds per map.
    mapgen = mettagrid.mapgen.mapgen.MapGen.Config(width=100, height=100, instance=scene_config).create()
    try:
        mapgen.build()
    except Exception:
        return None

    return scene_config

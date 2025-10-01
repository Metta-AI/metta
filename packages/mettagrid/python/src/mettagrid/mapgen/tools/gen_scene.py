#!/usr/bin/env -S uv run
import logging
from typing import Annotated, Optional

import typer
from omegaconf import OmegaConf

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import SceneConfig
from mettagrid.mapgen.utils.show import ShowMode, show_map
from mettagrid.mapgen.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


def main(
    scene: Annotated[str, typer.Argument(help="Path to the scene config file")],
    width: Annotated[int, typer.Option(help="Width of the map")],
    height: Annotated[int, typer.Option(help="Height of the map")],
    show_mode: Annotated[ShowMode, typer.Option(help="Show mode: ascii, ascii_border, or none")] = "ascii_border",
    scene_override: Annotated[
        Optional[list[str]], typer.Option("--scene-override", help="OmegaConf-style overrides for the scene config")
    ] = None,
):
    """
    Generate a map from a scene config and display it.
    """
    scene_omega_cfg = OmegaConf.load(scene)

    if not OmegaConf.is_dict(scene_omega_cfg):
        raise ValueError(f"Invalid config type: {type(scene_omega_cfg)}")

    # Apply overrides if any
    if scene_override:
        for override in scene_override:
            key, value = override.split("=", 1)
            OmegaConf.update(scene_omega_cfg, key, value)

    scene_cfg = SceneConfig.model_validate(scene_omega_cfg)

    mapgen_cfg = MapGen.Config(
        width=width,
        height=height,
        instance=scene_cfg,
    )
    storable_map = StorableMap.from_cfg(mapgen_cfg)
    show_map(storable_map, show_mode)


if __name__ == "__main__":
    typer.run(main)

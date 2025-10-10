#!/usr/bin/env -S uv run
import logging
from typing import Annotated, Optional

import typer
import yaml

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import validate_any_scene_config
from mettagrid.mapgen.utils.show import ShowMode, show_game_map

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
    with open(scene, "r") as fh:
        yaml_cfg = yaml.safe_load(fh)

    scene_cfg = validate_any_scene_config(yaml_cfg)
    for override in scene_override or []:
        key, value = override.split("=", 1)
        scene_cfg.override(key, value)

    mapgen_cfg = MapGen.Config(
        width=width,
        height=height,
        instance=scene_cfg,
    )
    game_map = mapgen_cfg.create().build()
    show_game_map(game_map, show_mode)


if __name__ == "__main__":
    typer.run(main)

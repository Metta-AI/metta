#!/usr/bin/env -S uv run
import logging
import typing

import typer
import yaml

import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scene
import mettagrid.mapgen.utils.show

logger = logging.getLogger(__name__)


def main(
    scene: typing.Annotated[str, typer.Argument(help="Path to the scene config file")],
    width: typing.Annotated[int, typer.Option(help="Width of the map")],
    height: typing.Annotated[int, typer.Option(help="Height of the map")],
    show_mode: typing.Annotated[
        mettagrid.mapgen.utils.show.ShowMode, typer.Option(help="Show mode: ascii, ascii_border, or none")
    ] = "ascii_border",
    scene_override: typing.Annotated[
        typing.Optional[list[str]],
        typer.Option("--scene-override", help="OmegaConf-style overrides for the scene config"),
    ] = None,
):
    """
    Generate a map from a scene config and display it.
    """
    with open(scene, "r") as fh:
        yaml_cfg = yaml.safe_load(fh)

    scene_cfg = mettagrid.mapgen.scene.SceneConfig.model_validate(yaml_cfg)
    for override in scene_override or []:
        key, value = override.split("=", 1)
        scene_cfg.override(key, value)

    mapgen_cfg = mettagrid.mapgen.mapgen.MapGen.Config(
        width=width,
        height=height,
        instance=scene_cfg,
    )
    game_map = mapgen_cfg.create().build()
    mettagrid.mapgen.utils.show.show_game_map(game_map, show_mode)


if __name__ == "__main__":
    typer.run(main)

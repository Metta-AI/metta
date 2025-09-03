#!/usr/bin/env -S uv run
import logging

import typer
from omegaconf import OmegaConf

from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.scene import SceneConfig
from metta.mettagrid.mapgen.utils.show import ShowMode, show_map
from metta.mettagrid.mapgen.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


def main(
    scene: str = typer.Argument(..., help="Path to the scene config file"),
    width: int = typer.Option(..., help="Width of the map"),
    height: int = typer.Option(..., help="Height of the map"),
    show_mode: ShowMode = typer.Option("ascii_border", help="Show mode: ascii, ascii_border, or none"),
    scene_override: list[str] = typer.Option(
        [], "--scene-override", help="OmegaConf-style overrides for the scene config"
    ),
):
    """
    Generate a map from a scene config and display it.
    """
    scene_omega_cfg = OmegaConf.load(scene)

    if not OmegaConf.is_dict(scene_omega_cfg):
        raise ValueError(f"Invalid config type: {type(scene_omega_cfg)}")

    # Apply overrides if any
    for override in scene_override:
        key, value = override.split("=", 1)
        OmegaConf.update(scene_omega_cfg, key, value)

    scene_cfg = SceneConfig.model_validate(scene_omega_cfg)

    mapgen_cfg = MapGen.Config(
        width=width,
        height=height,
        root=scene_cfg,
    )
    storable_map = StorableMap.from_cfg(mapgen_cfg)
    show_map(storable_map, show_mode)


if __name__ == "__main__":
    typer.run(main)

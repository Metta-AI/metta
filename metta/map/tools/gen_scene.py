#!/usr/bin/env -S uv run
import logging

from omegaconf import OmegaConf

from metta.common.config import Tool
from metta.map.mapgen import MapGen
from metta.map.scene import SceneConfig
from metta.map.utils.show import ShowMode, show_map
from metta.map.utils.storable_map import StorableMap

logger = logging.getLogger(__name__)


class GenSceneTool(Tool):
    scene: str  # Path to the scene config file
    width: int  # Width of the map
    height: int  # Height of the map
    scene_overrides: list[str] = []  # OmegaConf-style overrides for the scene config
    show_mode: ShowMode | None = None  # Show the map in the specified mode

    def invoke(self):
        scene_omega_cfg = OmegaConf.load(self.scene)

        if not OmegaConf.is_dict(scene_omega_cfg):
            raise ValueError(f"Invalid config type: {type(scene_omega_cfg)}")

        scene_cfg = SceneConfig.model_validate(scene_omega_cfg)

        mapgen_cfg = MapGen.Config(
            width=self.width,
            height=self.height,
            root=scene_cfg,
        )
        storable_map = StorableMap.from_cfg(mapgen_cfg)
        show_map(storable_map, self.show_mode)

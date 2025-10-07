from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from typing_extensions import TypedDict

from mettagrid.map_builder import MapBuilder, MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.types import MapGrid

logger = logging.getLogger(__name__)


class FrontmatterDict(TypedDict):
    metadata: dict
    config: dict
    scene_tree: dict | None
    char_to_name: dict[str, str]


class StorableMapDict(TypedDict):
    frontmatter: FrontmatterDict
    data: str


@dataclass
class StorableMap:
    """
    Wrapper around a MapGrid that includes information about the config that produced the map.
    """

    grid: MapGrid
    metadata: dict
    config: MapBuilderConfig  # config that was used to generate the map
    scene_tree: dict | None = None  # defined for mapgen maps

    def width(self) -> int:
        return self.grid.shape[1]

    def height(self) -> int:
        return self.grid.shape[0]

    @staticmethod
    def from_cfg(cfg: MapBuilderConfig[MapBuilder]) -> StorableMap:
        # Generate and measure time taken
        start = time.time()
        map_builder = cfg.create()
        level = map_builder.build()
        gen_time = time.time() - start
        logger.info(f"Time taken to build map: {gen_time}s")

        scene_tree = None
        if isinstance(map_builder, MapGen):
            scene_tree = map_builder.get_scene_tree()

        storable_map = StorableMap(
            grid=level.grid,
            metadata={
                "gen_time": gen_time,
                "timestamp": datetime.now().isoformat(),
            },
            config=cfg,
            scene_tree=scene_tree,
        )
        return storable_map

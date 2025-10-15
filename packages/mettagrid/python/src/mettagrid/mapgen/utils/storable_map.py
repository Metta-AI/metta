from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from typing_extensions import TypedDict

from mettagrid.map_builder import MapBuilder, MapBuilderConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.types import MapGrid
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME, grid_to_lines

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
    char_to_name: dict[str, str] = field(default_factory=dict)

    def width(self) -> int:
        return self.grid.shape[1]

    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def name_to_char(self) -> dict[str, str]:
        return {name: char for char, name in self.char_to_name.items()}

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

        # Extract char_to_name_map from config if available
        # Note that this is the only production code still using DEFAULT_CHAR_TO_NAME - should we remove this?
        char_to_name: dict[str, str] = DEFAULT_CHAR_TO_NAME
        if isinstance(cfg, AsciiMapBuilder.Config):
            char_to_name = cfg.char_to_name_map

        char_to_name = char_to_name.copy()

        # Assign unique chars to unknown names
        known_names = set(char_to_name.values())
        names = np.unique(level.grid)
        next_char = "A"
        for name in names:
            if name not in known_names:
                while next_char in char_to_name:
                    next_char = chr(ord(next_char) + 1)
                char_to_name[next_char] = name

        storable_map = StorableMap(
            grid=level.grid,
            metadata={
                "gen_time": gen_time,
                "timestamp": datetime.now().isoformat(),
            },
            config=cfg,
            scene_tree=scene_tree,
            char_to_name=char_to_name,
        )
        return storable_map

    # Useful in API responses
    def to_dict(self) -> StorableMapDict:
        config_dict = self.config.model_dump()
        assert isinstance(config_dict, dict)
        return {
            "frontmatter": {
                "metadata": self.metadata,
                "config": config_dict,
                "scene_tree": self.scene_tree,
                "char_to_name": self.char_to_name,
            },
            "data": "\n".join(grid_to_lines(self.grid, self.name_to_char)),
        }

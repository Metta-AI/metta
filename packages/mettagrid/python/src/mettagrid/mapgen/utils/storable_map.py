from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from typing_extensions import TypedDict

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.types import MapGrid
from mettagrid.mapgen.utils.ascii_grid import default_char_to_name, grid_to_lines, lines_to_grid

logger = logging.getLogger(__name__)


class FrontmatterDict(TypedDict):
    metadata: dict
    config: dict
    scene_tree: dict | None


class StorableMapDict(TypedDict):
    frontmatter: FrontmatterDict
    data: str


@dataclass
class StorableMap:
    """
    Wrapper around a MapGrid that includes information about the config that
    produces the map and can be saved to a file or S3.
    """

    grid: MapGrid
    metadata: dict
    config: MapBuilderConfig  # config that was used to generate the map
    scene_tree: dict | None = None
    char_to_name: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        frontmatter = yaml.safe_dump(
            {
                "metadata": self.metadata,
                "config": self.config.model_dump(),
                "scene_tree": self.scene_tree,
            }
        )
        content = frontmatter + "\n---\n" + "\n".join(grid_to_lines(self.grid, self.name_to_char)) + "\n"
        return content

    def width(self) -> int:
        return self.grid.shape[1]

    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def name_to_char(self) -> dict[str, str]:
        return {name: char for char, name in self.char_to_name.items()}

    @staticmethod
    def from_uri(uri: str, char_to_name: dict[str, str] | None = None) -> StorableMap:
        logger.info(f"Loading map from {uri}")
        # Only supports local files
        content = Path(uri).read_text()

        # TODO - validate content in a more principled way
        (frontmatter, content) = content.split("---\n", 1)

        frontmatter = yaml.safe_load(frontmatter)
        metadata = frontmatter["metadata"]
        config = frontmatter["config"]
        lines = content.split("\n")

        # make sure we didn't add extra lines because of newlines in the content
        lines = [line for line in lines if line]

        char_to_name = char_to_name or default_char_to_name()
        return StorableMap(
            lines_to_grid(lines, char_to_name), metadata=metadata, config=config, char_to_name=char_to_name
        )

    @staticmethod
    def from_cfg(cfg: MapBuilderConfig) -> StorableMap:
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
        char_to_name = {}
        if hasattr(cfg, "char_to_name_map"):
            char_to_name = cfg.char_to_name_map

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

    def save(self, uri: str):
        content = str(self)
        # Only supports local files
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        logger.info(f"Saved map to {uri}")

    # Useful in API responses
    def to_dict(self) -> StorableMapDict:
        config_dict = self.config.model_dump()
        assert isinstance(config_dict, dict)
        return {
            "frontmatter": {
                "metadata": self.metadata,
                "config": config_dict,
                "scene_tree": self.scene_tree,
            },
            "data": "\n".join(grid_to_lines(self.grid, self.name_to_char)),
        }

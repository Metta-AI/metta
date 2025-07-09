from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypedDict

from metta.common.util.instantiate import instantiate
from metta.map.mapgen import MapGen
from metta.map.types import MapGrid
from metta.map.utils.ascii_grid import grid_to_lines, lines_to_grid
from metta.mettagrid.util import file as file_utils

logger = logging.getLogger(__name__)


class FrontmatterDict(TypedDict):
    metadata: dict
    config: dict
    scene_tree: dict | None = None


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
    config: DictConfig  # config that was used to generate the map
    scene_tree: dict | None = None

    def __str__(self) -> str:
        frontmatter = OmegaConf.to_yaml(
            {
                "metadata": self.metadata,
                "config": self.config,
                "scene_tree": self.scene_tree,
            }
        )
        content = frontmatter + "\n---\n" + "\n".join(grid_to_lines(self.grid)) + "\n"
        return content

    def width(self) -> int:
        return self.grid.shape[1]

    def height(self) -> int:
        return self.grid.shape[0]

    @staticmethod
    def from_uri(uri: str) -> StorableMap:
        logger.info(f"Loading map from {uri}")
        content = file_utils.read(uri).decode()

        # TODO - validate content in a more principled way
        (frontmatter, content) = content.split("---\n", 1)

        frontmatter = OmegaConf.create(frontmatter)
        metadata = frontmatter.metadata
        config = frontmatter.config
        lines = content.split("\n")

        # make sure we didn't add extra lines because of newlines in the content
        lines = [line for line in lines if line]

        return StorableMap(lines_to_grid(lines), metadata=metadata, config=config)

    @staticmethod
    def from_cfg(cfg: DictConfig) -> StorableMap:
        # Generate and measure time taken
        start = time.time()
        # TODO(slava): Remove _recursive_=True once mapgen no longer needs it
        map_builder = instantiate(OmegaConf.to_container(cfg, resolve=True), _recursive_=True)
        level = map_builder.build()
        gen_time = time.time() - start
        logger.info(f"Time taken to build map: {gen_time}s")

        scene_tree = None
        if isinstance(map_builder, MapGen):
            scene_tree = map_builder.get_scene_tree()

        storable_map = StorableMap(
            grid=level.grid,
            metadata={
                "labels": level.labels,
                "gen_time": gen_time,
                "timestamp": datetime.now().isoformat(),
            },
            config=cfg,
            scene_tree=scene_tree,
        )
        return storable_map

    def save(self, uri: str):
        file_utils.write_data(uri, str(self), content_type="text/plain")
        logger.info(f"Saved map to {uri}")

    # Useful in API responses
    def to_dict(self) -> StorableMapDict:
        config_dict = OmegaConf.to_container(self.config, resolve=False)
        assert isinstance(config_dict, dict)
        return {
            "frontmatter": {
                "metadata": self.metadata,
                "config": config_dict,
                "scene_tree": self.scene_tree,
            },
            "data": "\n".join(grid_to_lines(self.grid)),
        }

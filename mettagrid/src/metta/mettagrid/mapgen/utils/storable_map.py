from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf
from typing_extensions import TypedDict

from metta.mettagrid.map_builder.map_builder import MapBuilderConfig
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.types import MapGrid
from metta.mettagrid.mapgen.utils.ascii_grid import grid_to_lines, lines_to_grid
from metta.mettagrid.mapgen.utils.map_compression import compress_map, decompress_map
from metta.mettagrid.util import file as file_utils

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

    # Optional compressed representation
    object_key: Optional[List[str]] = None

    def __str__(self) -> str:
        # Compress for storage
        byte_grid, object_key = compress_map(self.grid)

        frontmatter = OmegaConf.to_yaml(
            {
                "metadata": self.metadata,
                "config": self.config.model_dump(),
                "scene_tree": self.scene_tree,
                "object_key": object_key,  # Store the key
            }
        )

        # Convert byte grid to compact string representation
        # Use base64 for maximum compression
        import base64

        grid_bytes = byte_grid.tobytes()
        grid_b64 = base64.b64encode(grid_bytes).decode("ascii")

        content = frontmatter + "\n---\n"
        content += f"shape: {byte_grid.shape}\n"
        content += f"data: {grid_b64}\n"

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
        (frontmatter, data_section) = content.split("---\n", 1)

        frontmatter = OmegaConf.create(frontmatter)
        metadata = frontmatter.metadata
        config = frontmatter.config

        # Check if it's compressed format
        if "object_key" in frontmatter:
            # NEW: Decompress byte grid
            import base64
            import re

            shape_match = re.search(r"shape: \((\d+), (\d+)\)", data_section)
            data_match = re.search(r"data: (.+)", data_section)

            if shape_match and data_match:
                shape = (int(shape_match.group(1)), int(shape_match.group(2)))
                grid_b64 = data_match.group(1)
                grid_bytes = base64.b64decode(grid_b64)
                byte_grid = np.frombuffer(grid_bytes, dtype=np.uint8).reshape(shape)

                # Decompress to string grid
                grid = decompress_map(byte_grid, frontmatter.object_key)
            else:
                # Fallback to old format
                lines = data_section.split("\n")
                lines = [line for line in lines if line]
                grid = lines_to_grid(lines)
        else:
            # Old format (backward compatibility)
            lines = data_section.split("\n")
            lines = [line for line in lines if line]
            grid = lines_to_grid(lines)

        return StorableMap(
            grid=grid,
            metadata=metadata,
            config=config,
            scene_tree=frontmatter.get("scene_tree"),
            object_key=frontmatter.get("object_key"),
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

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.map.types import MapGrid
from metta.map.utils.ascii_grid import ascii_to_grid, grid_to_ascii
from mettagrid.util import file as file_utils

logger = logging.getLogger(__name__)


@dataclass
class StorableMap:
    """
    Wrapper around a MapGrid that includes information about the config that
    produces the map and can be saved to a file or S3.
    """

    grid: MapGrid
    metadata: dict
    config: DictConfig  # config that was used to generate the map

    def __str__(self) -> str:
        frontmatter = OmegaConf.to_yaml(
            {
                "metadata": self.metadata,
                "config": self.config,
            }
        )
        content = frontmatter + "\n---\n" + "\n".join(grid_to_ascii(self.grid)) + "\n"
        return content

    def width(self) -> int:
        return self.grid.shape[1]

    def height(self) -> int:
        return self.grid.shape[0]

    @staticmethod
    def from_uri(uri: str) -> "StorableMap":
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

        return StorableMap(ascii_to_grid(lines), metadata=metadata, config=config)

    def save(self, uri: str):
        file_utils.write_data(uri, str(self), content_type="text/plain")
        logger.info(f"Saved map to {uri}")

    # Useful in API responses
    def to_dict(self) -> dict:
        return {
            "frontmatter": {
                "metadata": self.metadata,
                "config": OmegaConf.to_container(self.config, resolve=False),
            },
            "data": "\n".join(grid_to_ascii(self.grid)),
        }


def map_builder_cfg_to_storable_map(cfg: DictConfig) -> StorableMap:
    # Generate and measure time taken
    start = time.time()
    map_builder = hydra.utils.instantiate(cfg, _recursive_=True)
    level = map_builder.build()
    gen_time = time.time() - start
    logger.info(f"Time taken to build map: {gen_time}s")

    storable_map = StorableMap(
        grid=level.grid,
        metadata={
            "gen_time": gen_time,
            "timestamp": datetime.now().isoformat(),
        },
        config=cfg,
    )
    return storable_map

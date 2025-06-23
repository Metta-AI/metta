import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypedDict

from metta.map.types import MapGrid
from metta.map.utils.ascii_grid import grid_to_lines, lines_to_grid
from metta.map.utils.s3utils import list_objects
from metta.mettagrid.util import file as file_utils

logger = logging.getLogger(__name__)


class FrontmatterDict(TypedDict):
    metadata: dict
    config: dict


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

    def __str__(self) -> str:
        frontmatter = OmegaConf.to_yaml(
            {
                "metadata": self.metadata,
                "config": self.config,
            }
        )
        content = frontmatter + "\n---\n" + "\n".join(grid_to_lines(self.grid)) + "\n"
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

        return StorableMap(lines_to_grid(lines), metadata=metadata, config=config)

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
            },
            "data": "\n".join(grid_to_lines(self.grid)),
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


@dataclass
class StorableMapIndex:
    """
    Inverted index of storable maps in an S3 directory.

    The index can quickly find all maps that have a particular value for a particular key in their configs.
    """

    dir: str
    index_data: dict[str, dict[str, list[str]]]

    def _flatten_nested_dict(self, obj, parent_key=""):
        """Flatten nested dictionaries and lists into dot-separated keys."""
        items = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                items.extend(self._flatten_nested_dict(v, new_key))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}.{i}" if parent_key else str(i)
                items.extend(self._flatten_nested_dict(v, new_key))
        else:
            items.append((parent_key, obj))

        return items

    def _process(self, map: StorableMap, uri: str):
        key_to_value: dict[str, str] = {}
        config_container = OmegaConf.to_container(map.config, resolve=False)
        for k, v in self._flatten_nested_dict(config_container):
            key = f"config.{k}"
            key_to_value[key] = str(v)

        key_to_value.update({f"metadata.{k}": v for k, v in map.metadata.items()})
        for key, value in key_to_value.items():
            if key not in self.index_data:
                self.index_data[key] = {}
            if value not in self.index_data[key]:
                self.index_data[key][value] = []
            self.index_data[key][value].append(uri)

    def find_maps(self, filters: list[tuple[str, str]]) -> list[str]:
        map_files: list[str]

        if filters:
            map_files_set: set[str] | None = None
            for key, value in filters:
                filter_files = self.index_data[key][value]
                map_files_set = map_files_set.intersection(set(filter_files)) if map_files_set else set(filter_files)

            map_files = list(map_files_set) if map_files_set else []
        else:
            map_files = [f for f in list_objects(self.dir) if f.endswith(".yaml")]

        return map_files

    def _index_dir(self):
        filenames = list_objects(self.dir)
        for filename in filenames:
            map = StorableMap.from_uri(filename)
            self._process(map, filename)

    def _save(self):
        index_uri = f"{self.dir}/index.json"
        file_utils.write_data(index_uri, json.dumps(self.index_data), content_type="text/plain")

    @staticmethod
    def load(dir: str):
        """Load an index from `dir`."""
        index_content = file_utils.read(f"{dir}/index.json")
        index_data = json.loads(index_content.decode("utf-8"))
        index = StorableMapIndex(dir=dir, index_data=index_data)
        return index

    @staticmethod
    def create(dir: str):
        """Create a new index in `dir`. If the index already exists, it will be overwritten."""
        index = StorableMapIndex(dir=dir, index_data={})
        index._index_dir()
        index._save()

import logging
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.map.mapgen import MapGrid

from . import storage

logger = logging.getLogger(__name__)

ascii_symbols = {
    "empty": " ",
    "wall": "#",
    "agent.agent": "A",
    "mine": "g",
    "generator": "c",
    "altar": "a",
    "armory": "r",
    "lasery": "l",
    "lab": "b",
    "factory": "f",
    "temple": "t",
}

reverse_ascii_symbols = {v: k for k, v in ascii_symbols.items()}


def grid_object_to_ascii(name: str) -> str:
    if name in ascii_symbols:
        return ascii_symbols[name]

    if name == "block":
        # FIXME - store maps in a different format, or pick a different character
        raise ValueError("Block is not supported in ASCII mode")

    if name.startswith("mine."):
        raise ValueError("Colored mines are not supported in ASCII mode")

    if name.startswith("agent."):
        raise ValueError("Agent groups are not supported in ASCII mode")

    raise ValueError(f"Unknown object type: {name}")


def ascii_to_grid_object(ascii: str) -> str:
    if ascii in reverse_ascii_symbols:
        return reverse_ascii_symbols[ascii]

    raise ValueError(f"Unknown character: {ascii}")


def grid_to_ascii(grid: MapGrid, border: bool = False) -> list[str]:
    lines: list[str] = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            row.append(grid_object_to_ascii(grid[r, c]))
        lines.append("".join(row))

    if border:
        width = len(lines[0])
        border_lines = ["┌" + "─" * width + "┐"]
        for row in lines:
            border_lines.append("│" + row + "│")
        border_lines.append("└" + "─" * width + "┘")
        lines = border_lines

    return lines


def ascii_to_grid(lines: list[str]) -> MapGrid:
    grid = np.full((len(lines), len(lines[0])), "empty", dtype="<U50")
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            grid[r, c] = ascii_to_grid_object(char)
    return grid


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
        content = storage.load_from_uri(uri)

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
        storage.save_to_uri(str(self), uri)
        logger.info(f"Saved map to {uri}")

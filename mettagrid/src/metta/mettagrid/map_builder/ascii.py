from typing import Literal

import numpy as np

from metta.mettagrid.char_encoder import char_to_grid_object
from metta.mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig


class AsciiMapBuilderConfig(MapBuilderConfig):
    """
    Configuration for building a game map from an ASCII string.
    """

    type: Literal["ascii"] = "ascii"
    uri: str

    def create(self) -> "AsciiMapBuilder":
        """Create an AsciiMapBuilder from this configuration."""
        return AsciiMapBuilder(self)


class AsciiMapBuilder(MapBuilder):
    def __init__(self, config: AsciiMapBuilderConfig):
        super().__init__(config=config)
        with open(config.uri, "r", encoding="utf-8") as f:
            ascii_map = f.read()
        lines = ascii_map.strip().splitlines()

        # Assert all lines are the same length
        if lines:
            expected_length = len(lines[0])
            for i, line in enumerate(lines):
                assert len(line) == expected_length, (
                    f"Line {i} has length {len(line)}, expected {expected_length}. "
                    f"All lines in ASCII map must have the same length."
                )

        self._level = np.array([list(line) for line in lines], dtype="U6")
        self._level = np.vectorize(char_to_grid_object)(self._level)

    def build(self) -> GameMap:
        return GameMap(self._level)

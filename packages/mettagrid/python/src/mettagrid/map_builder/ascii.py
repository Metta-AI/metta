from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import StringConstraints, field_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import merge_with_global_defaults


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]
        char_to_name_map: dict[
            Annotated[str, StringConstraints(min_length=1, max_length=1)],  # keys are single characters
            Annotated[str, StringConstraints(pattern=r"^[\w\.]+$")],  # values are object names
        ]

        @field_validator("map_data", mode="before")
        @classmethod
        def _validate_multiline_map_data(cls, value: Any):
            # coerce single multi-line string -> list[list[str]]
            if isinstance(value, str):
                return [list(line) for line in value.splitlines()]
            # coerce list[str] -> list[list[str]]
            if isinstance(value, list) and isinstance(value[0], str):
                return [list(line) for line in value]
            return value

        @field_validator("map_data", mode="after")
        @classmethod
        def _validate_map_data_lines(cls, map_data: list[str]):
            width = len(map_data[0])
            for i, line in enumerate(map_data):
                assert len(line) == width, (
                    f"Line {i} has length {len(line)}, expected {width}. "
                    f"All lines in ASCII map must have the same length."
                )
            return map_data

        @field_validator("char_to_name_map", mode="after")
        @classmethod
        def _validate_char_to_name_map(cls, value: dict[str, str]):
            return merge_with_global_defaults(value)

        @property
        def width(self) -> int:
            return len(self.map_data[0]) if self.map_data else 0

        @property
        def height(self) -> int:
            return len(self.map_data)

    def __init__(self, config: Config):
        self.config = config

        self._level = np.array([list(line) for line in config.map_data], dtype="U6")
        self._level = np.vectorize(self._char_to_object_name)(self._level)

    def _char_to_object_name(self, char: str) -> str:
        """Convert a map character to an object name."""
        if char in self.config.char_to_name_map:
            return self.config.char_to_name_map[char]
        raise ValueError(f"Unknown character: '{char}'. Available: {list(self.config.char_to_name_map.keys())}")

    def build(self) -> GameMap:
        return GameMap(self._level)

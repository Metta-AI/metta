import numpy as np
from pydantic import Field, model_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import (
    DEFAULT_CHAR_TO_NAME,
    parse_legend_lines,
    split_ascii_map_sections,
)


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]
        char_to_name_map: dict[str, str] = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_char_to_name_map(self) -> "AsciiMapBuilder.Config":
            self.char_to_name_map = DEFAULT_CHAR_TO_NAME | self.char_to_name_map
            return self

        @property
        def width(self) -> int:
            return len(self.map_data[0]) if self.map_data else 0

        @property
        def height(self) -> int:
            return len(self.map_data)

        @classmethod
        def from_uri(cls, uri: str, char_to_name_map: dict[str, str] | None = None) -> "AsciiMapBuilder.Config":
            with open(uri, "r", encoding="utf-8") as f:
                ascii_map = f.read()

            legend_lines, body_lines = split_ascii_map_sections(ascii_map)
            legend_map = parse_legend_lines(legend_lines)

            if not body_lines:
                raise ValueError(f"ASCII map at {uri!r} is empty")

            return cls(
                map_data=[list(line) for line in body_lines],
                char_to_name_map=(char_to_name_map or {}) | legend_map,
            )

    def __init__(self, config: Config):
        self.config = config

        # Assert all lines are the same length
        if config.map_data:
            expected_length = len(config.map_data[0])
            for i, line in enumerate(config.map_data):
                assert len(line) == expected_length, (
                    f"Line {i} has length {len(line)}, expected {expected_length}. "
                    f"All lines in ASCII map must have the same length."
                )

        self._level = np.array([list(line) for line in config.map_data], dtype="U6")
        self._level = np.vectorize(self._char_to_object_name)(self._level)

    def _char_to_object_name(self, char: str) -> str:
        """Convert a map character to an object name."""
        if char in self.config.char_to_name_map:
            return self.config.char_to_name_map[char]
        raise ValueError(f"Unknown character: '{char}'. Available: {list(self.config.char_to_name_map.keys())}")

    def build(self) -> GameMap:
        return GameMap(self._level)

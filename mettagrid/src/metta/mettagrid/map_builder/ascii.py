import numpy as np

from metta.mettagrid.char_encoder import char_to_grid_object
from metta.mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]

        @classmethod
        def from_uri(cls, uri: str) -> "AsciiMapBuilder.Config":
            with open(uri, "r", encoding="utf-8") as f:
                ascii_map = f.read()
            lines = ascii_map.strip().splitlines()
            return cls(map_data=[list(line) for line in lines])

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

        # Convert strings first to avoid truncation
        # np.vectorize infers dtype from first element which can truncate longer strings
        converted = [[char_to_grid_object(ch) for ch in row] for row in config.map_data]
        maxlen = max(1, max(len(ch) for row in converted for ch in row)) if converted else 1
        self._level = np.array(converted, dtype=f"U{maxlen}")

    def build(self) -> GameMap:
        return GameMap(self._level)

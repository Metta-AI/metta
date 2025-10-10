from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]
        char_to_name_map: dict[str, str] = Field(default_factory=dict)

        @field_validator("map_data", mode="before")
        @classmethod
        def _normalize_map_data(cls, value: Any) -> list[list[str]]:
            def _ensure_consistent(rows: list[list[str]]) -> list[list[str]]:
                if not rows:
                    return []
                width = len(rows[0])
                for row in rows:
                    if len(row) != width:
                        raise ValueError("All rows in map_data must have the same length")
                    for cell in row:
                        if not isinstance(cell, str) or len(cell) != 1:
                            raise ValueError("map_data must contain single-character strings")
                return rows

            if isinstance(value, str):
                lines = cls._map_lines_from_yaml(value)
                return _ensure_consistent([list(line) for line in lines])

            if isinstance(value, (list, tuple)):
                if not value:
                    return []
                if all(isinstance(row, list) for row in value):
                    return _ensure_consistent([list(row) for row in value])
                if all(isinstance(row, str) for row in value):
                    lines = cls._map_lines_from_yaml(list(value))
                    return _ensure_consistent([list(line) for line in lines])

            raise TypeError("map_data must be a string, sequence of strings, or list of character lists")

        @field_validator("char_to_name_map", mode="before")
        @classmethod
        def _normalize_char_map(cls, value: Any) -> dict[str, str]:
            if value is None:
                return {}
            if isinstance(value, dict):
                return cls._legend_from_yaml(value)
            if isinstance(value, (list, tuple)):
                return cls._legend_from_yaml(dict(value))
            raise TypeError("char_to_name_map must be a mapping of characters to names")

        @classmethod
        def _map_lines_from_yaml(cls, value: Any) -> list[str]:
            if isinstance(value, str):
                lines = value.splitlines()
            elif isinstance(value, (list, tuple)):
                lines = []
                for item in value:
                    if not isinstance(item, str):
                        raise ValueError("All map entries must be strings")
                    lines.append(item)
            else:
                raise ValueError("'map' must be a string or list of strings")

            if not lines:
                raise ValueError("Map must contain at least one line")

            width = len(lines[0])
            if any(len(line) != width for line in lines):
                raise ValueError("All lines in the map must have the same length")

            return lines

        @classmethod
        def _legend_from_yaml(cls, value: Any) -> dict[str, str]:
            if not isinstance(value, dict):
                raise ValueError("'legend' must be a mapping")
            legend: dict[str, str] = {}
            for token_raw, name_raw in value.items():
                if not isinstance(token_raw, str) or not isinstance(name_raw, str):
                    raise ValueError("Legend keys and values must be strings")
                token = token_raw.strip().strip("'\"")
                if len(token) != 1 or any(ch.isspace() for ch in token):
                    raise ValueError(
                        f"Legend token must be a single non-whitespace character: {token_raw!r}"
                    )
                value_stripped = name_raw.strip()
                if not value_stripped or any(ch.isspace() for ch in value_stripped):
                    raise ValueError(
                        f"Legend values must be non-empty and contain no whitespace: {name_raw!r}"
                    )
                legend[token] = value_stripped
            return legend

        @model_validator(mode="after")
        def validate_char_to_name_map(self) -> "AsciiMapBuilder.Config":
            if not self.char_to_name_map:
                raise ValueError("Ascii maps must define a non-empty char_to_name_map")
            return self

        @property
        def width(self) -> int:
            return len(self.map_data[0]) if self.map_data else 0

        @property
        def height(self) -> int:
            return len(self.map_data)

        @classmethod
        def from_uri(cls, uri: str, char_to_name_map: dict[str, str] | None = None) -> "AsciiMapBuilder.Config":
            config: AsciiMapBuilder.Config = super().from_uri(uri)  # type: ignore[return-value]

            if char_to_name_map:
                config.char_to_name_map |= cls._legend_from_yaml(char_to_name_map)
            return config

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

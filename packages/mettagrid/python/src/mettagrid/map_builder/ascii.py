from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import Field, field_validator, model_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME

MAP_KEY = "map_data"
LEGEND_KEY = "char_to_name_map"
COGS_VS_CLIPS_PATH_MARKER = Path("packages") / "cogames"

try:  # pragma: no cover - optional dependency in consumers outside cogames
    from cogames.cogs_vs_clips.missions import _get_default_map_objects as _get_cogs_vs_clips_defaults
except ImportError:  # pragma: no cover
    _get_cogs_vs_clips_defaults = None


def _build_cogs_vs_clips_char_map() -> dict[str, str]:
    if _get_cogs_vs_clips_defaults is None:
        return {}

    mapping: dict[str, str] = {}
    for config in _get_cogs_vs_clips_defaults().values():  # type: ignore[operator]
        token = getattr(config, "map_char", None)
        name = getattr(config, "name", None)
        if not token or not name:
            continue
        if len(token) != 1:
            raise ValueError(f"Legend token must be a single character: {token!r}")
        mapping[token] = name
    return mapping


COGS_VS_CLIPS_CHAR_MAP: dict[str, str] = _build_cogs_vs_clips_char_map()


def _parse_ascii_map(text: str) -> tuple[list[list[str]], dict[str, str]]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("Map file must be valid YAML") from exc

    if not isinstance(data, dict):
        raise ValueError("Map file must be a YAML mapping with 'map' and 'legend'")

    if MAP_KEY not in data or LEGEND_KEY not in data:
        raise ValueError("Map YAML must include both 'map' and 'legend'")

    map_rows = AsciiMapBuilder.Config._normalize_map_data(data[MAP_KEY])
    legend_map = AsciiMapBuilder.Config._normalize_char_map(data[LEGEND_KEY])
    return map_rows, legend_map


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

        @staticmethod
        def _validate_token(token: str) -> str:
            token = token.strip().strip("'\"")
            if len(token) != 1 or any(ch.isspace() for ch in token):
                raise ValueError(f"Legend token must be a single non-whitespace character: {token!r}")
            return token

        @staticmethod
        def _validate_value(value: str) -> str:
            value = value.strip()
            if not value or any(ch.isspace() for ch in value):
                raise ValueError(f"Legend values must be non-empty and contain no whitespace: {value!r}")
            return value

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
                token = cls._validate_token(token_raw)
                legend[token] = cls._validate_value(name_raw)
            return legend

        @classmethod
        def _build_from_ascii(cls, ascii_map: str, char_to_name_map: dict[str, str] | None = None) -> dict[str, Any]:
            try:
                map_rows, legend_map = _parse_ascii_map(ascii_map)
                map_lines = ["".join(row) for row in map_rows]
            except ValueError:
                if "map_data" in ascii_map and "char_to_name_map" in ascii_map:
                    raise
                map_lines = cls._map_lines_from_yaml(ascii_map)
                legend_map = {}

            merged = legend_map.copy()
            if char_to_name_map:
                merged |= char_to_name_map

            data: dict[str, Any] = {
                "map_data": [list(line) for line in map_lines],
            }
            if merged:
                data["char_to_name_map"] = merged

            return data

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
            config: AsciiMapBuilder.Config = super().from_uri(uri)  # type: ignore[return-value]

            if char_to_name_map:
                config.char_to_name_map |= cls._legend_from_yaml(char_to_name_map)

            return cls._apply_path_char_map(config, Path(uri))

        @classmethod
        def from_ascii_map(
            cls, ascii_map: str, char_to_name_map: dict[str, str] | None = None
        ) -> "AsciiMapBuilder.Config":
            data = cls._build_from_ascii(ascii_map, char_to_name_map)
            return cls.model_validate(data)

        @classmethod
        def _apply_path_char_map(cls, config: "AsciiMapBuilder.Config", path: Path) -> "AsciiMapBuilder.Config":
            preset = cls._legend_for_path(path)
            if not preset:
                return config

            char_map = dict(config.char_to_name_map)
            for char in char_map:
                if char in preset:
                    char_map[char] = preset[char]

            config.char_to_name_map = char_map
            return config

        @classmethod
        def _legend_for_path(cls, path: Path) -> dict[str, str]:
            if cls._path_contains_marker(path) and COGS_VS_CLIPS_CHAR_MAP:
                return COGS_VS_CLIPS_CHAR_MAP
            return {}

        @staticmethod
        def _path_contains_marker(path: Path) -> bool:
            marker_parts = COGS_VS_CLIPS_PATH_MARKER.parts

            def matches(candidate: Path) -> bool:
                parts = candidate.parts
                for idx in range(len(parts) - len(marker_parts) + 1):
                    if parts[idx : idx + len(marker_parts)] == marker_parts:
                        return True
                return False

            try:
                relative = path.relative_to(Path.cwd())
            except ValueError:
                relative = path

            return matches(relative) or matches(path)

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

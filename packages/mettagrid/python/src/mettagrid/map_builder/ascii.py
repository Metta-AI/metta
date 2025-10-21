from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from pydantic import Field, StringConstraints, field_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import merge_with_global_defaults


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]
        char_to_name_map: dict[
            Annotated[str, StringConstraints(min_length=1, max_length=1)],
            Annotated[str, StringConstraints(pattern=r"^[\w\.]+$")],
        ]
        target_agents: int | None = Field(default=None)

        @field_validator("map_data", mode="before")
        @classmethod
        def _validate_multiline_map_data(cls, value: Any):
            # coerce single multi-line string -> list[list[str]]
            if isinstance(value, str):
                return [list(line) for line in value.splitlines()]
            # coerce list[str] -> list[list[str]]
            if isinstance(value, list) and value and isinstance(value[0], str):
                return [list(line) for line in value]
            return value

        @field_validator("map_data", mode="after")
        @classmethod
        def _validate_map_data_lines(cls, map_data: list[list[str]]):
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

        @classmethod
        def from_uri(
            cls,
            uri: str | Path,
            *,
            char_to_name_map: dict[str, str] | None = None,
            target_agents: int | None = None,
        ) -> "AsciiMapBuilder.Config":
            path = Path(uri)
            raw = path.read_text(encoding="utf-8")
            try:
                parsed = yaml.safe_load(raw)
            except yaml.YAMLError:
                parsed = None

            if isinstance(parsed, dict):
                return cls.model_validate(parsed)

            lines = [list(line) for line in raw.strip().splitlines() if line]
            return cls(
                map_data=lines,
                char_to_name_map=char_to_name_map or {},
                target_agents=target_agents,
            )

    def __init__(self, config: Config):
        self.config = config

        self._char_grid = np.array(config.map_data, dtype="U6")
        self._apply_spawn_points()
        self._level = np.vectorize(self._char_to_object_name)(self._char_grid)

    def _apply_spawn_points(self) -> None:
        agent_positions = [tuple(pos) for pos in np.argwhere(self._char_grid == "@")]
        spawn_positions = [tuple(pos) for pos in np.argwhere(self._char_grid == "%")]

        target_agents = self.config.target_agents
        if target_agents is None:
            self._agents_count = int(np.count_nonzero(self._char_grid == "@"))
            self.config.target_agents = self._agents_count
            return

        target_agents = int(target_agents)

        if spawn_positions:
            for y, x in agent_positions:
                self._char_grid[y, x] = "."
            agent_positions = []

            if target_agents > len(spawn_positions):
                raise ValueError(
                    f"Requested {target_agents} agents but only {len(spawn_positions)} spawn points available"
                )

            for y, x in spawn_positions[:target_agents]:
                self._char_grid[y, x] = "@"
                agent_positions.append((y, x))

            for y, x in spawn_positions[target_agents:]:
                self._char_grid[y, x] = "."
        else:
            current_agents = len(agent_positions)
            if target_agents < current_agents:
                for y, x in agent_positions[target_agents:]:
                    self._char_grid[y, x] = "."
                agent_positions = agent_positions[:target_agents]
            elif target_agents > current_agents:
                raise ValueError("Cannot increase agent count without spawn points – please add '%' markers to the map")

        self._agents_count = int(np.count_nonzero(self._char_grid == "@"))
        self.config.target_agents = self._agents_count
        self.config.map_data = self._char_grid.tolist()

    def _char_to_object_name(self, char: str) -> str:
        """Convert a map character to an object name."""
        if char in self.config.char_to_name_map:
            return self.config.char_to_name_map[char]
        raise ValueError(f"Unknown character: '{char}'. Available: {list(self.config.char_to_name_map.keys())}")

    def build(self) -> GameMap:
        game_map = GameMap(self._level)
        game_map.num_agents = self._agents_count
        return game_map

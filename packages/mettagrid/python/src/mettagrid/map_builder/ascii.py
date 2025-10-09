from typing import Optional

import numpy as np
from pydantic import Field, model_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME


class AsciiMapBuilder(MapBuilder):
    """
    Builds a game map from an ASCII string.
    """

    class Config(MapBuilderConfig["AsciiMapBuilder"]):
        map_data: list[list[str]]
        char_to_name_map: dict[str, str] = Field(default_factory=dict)
        target_agents: Optional[int] = None

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
        def from_uri(
            cls,
            uri: str,
            char_to_name_map: dict[str, str] | None = None,
            target_agents: Optional[int] = None,
        ) -> "AsciiMapBuilder.Config":
            with open(uri, "r", encoding="utf-8") as f:
                ascii_map = f.read()
            lines = ascii_map.strip().splitlines()
            return cls(
                map_data=[list(line) for line in lines],
                char_to_name_map=char_to_name_map or {},
                target_agents=target_agents,
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

        self._char_grid = np.array(config.map_data, dtype="U6")
        self._apply_spawn_points()
        self._level = np.vectorize(self._char_to_object_name)(self._char_grid)
        self._agents_count = int(np.count_nonzero(self._char_grid == "@"))
        self.config.target_agents = self._agents_count

    def _apply_spawn_points(self) -> None:
        agent_positions = [tuple(pos) for pos in np.argwhere(self._char_grid == "@")]
        spawn_positions = [tuple(pos) for pos in np.argwhere(self._char_grid == "%")]

        # Only apply spawn-point based placement when a target is explicitly provided.
        if self.config.target_agents is not None:
            target_agents = int(self.config.target_agents)

            if spawn_positions:
                # Remove any existing agents – rebuild from spawn points.
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

                # Clear any remaining spawn markers to empty so they don't render as special tiles
                for y, x in spawn_positions[target_agents:]:
                    self._char_grid[y, x] = "."

            else:
                # Fall back to existing agent placements if no spawn points are defined.
                current_agents = len(agent_positions)
                if target_agents < current_agents:
                    for y, x in agent_positions[target_agents:]:
                        self._char_grid[y, x] = "."
                    agent_positions = agent_positions[:target_agents]
                elif target_agents > current_agents:
                    raise ValueError(
                        "Cannot increase agent count without spawn points – please add '%' markers to the map"
                    )

        # When target_agents is None, we leave the map as-is:
        # - Existing '@' agents remain
        # - '%' markers remain as placeholders and will map to 'empty' via char_to_name_map

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

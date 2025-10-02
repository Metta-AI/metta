"""Utilities for map-based CoGames scenarios."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import GameMap


class DynamicAgentAsciiMapBuilder(AsciiMapBuilder):
    """ASCII map builder with dynamic agent placement near the map center."""

    class Config(AsciiMapBuilder.Config):
        num_agents: int = Field(default=4, description="Number of dynamically spawned agents")

        @classmethod
        def from_uri(
            cls,
            uri: str,
            char_to_name_map: dict[str, str] | None = None,
            num_agents: int = 4,
        ) -> "DynamicAgentAsciiMapBuilder.Config":
            with open(uri, "r", encoding="utf-8") as file:
                ascii_map = file.read()
            lines = ascii_map.strip().splitlines()
            return cls(
                map_data=[list(line) for line in lines],
                char_to_name_map=char_to_name_map or {},
                num_agents=num_agents,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.dynamic_config = config

    def build(self) -> GameMap:
        # Replace explicit agent markers with empty cells before placement
        map_data_no_agents = []
        for line in self.config.map_data:
            new_line = [char if char != "@" else "." for char in line]
            map_data_no_agents.append(new_line)

        # Convert characters to object names
        level = np.array(map_data_no_agents, dtype="U6")
        level = np.vectorize(self._char_to_object_name)(level)

        # Locate center of the map
        height, width = level.shape
        center_y, center_x = height // 2, width // 2

        # Expand outward from the center looking for empty cells
        max_radius = max(1, min(height, width) // 3)
        agent_positions: list[tuple[int, int]] = []
        used_positions: set[tuple[int, int]] = set()

        for radius in range(1, max_radius + 1):
            candidates: list[tuple[int, int]] = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = center_y + dy, center_x + dx
                    if (
                        0 <= y < height
                        and 0 <= x < width
                        and level[y, x] == "empty"
                        and (dy * dy + dx * dx) <= radius * radius
                        and (y, x) not in used_positions
                    ):
                        candidates.append((y, x))

            if not candidates:
                continue

            np.random.shuffle(candidates)
            for candidate in candidates:
                if len(agent_positions) >= self.dynamic_config.num_agents:
                    break
                agent_positions.append(candidate)
                used_positions.add(candidate)

            if len(agent_positions) >= self.dynamic_config.num_agents:
                break

        if len(agent_positions) < self.dynamic_config.num_agents:
            remaining: list[tuple[int, int]] = []
            for y in range(height):
                for x in range(width):
                    if level[y, x] == "empty" and (y, x) not in used_positions:
                        remaining.append((y, x))

            np.random.shuffle(remaining)
            needed = self.dynamic_config.num_agents - len(agent_positions)
            agent_positions.extend(remaining[:needed])

        for y, x in agent_positions[: self.dynamic_config.num_agents]:
            level[y, x] = "agent.agent"

        return GameMap(level)


__all__ = ["DynamicAgentAsciiMapBuilder"]

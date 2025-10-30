from __future__ import annotations

from collections import deque
from typing import Annotated, Any

import numpy as np
from pydantic import Field, StringConstraints, field_validator

from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mapgen.utils.ascii_grid import merge_with_global_defaults


class AsciiMapBuilderConfig(MapBuilderConfig["AsciiMapBuilder"]):
    """Configuration for the ASCII map builder.

    Multi-cell grouping uses object type heuristics to pick layers: types whose
    name starts with "agent" default to the agent layer (0); everything else
    defaults to the object layer (1). Use "auto_group_layer_overrides" to set a
    specific layer per grouped type when the heuristic is insufficient.
    """

    map_data: list[list[str]]
    char_to_name_map: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=1)],
        Annotated[str, StringConstraints(pattern=r"^[\w\.]+$")],
    ]
    auto_group_types: list[str] = Field(
        default_factory=list,
        description="Object types that should be merged into multi-cell groups via adjacency.",
    )
    auto_group_layer_overrides: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-type layer overrides for grouped objects (0 = agents, 1 = objects).",
    )

    @field_validator("map_data", mode="before")
    @classmethod
    def _validate_multiline_map_data(cls, value: Any):
        if isinstance(value, str):
            return [list(line) for line in value.splitlines()]
        if isinstance(value, list) and value and isinstance(value[0], str):
            return [list(line) for line in value]
        return value

    @field_validator("map_data", mode="after")
    @classmethod
    def _validate_map_data_lines(cls, map_data: list[str]):
        width = len(map_data[0])
        for i, line in enumerate(map_data):
            assert len(line) == width, (
                f"Line {i} has length {len(line)}, expected {width}. All lines in ASCII map must have the same length."
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


class AsciiMapBuilder(MapBuilder[AsciiMapBuilderConfig]):
    """
    Builds a game map from an ASCII string.
    """

    def __init__(self, config: AsciiMapBuilderConfig):
        super().__init__(config)

        self._level = np.array([list(line) for line in config.map_data], dtype="U6")
        self._level = np.vectorize(self._char_to_object_name)(self._level)

    def _char_to_object_name(self, char: str) -> str:
        """Convert a map character to an object name."""
        if char in self.config.char_to_name_map:
            return self.config.char_to_name_map[char]
        raise ValueError(f"Unknown character: '{char}'. Available: {list(self.config.char_to_name_map.keys())}")

    def _find_connected_components(self, grid: np.ndarray, object_name: str) -> list[list[tuple[int, int]]]:
        """Find connected components with 4-neighbor grouping for grouping heuristics.

        Note: Engine does not require connectivity; this is for ASCII authoring
        convenience when merging adjacent same-type cells into one object.
        """
        height, width = grid.shape
        visited = np.zeros((height, width), dtype=bool)
        components = []

        def bfs(start_r: int, start_c: int) -> list[tuple[int, int]]:
            """Breadth-first search to find all connected cells."""
            component = []
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True

            while queue:
                r, c = queue.popleft()
                component.append((r, c))

                # Check 4 neighbors (no diagonals)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and grid[nr, nc] == object_name:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            return component

        # Find all components (size 1 components ignored)
        for r in range(height):
            for c in range(width):
                if grid[r, c] == object_name and not visited[r, c]:
                    component = bfs(r, c)
                    # Only interested in multi-cell components
                    if len(component) > 1:
                        components.append(component)

        return components

    def build(self) -> GameMap:
        grid = self._level.copy()
        multi_cell_groups: list[tuple[int, int, int, list[tuple[int, int]]]] = []

        # Process each auto_group type
        layer_overrides = self.config.auto_group_layer_overrides

        for object_name in self.config.auto_group_types:
            components = self._find_connected_components(grid, object_name)

            # Determine layer: agents on layer 0, other objects on layer 1
            # GridLayer::AgentLayer = 0, GridLayer::ObjectLayer = 1
            if object_name in layer_overrides:
                layer = layer_overrides[object_name]
            else:
                layer = 0 if object_name.startswith("agent") else 1

            for component in components:
                # Sort cells to have consistent primary selection (top-left)
                sorted_component = sorted(component)
                primary_r, primary_c = sorted_component[0]
                extra_cells = sorted_component[1:]

                # Keep object at primary location, set others to empty
                for r, c in extra_cells:
                    grid[r, c] = "empty"

                # Store grouping metadata with correct layer
                multi_cell_groups.append((primary_r, primary_c, layer, extra_cells))

        return GameMap(grid, multi_cell_groups)

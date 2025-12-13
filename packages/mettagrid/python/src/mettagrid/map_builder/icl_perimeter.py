"""ICL Perimeter Map Builder - A variant of PerimeterInContextMapBuilder for multi-agent ICL training.

This builder creates maps with:
- An outer walkable border (so agents can surround objects)
- An inner object border (extractors, assemblers, chests, etc.)
- Interior space where agents spawn

This allows agents to properly surround assemblers and other objects that require
multi-agent interaction.
"""

import numpy as np

from mettagrid.map_builder.map_builder import GameMap, MapBuilder
from mettagrid.map_builder.perimeter_incontext import PerimeterInContextMapBuilderConfig


class ICLPerimeterMapBuilderConfig(PerimeterInContextMapBuilderConfig):
    """Configuration for ICL perimeter maps with double-border structure.

    Extends PerimeterInContextMapBuilderConfig to add:
    - Support for multiple agents (placed in interior)
    - An outer walkable border so agents can surround objects

    The resulting map structure:
        [wall border if border_width > 0]
        [outer walkable border]
        [inner object border with extractors/assemblers/etc]
        [interior where agents spawn]
    """

    # Reset _builder_cls to prevent MapBuilder.__init_subclass__ from creating
    # an unpicklable local CloneConfig class when this config inherits from
    # PerimeterInContextMapBuilderConfig (which is already bound to PerimeterInContextMapBuilder)
    _builder_cls = None  # type: ignore[assignment]

    # Number of agents to place in the interior
    num_agents: int = 4

    # Width of the outer walkable border (for agents to surround objects)
    outer_walkable_width: int = 1


class ICLPerimeterMapBuilder(MapBuilder[ICLPerimeterMapBuilderConfig]):
    """Map builder for ICL training with double-border structure."""

    def __init__(self, config: ICLPerimeterMapBuilderConfig):
        super().__init__(config)
        self._rng = np.random.default_rng(self.config.seed)

    def build(self) -> GameMap:
        """Build map with config's num_agents spawn points."""
        return self._build_with_agents(self.config.num_agents)

    def build_for_num_agents(self, num_agents: int) -> GameMap:
        """Override to use our internal agent placement logic."""
        return self._build_with_agents(num_agents)

    def _build_with_agents(self, num_agents: int) -> GameMap:
        height = self.config.height
        width = self.config.width

        # Create empty grid
        grid = np.full((height, width), "empty", dtype="<U50")

        # Layer 1: Wall border (if specified)
        wall_offset = 0
        if self.config.border_width > 0:
            for i in range(self.config.border_width):
                grid[i, :] = self.config.border_object
                grid[height - 1 - i, :] = self.config.border_object
                grid[:, i] = self.config.border_object
                grid[:, width - 1 - i] = self.config.border_object
            wall_offset = self.config.border_width

        # Layer 2: Outer walkable border (stays empty - agents can walk here)
        walkable_offset = wall_offset + self.config.outer_walkable_width

        # Layer 3: Inner object perimeter (where objects are placed)
        object_ring_offset = walkable_offset

        # Calculate the object ring positions (one cell thick ring)
        object_positions = []

        # Top row of object ring
        for j in range(object_ring_offset, width - object_ring_offset):
            object_positions.append((object_ring_offset, j))

        # Bottom row of object ring
        for j in range(object_ring_offset, width - object_ring_offset):
            object_positions.append((height - 1 - object_ring_offset, j))

        # Left column of object ring (excluding corners already added)
        for i in range(object_ring_offset + 1, height - object_ring_offset - 1):
            object_positions.append((i, object_ring_offset))

        # Right column of object ring (excluding corners already added)
        for i in range(object_ring_offset + 1, height - object_ring_offset - 1):
            object_positions.append((i, width - 1 - object_ring_offset))

        # Shuffle object positions for random placement
        self._rng.shuffle(object_positions)

        # Prepare objects to place
        object_symbols = []
        for obj_name, count in self.config.objects.items():
            object_symbols.extend([obj_name] * count)

        # Place objects on the object ring
        num_placeable = min(len(object_symbols), len(object_positions))
        for idx in range(num_placeable):
            r, c = object_positions[idx]
            grid[r, c] = object_symbols[idx]

        # Layer 4: Place agents in the interior
        # Interior is everything inside the object ring
        interior_offset = object_ring_offset + 1

        # Find all interior positions
        interior_positions = []
        for i in range(interior_offset, height - interior_offset):
            for j in range(interior_offset, width - interior_offset):
                if grid[i, j] == "empty":
                    interior_positions.append((i, j))

        # Shuffle and place agents
        self._rng.shuffle(interior_positions)

        if len(interior_positions) < num_agents:
            raise ValueError(
                f"Not enough interior space for {num_agents} agents. "
                f"Available: {len(interior_positions)}. "
                f"Try increasing map size or reducing num_agents."
            )

        for idx in range(num_agents):
            r, c = interior_positions[idx]
            grid[r, c] = "agent.agent"

        return GameMap(grid)

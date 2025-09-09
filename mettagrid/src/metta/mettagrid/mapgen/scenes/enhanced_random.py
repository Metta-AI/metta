"""Enhanced Random scene using int-based storage and type constants.

This demonstrates the migration pattern for updating scene classes to support
both legacy string-based and new int-based grids.
"""

from typing import Union

import numpy as np

from metta.mettagrid.config import Config
from metta.mettagrid.mapgen.enhanced_scene import EnhancedScene


class EnhancedRandomParams(Config):
    """Parameters for enhanced random scene generation."""

    objects: dict[str, int] = {}
    agents: Union[int, dict[str, int]] = 0
    too_many_is_ok: bool = True


class EnhancedRandom(EnhancedScene[EnhancedRandomParams]):
    """
    Enhanced random scene that supports both int-based and string-based grids.

    This scene fills the grid with random objects and agents based on configuration.
    It automatically detects the grid format and uses appropriate operations.

    Key improvements over legacy Random scene:
    - Type-safe object placement using constants
    - Support for both grid formats during migration
    - Better error handling for unknown objects
    - Performance optimizations for int-based grids
    """

    def render(self):
        """Render the random scene with objects and agents."""
        params = self.params

        # Find empty positions in the grid
        empty_positions = self.find_empty_positions()
        empty_count = len(empty_positions)

        if empty_count == 0:
            return  # No empty space to place objects

        # Prepare agent values for placement
        agent_values = self.prepare_agent_values(params.agents)

        # Prepare object values for placement
        object_values = self.prepare_object_values(params.objects)

        # Combine all values to place
        all_placement_values = object_values + agent_values

        # Check if we have too many objects for available space
        if not params.too_many_is_ok and len(all_placement_values) > empty_count:
            raise ValueError(f"Too many objects for available empty cells: {len(all_placement_values)} > {empty_count}")

        # Truncate if we have more objects than space
        if len(all_placement_values) > empty_count:
            all_placement_values = all_placement_values[:empty_count]

        if not all_placement_values:
            return  # Nothing to place

        # Place objects randomly
        self.place_objects_randomly(all_placement_values, empty_positions)

    def render_legacy_compatible(self):
        """
        Alternative render method that maintains exact compatibility with legacy Random scene.

        This method replicates the exact behavior of the original Random scene,
        including the specific way it handles grid operations and object placement.
        Use this if you need exact backward compatibility during migration.
        """
        height, width, params = self.height, self.width, self.params

        # Replicate legacy agent preparation logic
        if isinstance(params.agents, int):
            if self._grid_is_int:
                from metta.mettagrid.object_types import ObjectTypes

                agents = [ObjectTypes.AGENT_DEFAULT] * params.agents
            else:
                agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = []
            for agent, na in params.agents.items():
                if self._grid_is_int:
                    try:
                        agent_name = f"agent.{agent}"
                        type_id = self.get_object_type_id(agent_name)
                        agents.extend([type_id] * na)
                    except KeyError:
                        from metta.mettagrid.object_types import ObjectTypes

                        agents.extend([ObjectTypes.AGENT_DEFAULT] * na)
                else:
                    agents.extend([f"agent.{agent}"] * na)
        else:
            raise ValueError(f"Invalid agents: {params.agents}")

        # Find empty cells using legacy-compatible approach
        empty_mask = self.grid == self._empty_value
        empty_count = np.sum(empty_mask)
        empty_indices = np.where(empty_mask.flatten())[0]

        # Prepare objects list
        symbols = []
        for obj_name, count in params.objects.items():
            if self._grid_is_int:
                try:
                    type_id = self.get_object_type_id(obj_name)
                    symbols.extend([type_id] * count)
                except KeyError:
                    print(f"Warning: Unknown object {obj_name}, skipping")
                    continue
            else:
                symbols.extend([obj_name] * count)

        symbols.extend(agents)

        if not params.too_many_is_ok and len(symbols) > empty_count:
            raise ValueError(f"Too many objects for available empty cells: {len(symbols)} > {empty_count}")

        # Truncate to fit available space
        symbols = symbols[:empty_count]

        if not symbols:
            return

        # Shuffle symbols and place them (legacy approach)
        if self._grid_is_int:
            symbols_array = np.array(symbols, dtype=np.uint8)
        else:
            symbols_array = np.array(symbols).astype(str)

        self.rng.shuffle(symbols_array)
        self.rng.shuffle(empty_indices)

        # Take only as many indices as we have symbols
        selected_indices = empty_indices[: len(symbols_array)]

        # Create a flat copy of the grid and place symbols
        flat_grid = self.grid.flatten()
        flat_grid[selected_indices] = symbols_array

        # Reshape back to original dimensions
        self.grid[:] = flat_grid.reshape(height, width)

    def get_placement_stats(self) -> dict:
        """Get statistics about what would be placed with current parameters.

        Returns:
            Dictionary with placement statistics and validation info
        """
        params = self.params

        # Calculate what would be placed
        agent_values = self.prepare_agent_values(params.agents)
        object_values = self.prepare_object_values(params.objects)
        total_objects = len(agent_values) + len(object_values)

        empty_count = self.count_empty_cells()

        stats = {
            "grid_format": "int" if self._grid_is_int else "string",
            "empty_cells": empty_count,
            "total_objects_to_place": total_objects,
            "agent_count": len(agent_values),
            "non_agent_objects": len(object_values),
            "fits_in_available_space": total_objects <= empty_count,
            "would_be_truncated": total_objects > empty_count and params.too_many_is_ok,
            "would_raise_error": total_objects > empty_count and not params.too_many_is_ok,
        }

        # Break down objects by type
        if self._grid_is_int:
            object_breakdown = {}
            for obj_name, count in params.objects.items():
                try:
                    type_id = self.get_object_type_id(obj_name)
                    object_breakdown[f"{obj_name} (id:{type_id})"] = count
                except KeyError:
                    object_breakdown[f"{obj_name} (unknown)"] = count
            stats["object_breakdown"] = object_breakdown
        else:
            stats["object_breakdown"] = dict(params.objects)

        return stats

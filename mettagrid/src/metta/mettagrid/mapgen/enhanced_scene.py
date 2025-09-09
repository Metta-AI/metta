"""Enhanced scene base class with int-based grid support.

This module provides enhanced scene classes that support both legacy string-based
and new int-based grids during the migration period.
"""

from typing import Optional, Union

import numpy as np

from metta.mettagrid.mapgen.scene import ParamsT, Scene
from metta.mettagrid.object_types import ObjectTypes
from metta.mettagrid.type_mapping import TypeMapping


class EnhancedScene(Scene[ParamsT]):
    """
    Enhanced scene base class with int-based grid support.

    This scene class can work with both legacy string-based grids and new int-based grids.
    It provides utility methods for type-safe object placement and grid manipulation.

    During the migration period, scenes inheriting from this class will automatically
    detect the grid format and use appropriate operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up type mapping for int-based operations
        self.type_mapping = TypeMapping()  # Use standard mappings
        self._grid_is_int = self._detect_int_grid()

        # Cache for performance
        self._empty_value = self._get_empty_value()

    def _detect_int_grid(self) -> bool:
        """Detect if the current grid uses int-based format."""
        return self.grid.dtype == np.uint8

    def _get_empty_value(self) -> Union[str, int]:
        """Get the empty value for the current grid format."""
        if self._grid_is_int:
            return ObjectTypes.EMPTY
        else:
            return "empty"

    def get_object_type_id(self, obj_name: str) -> int:
        """Get type ID for object name.

        Args:
            obj_name: Object name to look up

        Returns:
            Type ID for the object

        Raises:
            KeyError: If object name is not found
        """
        return self.type_mapping.get_type_id(obj_name)

    def get_object_name(self, type_id: int) -> str:
        """Get object name for type ID.

        Args:
            type_id: Type ID to look up

        Returns:
            Object name for the type ID

        Raises:
            KeyError: If type ID is not found
        """
        return self.type_mapping.get_name(type_id)

    def is_empty(self, row: int, col: int) -> bool:
        """Check if a grid position is empty."""
        if self._grid_is_int:
            return self.grid[row, col] == ObjectTypes.EMPTY
        else:
            return self.grid[row, col] == "empty"

    def set_object(self, row: int, col: int, obj_name: str):
        """Set object at grid position using object name.

        Args:
            row: Grid row index
            col: Grid column index
            obj_name: Object name to place
        """
        if self._grid_is_int:
            try:
                type_id = self.get_object_type_id(obj_name)
                self.grid[row, col] = type_id
            except KeyError:
                # Unknown object - skip or use fallback
                print(f"Warning: Unknown object {obj_name}, skipping placement")
        else:
            self.grid[row, col] = obj_name

    def set_object_by_type_id(self, row: int, col: int, type_id: int):
        """Set object at grid position using type ID.

        Args:
            row: Grid row index
            col: Grid column index
            type_id: Type ID to place
        """
        if self._grid_is_int:
            self.grid[row, col] = type_id
        else:
            try:
                obj_name = self.get_object_name(type_id)
                self.grid[row, col] = obj_name
            except KeyError:
                # Unknown type ID - use generic name
                self.grid[row, col] = f"unknown_type_{type_id}"

    def find_empty_positions(self) -> list[tuple[int, int]]:
        """Find all empty positions in the grid.

        Returns:
            List of (row, col) tuples for empty positions
        """
        if self._grid_is_int:
            empty_mask = self.grid == ObjectTypes.EMPTY
        else:
            empty_mask = self.grid == "empty"

        empty_positions = np.where(empty_mask)
        return list(zip(empty_positions[0], empty_positions[1], strict=False))

    def count_empty_cells(self) -> int:
        """Count the number of empty cells in the grid."""
        if self._grid_is_int:
            return np.sum(self.grid == ObjectTypes.EMPTY)
        else:
            return np.sum(self.grid == "empty")

    def prepare_agent_values(self, agents_config: Union[int, dict[str, int]]) -> list[Union[str, int]]:
        """Prepare agent values for placement based on grid format.

        Args:
            agents_config: Agent configuration (count or dict of group names to counts)

        Returns:
            List of agent values ready for placement
        """
        if isinstance(agents_config, int):
            # Simple count of default agents
            if self._grid_is_int:
                return [ObjectTypes.AGENT_DEFAULT] * agents_config
            else:
                return ["agent.agent"] * agents_config

        elif isinstance(agents_config, dict):
            # Dict of group names to counts
            agent_values = []
            for agent_group, count in agents_config.items():
                if self._grid_is_int:
                    # Try to get specific type ID, fallback to default
                    try:
                        agent_name = f"agent.{agent_group}"
                        type_id = self.get_object_type_id(agent_name)
                        agent_values.extend([type_id] * count)
                    except KeyError:
                        agent_values.extend([ObjectTypes.AGENT_DEFAULT] * count)
                else:
                    agent_values.extend([f"agent.{agent_group}"] * count)
            return agent_values

        else:
            raise ValueError(f"Invalid agents configuration: {agents_config}")

    def prepare_object_values(self, objects_config: dict[str, int]) -> list[Union[str, int]]:
        """Prepare object values for placement based on grid format.

        Args:
            objects_config: Dict of object names to counts

        Returns:
            List of object values ready for placement
        """
        object_values = []

        for obj_name, count in objects_config.items():
            if self._grid_is_int:
                try:
                    type_id = self.get_object_type_id(obj_name)
                    object_values.extend([type_id] * count)
                except KeyError:
                    print(f"Warning: Unknown object {obj_name}, skipping")
                    continue
            else:
                object_values.extend([obj_name] * count)

        return object_values

    def place_objects_randomly(
        self, placement_values: list[Union[str, int]], empty_positions: Optional[list[tuple[int, int]]] = None
    ):
        """Place objects randomly in empty positions.

        Args:
            placement_values: List of values to place (strings for legacy, ints for new format)
            empty_positions: Optional list of empty positions. If None, will find automatically.
        """
        if empty_positions is None:
            empty_positions = self.find_empty_positions()

        if len(placement_values) > len(empty_positions):
            print(
                f"Warning: Not enough empty positions ({len(empty_positions)}) "
                f"for all objects ({len(placement_values)})"
            )
            placement_values = placement_values[: len(empty_positions)]

        if not placement_values:
            return

        # Shuffle placement positions
        selected_positions = self.rng.choice(len(empty_positions), size=len(placement_values), replace=False)

        # Place objects
        for i, pos_idx in enumerate(selected_positions):
            row, col = empty_positions[pos_idx]
            value = placement_values[i]

            if self._grid_is_int:
                self.grid[row, col] = value
            else:
                self.grid[row, col] = value

    def clear_area(
        self, start_row: int = 0, start_col: int = 0, end_row: Optional[int] = None, end_col: Optional[int] = None
    ):
        """Clear an area of the grid (set to empty).

        Args:
            start_row: Starting row (inclusive)
            start_col: Starting column (inclusive)
            end_row: Ending row (exclusive). If None, uses grid height.
            end_col: Ending column (exclusive). If None, uses grid width.
        """
        if end_row is None:
            end_row = self.height
        if end_col is None:
            end_col = self.width

        self.grid[start_row:end_row, start_col:end_col] = self._empty_value

    def get_grid_summary(self) -> dict:
        """Get summary statistics of the current grid contents.

        Returns:
            Dictionary with object counts and grid format info
        """
        summary = {
            "format": "int" if self._grid_is_int else "string",
            "shape": self.grid.shape,
            "empty_cells": self.count_empty_cells(),
            "total_cells": self.grid.size,
        }

        if self._grid_is_int:
            # Count each type ID
            unique_ids, counts = np.unique(self.grid, return_counts=True)
            type_counts = {}
            for type_id, count in zip(unique_ids, counts, strict=False):
                try:
                    obj_name = self.get_object_name(int(type_id))
                    type_counts[obj_name] = int(count)
                except KeyError:
                    type_counts[f"unknown_type_{type_id}"] = int(count)
            summary["object_counts"] = type_counts
        else:
            # Count each string value
            unique_strs, counts = np.unique(self.grid, return_counts=True)
            summary["object_counts"] = {str(name): int(count) for name, count in zip(unique_strs, counts, strict=False)}

        return summary

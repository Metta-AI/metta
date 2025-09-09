"""Enhanced RandomMapBuilder using int-based storage and GameConfig parameterization.

This demonstrates the new architecture during the migration period.
"""

from typing import Optional, Union

import numpy as np

from metta.mettagrid.map_builder.map_builder import EnhancedMapBuilder, GameMap, MapBuilderConfig
from metta.mettagrid.map_builder.utils import draw_border
from metta.mettagrid.mettagrid_config import GameConfig
from metta.mettagrid.object_types import ObjectTypes


class EnhancedRandomMapBuilder(EnhancedMapBuilder):
    """Enhanced RandomMapBuilder with int-based storage and GameConfig validation."""

    class Config(MapBuilderConfig["EnhancedRandomMapBuilder"]):
        """Configuration for building an enhanced random map with type validation."""

        seed: Optional[int] = None
        width: int = 10
        height: int = 10
        objects: dict[str, int] = {}
        agents: Union[int, dict[str, int]] = 0
        border_width: int = 0
        border_object: str = "wall"

        # New options for enhanced builder
        use_int_format: bool = True  # Generate int-based maps by default
        validate_objects: bool = True  # Validate objects against GameConfig

    def __init__(self, config: Config, game_config: Optional[GameConfig] = None):
        """Initialize enhanced random map builder.

        Args:
            config: Map builder configuration
            game_config: Optional game configuration for validation and type mapping
        """
        super().__init__(config, game_config)
        self._config = config
        self._rng = np.random.default_rng(self._config.seed)

        # Validate objects if requested and GameConfig is provided
        if self._config.validate_objects and game_config:
            self._validate_object_config()

    def _validate_object_config(self):
        """Validate that all configured objects are available in GameConfig."""
        invalid_objects = []

        # Check border object
        if not self.validate_object_availability(self._config.border_object):
            invalid_objects.append(self._config.border_object)

        # Check configured objects
        for obj_name in self._config.objects.keys():
            if not self.validate_object_availability(obj_name):
                invalid_objects.append(obj_name)

        if invalid_objects:
            raise ValueError(
                f"Objects not available in GameConfig: {invalid_objects}. "
                f"Available objects: {list(self.game_config.objects.keys()) if self.game_config else 'None'}"
            )

    def build(self) -> GameMap:
        """Build the random map using either int-based or legacy format."""
        # Reset RNG to ensure deterministic builds across multiple calls
        if self._config.seed is not None:
            self._rng = np.random.default_rng(self._config.seed)

        if self._config.use_int_format and self.supports_int_format():
            return self._build_int_format()
        else:
            return self._build_legacy_format()

    def _build_int_format(self) -> GameMap:
        """Build map using new int-based format."""
        # Create empty grid (filled with type_id 0 = empty)
        game_map = self.create_int_map(self._config.height, self._config.width, ObjectTypes.EMPTY)
        grid = game_map.grid

        # Draw border if needed
        if self._config.border_width > 0:
            try:
                border_type_id = self.get_type_id(self._config.border_object)
                draw_border(grid, self._config.border_width, border_type_id)
            except KeyError:
                # Fallback to wall if border object not found
                draw_border(grid, self._config.border_width, ObjectTypes.WALL)

        # Calculate inner area for object placement
        inner_height, inner_width, inner_area = self._calculate_inner_area()
        if inner_area <= 0:
            return game_map

        # Prepare agent type IDs
        agent_type_ids = self._prepare_agent_type_ids()

        # Convert object names to type IDs
        object_type_ids = self._prepare_object_type_ids()

        # Check capacity and place objects
        total_objects = len(object_type_ids) + len(agent_type_ids)
        while total_objects > inner_area:
            if not self._reduce_object_counts():
                break
            object_type_ids = self._prepare_object_type_ids()
            total_objects = len(object_type_ids) + len(agent_type_ids)

        # Create and shuffle placement array
        placement_ids = object_type_ids + agent_type_ids
        placement_ids.extend([ObjectTypes.EMPTY] * (inner_area - len(placement_ids)))

        placement_array = np.array(placement_ids, dtype=np.uint8)
        self._rng.shuffle(placement_array)
        inner_grid = placement_array.reshape(inner_height, inner_width)

        # Place inner grid into main grid
        if self._config.border_width > 0:
            grid[
                self._config.border_width : self._config.border_width + inner_height,
                self._config.border_width : self._config.border_width + inner_width,
            ] = inner_grid
        else:
            game_map.grid[:] = inner_grid

        return game_map

    def _build_legacy_format(self) -> GameMap:
        """Build map using legacy string-based format (for backward compatibility)."""
        # Create empty grid with legacy format
        game_map = self.create_legacy_map(self._config.height, self._config.width, "empty")
        grid = game_map.grid

        # Draw border if needed
        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        # Calculate inner area
        inner_height, inner_width, inner_area = self._calculate_inner_area()
        if inner_area <= 0:
            return game_map

        # Prepare agent symbols (legacy format)
        if isinstance(self._config.agents, int):
            agents = ["agent.agent"] * self._config.agents
        elif isinstance(self._config.agents, dict):
            agents = [f"agent.{agent}" for agent, na in self._config.agents.items() for _ in range(na)]
        else:
            raise ValueError(f"Invalid agents configuration: {self._config.agents}")

        # Check capacity and reduce if needed
        total_objects = sum(self._config.objects.values()) + len(agents)
        while total_objects > inner_area:
            if not self._reduce_object_counts():
                break
            total_objects = sum(self._config.objects.values()) + len(agents)

        # Create symbols array
        symbols = []
        for obj_name, count in self._config.objects.items():
            symbols.extend([obj_name] * count)
        symbols.extend(agents)
        symbols.extend(["empty"] * (inner_area - len(symbols)))

        # Shuffle and place
        symbols_array = np.array(symbols).astype(str)
        self._rng.shuffle(symbols_array)
        inner_grid = symbols_array.reshape(inner_height, inner_width)

        # Place inner grid into main grid
        if self._config.border_width > 0:
            grid[
                self._config.border_width : self._config.border_width + inner_height,
                self._config.border_width : self._config.border_width + inner_width,
            ] = inner_grid
        else:
            game_map.grid[:] = inner_grid

        return game_map

    def _calculate_inner_area(self) -> tuple[int, int, int]:
        """Calculate inner area dimensions after accounting for border."""
        if self._config.border_width > 0:
            inner_height = max(0, self._config.height - 2 * self._config.border_width)
            inner_width = max(0, self._config.width - 2 * self._config.border_width)
        else:
            inner_height = self._config.height
            inner_width = self._config.width

        inner_area = inner_height * inner_width
        return inner_height, inner_width, inner_area

    def _prepare_agent_type_ids(self) -> list[int]:
        """Prepare agent type IDs for placement."""
        agent_type_ids = []

        if isinstance(self._config.agents, int):
            # Use default agent type for all agents
            agent_type_ids = [ObjectTypes.AGENT_DEFAULT] * self._config.agents
        elif isinstance(self._config.agents, dict):
            for agent_group, count in self._config.agents.items():
                try:
                    # Try to get type ID for specific agent group
                    agent_name = f"agent.{agent_group}"
                    type_id = self.get_type_id(agent_name)
                    agent_type_ids.extend([type_id] * count)
                except KeyError:
                    # Fallback to default agent type
                    agent_type_ids.extend([ObjectTypes.AGENT_DEFAULT] * count)
        else:
            raise ValueError(f"Invalid agents configuration: {self._config.agents}")

        return agent_type_ids

    def _prepare_object_type_ids(self) -> list[int]:
        """Prepare object type IDs for placement."""
        object_type_ids = []

        for obj_name, count in self._config.objects.items():
            try:
                type_id = self.get_type_id(obj_name)
                object_type_ids.extend([type_id] * count)
            except KeyError:
                # Skip unknown objects or use fallback
                print(f"Warning: Unknown object {obj_name}, skipping")
                continue

        return object_type_ids

    def _reduce_object_counts(self) -> bool:
        """Reduce object counts when they exceed available space."""
        all_ones = all(count <= 1 for count in self._config.objects.values())
        if all_ones and (not isinstance(self._config.agents, int) or self._config.agents <= 1):
            return False

        # Halve all object counts
        for obj_name in self._config.objects:
            self._config.objects[obj_name] = max(1, self._config.objects[obj_name] // 2)

        return True

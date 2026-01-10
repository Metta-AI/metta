"""Map management for harvest policy.

Builds and maintains a complete map representation as the agent explores.
Tracks all discovered objects (chargers, extractors, assemblers, chests, walls).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .harvest_policy import HarvestState
    from .types import CellType


class MapCellType(Enum):
    """Types of cells in the explored map."""
    UNKNOWN = 0         # Not yet explored
    FREE = 1           # Empty walkable space
    WALL = 2           # Impassable wall
    CHARGER = 3        # Energy charger station
    ASSEMBLER = 4      # Heart crafting station
    CHEST = 5          # Heart deposit location
    CARBON_EXTRACTOR = 6     # Carbon resource extractor
    OXYGEN_EXTRACTOR = 7     # Oxygen resource extractor
    GERMANIUM_EXTRACTOR = 8  # Germanium resource extractor
    SILICON_EXTRACTOR = 9    # Silicon resource extractor
    DEAD_END = 10      # Marked dead-end (explored but leads nowhere)


class MapManager:
    """Manages the agent's understanding of the map.

    Builds a complete 2D grid representation as the agent explores.
    Updates map cells based on observations and tracks all discovered objects.
    """

    def __init__(self, map_height: int, map_width: int, obs_hr: int, obs_wr: int, tag_names: dict, logger):
        """Initialize map manager.

        Args:
            map_height: Height of the map
            map_width: Width of the map
            obs_hr: Observation half-height radius
            obs_wr: Observation half-width radius
            tag_names: Mapping of tag IDs to names
            logger: Logger instance
        """
        self.map_height = map_height
        self.map_width = map_width
        self._obs_hr = obs_hr
        self._obs_wr = obs_wr
        self._tag_names = tag_names
        self._logger = logger

        # DEBUG: Log instance ID
        self._instance_id = id(self)
        logger.info(f"  MAP INIT: Created MapManager instance {self._instance_id}")

        # Initialize map grid - all cells start as UNKNOWN
        self.grid: list[list[MapCellType]] = [
            [MapCellType.UNKNOWN for _ in range(map_width)]
            for _ in range(map_height)
        ]

        # Track object positions by type (for quick lookup)
        self.chargers: set[tuple[int, int]] = set()
        self.assemblers: set[tuple[int, int]] = set()
        self.chests: set[tuple[int, int]] = set()
        self.carbon_extractors: set[tuple[int, int]] = set()
        self.oxygen_extractors: set[tuple[int, int]] = set()
        self.germanium_extractors: set[tuple[int, int]] = set()
        self.silicon_extractors: set[tuple[int, int]] = set()
        self.dead_ends: set[tuple[int, int]] = set()

    def update_from_observation(self, state: HarvestState):
        """Update map based on current observation.

        Processes all tokens in observation and updates corresponding map cells.

        Args:
            state: Current agent state with observation
        """
        if state.current_obs is None or not state.current_obs.tokens:
            return

        # Process each token in observation
        for tok in state.current_obs.tokens:
            if tok.feature.name != "tag":
                continue

            tag_name = self._tag_names.get(tok.value, "").lower()

            # Convert observation position to world position
            obs_r, obs_c = tok.location
            world_r = obs_r - self._obs_hr + state.row
            world_c = obs_c - self._obs_wr + state.col

            # Check bounds
            if not self._is_valid_position(world_r, world_c):
                continue

            # Update map cell based on tag type
            self._update_cell(world_r, world_c, tag_name)

        # Also mark all visible cells as explored (FREE if no object)
        self._mark_visible_cells_as_explored(state)

    def _update_cell(self, row: int, col: int, tag_name: str):
        """Update a single map cell based on tag name.

        CRITICAL: Don't overwrite cells learned as WALL from failed moves.
        Game state (failed moves) is ground truth - observations can be wrong.

        Args:
            row: Cell row
            col: Cell column
            tag_name: Name of the tag at this position
        """
        pos = (row, col)

        # Don't overwrite walls learned from failed moves - game state is truth
        if self.grid[row][col] == MapCellType.WALL:
            return

        # Charger
        if "charger" in tag_name:
            self.grid[row][col] = MapCellType.CHARGER
            if pos not in self.chargers:
                self.chargers.add(pos)
                self._logger.info(f"  MAP: Charger at {pos}")

        # Wall
        elif "wall" in tag_name:
            self.grid[row][col] = MapCellType.WALL

        # Assembler
        elif "assembler" in tag_name:
            self.grid[row][col] = MapCellType.ASSEMBLER
            if pos not in self.assemblers:
                self.assemblers.add(pos)
                self._logger.info(f"  MAP: Assembler at {pos}")

        # Chest
        elif "chest" in tag_name:
            self.grid[row][col] = MapCellType.CHEST
            if pos not in self.chests:
                self.chests.add(pos)
                self._logger.info(f"  MAP: Chest at {pos}")

        # Extractors
        elif "carbon_extractor" in tag_name:
            self.grid[row][col] = MapCellType.CARBON_EXTRACTOR
            if pos not in self.carbon_extractors:
                self.carbon_extractors.add(pos)
                self._logger.info(f"  MAP: Carbon extractor at {pos}")

        elif "oxygen_extractor" in tag_name:
            self.grid[row][col] = MapCellType.OXYGEN_EXTRACTOR
            if pos not in self.oxygen_extractors:
                self.oxygen_extractors.add(pos)
                self._logger.info(f"  MAP: Oxygen extractor at {pos}")

        elif "germanium_extractor" in tag_name:
            self.grid[row][col] = MapCellType.GERMANIUM_EXTRACTOR
            if pos not in self.germanium_extractors:
                self.germanium_extractors.add(pos)
                self._logger.info(f"  MAP: Germanium extractor at {pos}")

        elif "silicon_extractor" in tag_name:
            self.grid[row][col] = MapCellType.SILICON_EXTRACTOR
            if pos not in self.silicon_extractors:
                self.silicon_extractors.add(pos)
                self._logger.info(f"  MAP: Silicon extractor at {pos} (instance {self._instance_id}, total={len(self.silicon_extractors)})")

    def _mark_visible_cells_as_explored(self, state: HarvestState):
        """Mark all cells in observation window as explored (if not already marked).

        Args:
            state: Current agent state
        """
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                world_r = obs_r - self._obs_hr + state.row
                world_c = obs_c - self._obs_wr + state.col

                if not self._is_valid_position(world_r, world_c):
                    continue

                # If cell is UNKNOWN, mark as FREE (no objects seen there)
                if self.grid[world_r][world_c] == MapCellType.UNKNOWN:
                    self.grid[world_r][world_c] = MapCellType.FREE

    def mark_dead_end(self, row: int, col: int):
        """Mark a position as a dead-end.

        Args:
            row: Cell row
            col: Cell column
        """
        if self._is_valid_position(row, col):
            self.grid[row][col] = MapCellType.DEAD_END
            self.dead_ends.add((row, col))

    def mark_wall(self, row: int, col: int):
        """Mark a position as a wall (impassable obstacle).

        Called when agent tries to move to a cell but the move fails,
        indicating the cell is actually blocked even if not observed as a wall.

        CRITICAL: Game state is the source of truth. If a move fails, the cell
        IS blocked, regardless of what observations claimed. We mark extractors
        as WALL if moves fail, since observations can be wrong. We DON'T mark
        chargers/assemblers/chests since those are correctly non-traversable.

        Args:
            row: Cell row
            col: Cell column
        """
        if self._is_valid_position(row, col):
            current_type = self.grid[row][col]
            # Mark as wall if: UNKNOWN, FREE, or EXTRACTOR (extractors should be traversable)
            # Don't overwrite chargers/assemblers/chests (correctly non-traversable)
            if current_type in (MapCellType.UNKNOWN, MapCellType.FREE,
                              MapCellType.CARBON_EXTRACTOR, MapCellType.OXYGEN_EXTRACTOR,
                              MapCellType.GERMANIUM_EXTRACTOR, MapCellType.SILICON_EXTRACTOR):
                old_type = current_type.name
                self.grid[row][col] = MapCellType.WALL
                self._logger.debug(f"  MAP: Marked ({row}, {col}) as WALL due to failed move (was {old_type})")

    def is_traversable(self, row: int, col: int) -> bool:
        """Check if a cell is traversable for pathfinding.

        Only explored cells are traversable. UNKNOWN cells are NOT traversable
        because we don't know if they contain walls.

        Args:
            row: Cell row
            col: Cell column

        Returns:
            True if traversable, False otherwise
        """
        if not self._is_valid_position(row, col):
            return False

        cell_type = self.grid[row][col]

        # UNKNOWN cells are NOT traversable - we can't path through unexplored territory
        # WALL and DEAD_END are NOT traversable
        # CRITICAL FIX: CHARGER, ASSEMBLER, CHEST ARE traversable - you move ONTO them to use them!
        # They are destinations, not obstacles. Only WALL and DEAD_END block movement.
        return cell_type not in (
            MapCellType.UNKNOWN,
            MapCellType.WALL,
            MapCellType.DEAD_END
        )

    def get_nearest_object(
        self,
        current_pos: tuple[int, int],
        object_type: str
    ) -> Optional[tuple[int, int]]:
        """Find nearest object of specified type.

        Args:
            current_pos: Current agent position
            object_type: Type of object ("charger", "carbon_extractor", etc.)

        Returns:
            Position of nearest object, or None if none found
        """
        object_set_map = {
            "charger": self.chargers,
            "assembler": self.assemblers,
            "chest": self.chests,
            "carbon_extractor": self.carbon_extractors,
            "oxygen_extractor": self.oxygen_extractors,
            "germanium_extractor": self.germanium_extractors,
            "silicon_extractor": self.silicon_extractors,
        }

        object_set = object_set_map.get(object_type)
        if not object_set:
            return None

        if not object_set:
            return None

        # Find nearest using Manhattan distance
        nearest = min(
            object_set,
            key=lambda pos: abs(pos[0] - current_pos[0]) + abs(pos[1] - current_pos[1]),
            default=None
        )
        return nearest

    def get_explored_cell_count(self) -> int:
        """Count how many cells have been explored.

        Returns:
            Number of explored cells
        """
        count = 0
        for row in self.grid:
            for cell in row:
                if cell != MapCellType.UNKNOWN:
                    count += 1
        return count

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within map bounds.

        Args:
            row: Cell row
            col: Cell column

        Returns:
            True if valid, False otherwise
        """
        return 0 <= row < self.map_height and 0 <= col < self.map_width

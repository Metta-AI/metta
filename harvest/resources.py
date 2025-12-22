"""Resource management for harvest policy.

Handles resource tracking, deficit calculation, and extractor discovery.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .harvest_policy import HarvestState


class ResourceManager:
    """Manages resource gathering and extractor tracking."""

    # Recipe for crafting a heart
    HEART_RECIPE = {
        "carbon": 10,
        "oxygen": 10,
        "germanium": 2,
        "silicon": 30,
    }

    def __init__(self, logger):
        """Initialize resource manager.

        Args:
            logger: Logger instance
        """
        self._logger = logger

    def calculate_deficits(self, state: HarvestState) -> dict[str, int]:
        """Calculate what resources are still needed.

        Args:
            state: Current agent state

        Returns:
            Dict mapping resource name to deficit amount
        """
        deficits = {}

        # How many hearts do we need to craft?
        hearts_needed = max(0, state.target_hearts - state.hearts)

        for resource, amount_per_heart in self.HEART_RECIPE.items():
            needed = hearts_needed * amount_per_heart
            current = state.inventory.get(resource, 0)
            deficit = needed - current

            if deficit > 0:
                deficits[resource] = deficit

        return deficits

    def get_priority_order(self, deficits: dict[str, int]) -> list[str]:
        """Order resources by priority (deficit per unit).

        Silicon has highest deficit (30) so prioritized first.
        Germanium second (2), then carbon/oxygen (10 each).

        Args:
            deficits: Resource deficits from calculate_deficits

        Returns:
            List of resource names in priority order
        """
        # Sort by deficit amount (descending)
        priority = sorted(deficits.keys(), key=lambda r: deficits[r], reverse=True)
        return priority

    def find_extractor_positions(
        self,
        state: HarvestState,
        resource_type: str
    ) -> list[tuple[int, int]]:
        """Find all known extractors for a resource type.

        Filters out used/depleted extractors.

        Args:
            state: Current agent state
            resource_type: Type of resource ("carbon", "oxygen", etc.)

        Returns:
            List of extractor positions
        """
        extractor_key = f"{resource_type}_extractors"
        all_extractors = state.discovered_extractors.get(extractor_key, set())

        # Filter out used extractors
        available = [
            pos for pos in all_extractors
            if pos not in state.used_extractors
        ]

        return available

    def mark_extractor_used(self, state: HarvestState, position: tuple[int, int]):
        """Mark an extractor as used/depleted.

        Args:
            state: Current agent state (will be modified)
            position: Position of used extractor
        """
        state.used_extractors.add(position)

    def track_resource_found(self, state: HarvestState, resource_type: str):
        """Record that a resource type has been discovered.

        Args:
            state: Current agent state (will be modified)
            resource_type: Type of resource found
        """
        state.found_resource_types.add(resource_type)

    def all_resources_found(self, state: HarvestState) -> bool:
        """Check if all required resource types have been discovered.

        Returns:
            True if all 4 resources found, False otherwise
        """
        required = {"carbon", "oxygen", "germanium", "silicon"}
        return required.issubset(state.found_resource_types)

    def find_nearest_available_extractor(
        self,
        state: HarvestState,
        resource_type: str,
        map_manager: 'MapManager'
    ) -> Optional[tuple[int, int]]:
        """Find nearest available extractor using MapManager.

        Args:
            state: Current agent state
            resource_type: Type of resource ("carbon", "oxygen", "germanium", "silicon")
            map_manager: MapManager instance with complete map knowledge

        Returns:
            Position of nearest available extractor, or None if none found.
        """
        # Get extractor positions from MapManager
        extractor_attr = f"{resource_type}_extractors"
        all_extractors = getattr(map_manager, extractor_attr, set())

        map_instance = getattr(map_manager, '_instance_id', 'unknown')
        self._logger.debug(f"Step {state.step_count}: EXTRACTOR SEARCH: {resource_type} - found {len(all_extractors)} in MapManager instance {map_instance}, {len(state.used_extractors)} used")

        # Filter out used extractors
        available = [
            pos for pos in all_extractors
            if pos not in state.used_extractors
        ]

        if not available:
            self._logger.debug(f"Step {state.step_count}: EXTRACTOR SEARCH: No available {resource_type} extractors (all {len(all_extractors)} are used)")
            return None

        # Return nearest using Manhattan distance
        current = (state.row, state.col)
        nearest = min(
            available,
            key=lambda pos: abs(pos[0] - current[0]) + abs(pos[1] - current[1])
        )
        self._logger.debug(f"Step {state.step_count}: EXTRACTOR SEARCH: Found {len(available)} available {resource_type} extractors, nearest at {nearest}")
        return nearest

"""Resource management for harvest policy.

Handles resource tracking, deficit calculation, and extractor discovery.
"""
from typing import Optional

from .types import HarvestState


class ResourceManager:
    """Manages resource gathering and extractor tracking."""

    # Recipe for crafting a heart
    HEART_RECIPE = {
        "carbon": 10,
        "oxygen": 10,
        "germanium": 2,
        "silicon": 30,
    }

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

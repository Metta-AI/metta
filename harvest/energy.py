"""Energy management for harvest policy.

Handles charger tracking, energy safety, and recharge decisions.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .harvest_policy import HarvestState, ChargerInfo


class EnergyManager:
    """Manages energy safety and charger quality tracking."""

    def calculate_safe_radius(self, state: HarvestState) -> int:
        """Calculate how far agent can safely explore from nearest charger.

        Uses conservative formula: (energy - safety_margin) / distance_factor
        More chargers = more aggressive exploration.
        Scales with map size for large maps.

        Args:
            state: Current agent state

        Returns:
            Maximum safe distance in Manhattan distance
        """
        safety_margin = 10  # Always keep 10 energy buffer
        num_chargers = len(state.discovered_chargers)

        # Map size awareness: larger maps need larger exploration radius
        map_size = max(state.map_height, state.map_width)
        if map_size > 200:
            # Large maps (500x500, etc.) - very aggressive
            base_min_radius = 30
            distance_factor = 1.2 if num_chargers >= 5 else 1.5
        elif map_size > 100:
            # Medium-large maps (200x200) - moderately aggressive
            base_min_radius = 20
            distance_factor = 1.5 if num_chargers >= 5 else 1.8
        elif map_size > 50:
            # Medium maps (100x100) - slightly aggressive
            base_min_radius = 15
            distance_factor = 1.5 if num_chargers >= 5 else 2.0
        else:
            # Small maps (13x13, etc.) - conservative
            base_min_radius = 5
            distance_factor = 1.5 if num_chargers >= 5 else 2.0

        max_safe_distance = max(base_min_radius, int((state.energy - safety_margin) / distance_factor))
        return max_safe_distance

    def should_recharge(self, state: HarvestState) -> bool:
        """Determine if agent should enter recharge phase.

        Args:
            state: Current agent state

        Returns:
            True if should recharge, False otherwise
        """
        # Use mission-aware thresholds if available
        if state.mission_profile:
            recharge_low = state.mission_profile.recharge_low
        else:
            recharge_low = 20  # Default threshold

        return state.energy < recharge_low

    def find_best_charger(self, state: HarvestState) -> tuple[int, int]:
        """Find best charger based on reliability and distance.

        Prefers reliable chargers that are reasonably close.

        Args:
            state: Current agent state

        Returns:
            Position (row, col) of best charger
        """
        if not state.discovered_chargers:
            raise ValueError("No chargers discovered")

        # Score each charger: reliability * 100 - distance
        best_charger = None
        best_score = float('-inf')

        for charger_pos in state.discovered_chargers:
            # Get charger info (reliability)
            charger_info = state.charger_info.get(charger_pos)
            if charger_info:
                reliability = charger_info.success_rate
            else:
                reliability = 1.0  # Assume reliable if never tried

            # Calculate distance
            distance = abs(charger_pos[0] - state.row) + abs(charger_pos[1] - state.col)

            # Score: high reliability, low distance
            score = reliability * 100 - distance

            if score > best_score:
                best_score = score
                best_charger = charger_pos

        return best_charger

    def track_charger_approach(self, state: HarvestState, charger_pos: tuple[int, int]):
        """Record that agent is approaching a charger.

        Args:
            state: Current agent state (will be modified)
            charger_pos: Position of charger being approached
        """
        if charger_pos not in state.charger_info:
            state.charger_info[charger_pos] = ChargerInfo(position=charger_pos)

        state.charger_info[charger_pos].times_approached += 1
        state.charger_info[charger_pos].last_attempt_step = state.step_count

    def track_charger_success(self, state: HarvestState, charger_pos: tuple[int, int]):
        """Record successful charger use.

        Args:
            state: Current agent state (will be modified)
            charger_pos: Position of charger successfully used
        """
        if charger_pos not in state.charger_info:
            state.charger_info[charger_pos] = ChargerInfo(position=charger_pos)

        state.charger_info[charger_pos].times_successfully_used += 1

    def get_recharge_target(self, state: HarvestState) -> int:
        """Get energy level to recharge to.

        Args:
            state: Current agent state

        Returns:
            Target energy level
        """
        # Use mission-aware thresholds if available
        if state.mission_profile:
            return state.mission_profile.recharge_high
        else:
            return 85  # Default target

"""Navigation management for harvest policy.

Handles movement, stuck detection, and recovery.
"""
from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .harvest_policy import HarvestState


class NavigationManager:
    """Manages navigation and stuck recovery."""

    def __init__(self, logger):
        """Initialize navigation manager.

        Args:
            logger: Logger instance for debug output
        """
        self._logger = logger

    def calculate_direction_to(
        self,
        state: HarvestState,
        target: tuple[int, int]
    ) -> str:
        """Calculate which direction to move toward target.

        Uses simple greedy approach: reduce largest distance component.

        Args:
            state: Current agent state
            target: Target position (row, col)

        Returns:
            Direction to move ("north", "south", "east", "west")
        """
        dr = target[0] - state.row
        dc = target[1] - state.col

        # Prefer reducing larger distance component
        if abs(dr) > abs(dc):
            return "south" if dr > 0 else "north"
        else:
            return "east" if dc > 0 else "west"

    def is_stuck(self, state: HarvestState, threshold: int = 5) -> bool:
        """Check if agent is stuck at current position.

        Args:
            state: Current agent state
            threshold: Number of consecutive failed moves to consider stuck

        Returns:
            True if stuck, False otherwise
        """
        return state.consecutive_failed_moves >= threshold

    def handle_stuck_recovery(
        self,
        state: HarvestState,
        nearest_charger: Optional[tuple[int, int]] = None
    ) -> str:
        """Choose direction for stuck recovery.

        Prioritizes backtracking toward charger, with rotation to try all directions.

        Args:
            state: Current agent state
            nearest_charger: Position of nearest charger (if any)

        Returns:
            Direction to try
        """
        self._logger.warning(
            f"  NAVIGATION: STUCK ({state.consecutive_failed_moves} fails) - attempting recovery"
        )

        all_dirs = ["north", "south", "east", "west"]

        # If we have charger, prioritize direction toward it (backtracking)
        if nearest_charger:
            dr = nearest_charger[0] - state.row
            dc = nearest_charger[1] - state.col

            # Determine preferred direction (toward charger = out of corridor)
            if abs(dr) > abs(dc):
                preferred = "south" if dr > 0 else "north"
            else:
                preferred = "west" if dc < 0 else "east"

            # Reorder: preferred first, then others random
            all_dirs.remove(preferred)
            random.shuffle(all_dirs)
            all_dirs.insert(0, preferred)
        else:
            # No charger - just shuffle randomly
            random.shuffle(all_dirs)

        # Rotate through directions each step to try different ones
        # Use step_count to pick different direction each time
        direction_index = state.step_count % 4
        direction = all_dirs[direction_index]

        self._logger.info(f"  NAVIGATION: Trying {direction} (rotation {direction_index}/4)")
        return direction

    def verify_move_succeeded(
        self,
        state: HarvestState,
        prev_energy: Optional[int]
    ) -> bool:
        """Verify if last move actually succeeded.

        Uses energy-based verification: energy should decrease by 1 on successful move.

        Args:
            state: Current agent state
            prev_energy: Energy from previous step

        Returns:
            True if move succeeded, False if failed
        """
        if prev_energy is None:
            return True  # Can't verify first move

        # Energy-based verification
        expected_energy = prev_energy - 1

        if state.energy == prev_energy:
            # Energy unchanged → move failed (hit wall)
            return False
        elif state.energy == expected_energy:
            # Energy decreased by 1 → move succeeded
            return True
        elif state.energy > prev_energy:
            # Energy increased (charger) → move probably succeeded
            return True
        else:
            # Energy decreased differently → ambiguous, assume success
            return True

    def update_move_counters(self, state: HarvestState, move_succeeded: bool):
        """Update movement tracking counters.

        Args:
            state: Current agent state (will be modified)
            move_succeeded: Whether last move succeeded
        """
        if move_succeeded:
            state.consecutive_failed_moves = 0
            state.last_successful_move_step = state.step_count
        else:
            state.consecutive_failed_moves += 1

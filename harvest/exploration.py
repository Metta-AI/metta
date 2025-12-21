"""Exploration management for harvest policy.

Handles corridor detection, dead-end avoidance, and breadth-first exploration.
"""
import random
from typing import Optional

from .types import HarvestState


class ExplorationManager:
    """Manages exploration strategy with corridor detection and dead-end avoidance."""

    def __init__(self, obs_hr: int, obs_wr: int, tag_names: dict):
        """Initialize exploration manager.

        Args:
            obs_hr: Observation half-height radius
            obs_wr: Observation half-width radius
            tag_names: Mapping of tag IDs to names
        """
        self._obs_hr = obs_hr
        self._obs_wr = obs_wr
        self._tag_names = tag_names

    def choose_exploration_direction(
        self,
        state: HarvestState,
        nearest_charger: Optional[tuple[int, int]] = None,
        max_safe_distance: int = 20
    ) -> str:
        """Choose best direction for exploration.

        Uses corridor detection and dead-end avoidance to make smart choices.
        Prefers breadth-first exploration (expanding circles from charger).

        Args:
            state: Current agent state
            nearest_charger: Position of nearest charger (if any)
            max_safe_distance: Maximum distance from charger to explore

        Returns:
            Direction to move ("north", "south", "east", "west")
        """
        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

        # Step 1: Find passable directions (not blocked, not dead-ends)
        passable_directions = []
        for direction in ["north", "south", "east", "west"]:
            if not self._is_direction_clear(state, direction):
                continue  # Blocked by wall or obstacle

            dr, dc = dir_offsets[direction]
            target_r, target_c = state.row + dr, state.col + dc

            # Skip dead-ends
            if (target_r, target_c) in state.dead_end_positions:
                continue

            passable_directions.append((direction, target_r, target_c))

        if not passable_directions:
            # All directions blocked or lead to dead-ends - noop
            return "north"  # Default fallback

        # Step 2: Corridor detection - check if we're in a narrow passage
        in_corridor = len(passable_directions) <= 2

        # Step 3: Score each direction
        direction_scores = []
        for direction, target_r, target_c in passable_directions:
            score = 0

            # Base score: unexplored > explored
            if (target_r, target_c) not in state.explored_cells:
                score += 100  # Strong preference for new areas

            # Energy safety: how far is target from charger?
            if nearest_charger:
                dist_to_charger = abs(nearest_charger[0] - target_r) + abs(nearest_charger[1] - target_c)

                if dist_to_charger > max_safe_distance:
                    # Too far from charger - heavily penalize
                    score -= 1000
                else:
                    # Breadth-first: slightly prefer staying closer to charger
                    # This encourages expanding circles rather than deep corridors
                    score -= dist_to_charger * 0.5

            # Corridor behavior: if in corridor, prefer backtracking
            if in_corridor and nearest_charger:
                current_dist = abs(nearest_charger[0] - state.row) + abs(nearest_charger[1] - state.col)
                new_dist = abs(nearest_charger[0] - target_r) + abs(nearest_charger[1] - target_c)

                if new_dist < current_dist:
                    # Moving closer to charger (backtracking) - bonus
                    score += 50
                else:
                    # Going deeper into corridor - penalty
                    score -= 30

            # Small random component to break ties
            score += random.random() * 5

            direction_scores.append((direction, score))

        # Pick highest scoring direction
        best_direction = max(direction_scores, key=lambda x: x[1])[0]
        return best_direction

    def detect_corridor(self, state: HarvestState) -> bool:
        """Detect if agent is in a narrow corridor.

        A corridor is defined as having 2 or fewer passable directions.

        Args:
            state: Current agent state

        Returns:
            True if in corridor, False otherwise
        """
        passable_count = 0
        for direction in ["north", "south", "east", "west"]:
            if self._is_direction_clear(state, direction):
                passable_count += 1

        return passable_count <= 2

    def mark_dead_end(self, state: HarvestState):
        """Mark current position and recent path as dead-end.

        When stuck, marks the current position AND the last 5 positions
        in the movement history. This prevents re-entering the same corridor
        from further back.

        Args:
            state: Current agent state (will be modified)
        """
        current_pos = (state.row, state.col)
        positions_to_mark = [current_pos]

        # Mark last 5 positions in history as dead-end (the corridor leading here)
        for pos in state.position_history[-5:]:
            if pos not in state.dead_end_positions:
                positions_to_mark.append(pos)

        # Add all positions to dead-end set
        for pos in positions_to_mark:
            state.dead_end_positions.add(pos)

        return len(positions_to_mark)

    def _is_direction_clear(self, state: HarvestState, direction: str) -> bool:
        """Check if direction is clear in observation.

        Args:
            state: Current agent state
            direction: Direction to check

        Returns:
            True if clear, False if blocked
        """
        if state.current_obs is None:
            return True  # Optimistic default

        dir_offsets = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dir_offsets[direction]
        target_obs_pos = (self._obs_hr + dr, self._obs_wr + dc)

        # Check all tokens at target position
        for tok in state.current_obs.tokens:
            if tok.location == target_obs_pos and tok.feature.name == "tag":
                tag_name = self._tag_names.get(tok.value, "").lower()
                # Block on walls and agents only
                if "wall" in tag_name or tag_name == "agent":
                    return False

        return True

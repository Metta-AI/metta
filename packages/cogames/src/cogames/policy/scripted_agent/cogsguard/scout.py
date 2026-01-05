"""
Scout role for CoGsGuard.

Scouts explore the map and patrol to discover objects.
With scout gear, they get +400 HP and +100 energy capacity.

Scouts prioritize filling out their internal map by:
1. Moving towards unexplored frontiers (unexplored cells adjacent to explored cells)
2. Using systematic patrol when no clear frontier is available
"""

from __future__ import annotations

from collections import deque

from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role


class ScoutAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Scout agent: explore and patrol the map to fill out internal knowledge."""

    ROLE = Role.SCOUT

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute scout behavior: prioritize filling out unexplored areas."""
        # Try frontier-based exploration first
        frontier_action = self._explore_frontier(s)
        if frontier_action is not None:
            return frontier_action

        # Fall back to systematic patrol if no frontier found
        return self._patrol(s)

    def _explore_frontier(self, s: CogsguardAgentState) -> Action | None:
        """Find and move towards the nearest unexplored frontier.

        A frontier is an unexplored cell adjacent to an explored cell.
        Uses BFS to find the nearest reachable frontier.
        """
        # Check if explored grid is initialized
        if not s.explored or len(s.explored) == 0:
            return None

        # BFS to find nearest unexplored cell reachable through explored cells
        start = (s.row, s.col)
        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[tuple[int, int], tuple[int, int] | None]] = deque()

        # (current_pos, first_step_from_start)
        queue.append((start, None))

        directions = [("north", -1, 0), ("south", 1, 0), ("east", 0, 1), ("west", 0, -1)]

        while queue:
            pos, first_step = queue.popleft()
            r, c = pos

            for direction, dr, dc in directions:
                nr, nc = r + dr, c + dc

                # Check bounds
                if not (0 <= nr < s.map_height and 0 <= nc < s.map_width):
                    continue

                # Skip already visited
                if (nr, nc) in visited:
                    continue

                visited.add((nr, nc))

                # Check if this cell is unexplored - this is our target!
                if not s.explored[nr][nc]:
                    # Found unexplored cell - move towards it
                    # first_step tells us which direction to go from start
                    if first_step is None:
                        # Unexplored cell is directly adjacent - move there
                        if s.occupancy[nr][nc] == 1 and (nr, nc) not in s.agent_occupancy:  # FREE
                            if DEBUG and s.step_count <= 100:
                                print(f"[A{s.agent_id}] SCOUT_FRONTIER: Moving {direction} to unexplored ({nr},{nc})")
                            return self._actions.move.Move(direction)
                    else:
                        # Move towards the first step that leads to this frontier
                        if DEBUG and s.step_count <= 100:
                            explored_count = sum(sum(row) for row in s.explored)
                            total_cells = s.map_height * s.map_width
                            print(
                                f"[A{s.agent_id}] SCOUT_FRONTIER: Heading {first_step} towards "
                                f"frontier at ({nr},{nc}), explored={explored_count}/{total_cells}"
                            )
                        return self._actions.move.Move(first_step)

                # Only continue BFS through explored, free cells
                if s.explored[nr][nc] and s.occupancy[nr][nc] == 1:  # FREE
                    # Track first step direction if this is directly adjacent to start
                    next_first_step = first_step
                    if first_step is None and (r, c) == start:
                        next_first_step = direction

                    queue.append(((nr, nc), next_first_step))

        # No frontier found - map might be fully explored or we're boxed in
        if DEBUG and s.step_count % 50 == 0:
            explored_count = sum(sum(row) for row in s.explored)
            total_cells = s.map_height * s.map_width
            print(f"[A{s.agent_id}] SCOUT: No frontier found, explored={explored_count}/{total_cells}")
        return None

    def _patrol(self, s: CogsguardAgentState) -> Action:
        """Fall back patrol behavior when no frontier is available."""
        # Use longer exploration persistence for scouts
        if s.exploration_target is not None and isinstance(s.exploration_target, str):
            steps_in_direction = s.step_count - s.exploration_target_step
            # Scouts persist longer in each direction (25 steps vs 15)
            if steps_in_direction < 25:
                dr, dc = self._move_deltas.get(s.exploration_target, (0, 0))
                next_r, next_c = s.row + dr, s.col + dc
                if 0 <= next_r < s.map_height and 0 <= next_c < s.map_width:
                    if s.occupancy[next_r][next_c] == 1:  # FREE
                        if (next_r, next_c) not in s.agent_occupancy:
                            return self._actions.move.Move(s.exploration_target)

        # Cycle through directions systematically
        direction_cycle = ["north", "east", "south", "west"]
        current_dir = s.exploration_target
        if current_dir in direction_cycle:
            idx = direction_cycle.index(current_dir)
            next_idx = (idx + 1) % 4
        else:
            next_idx = 0

        for i in range(4):
            direction = direction_cycle[(next_idx + i) % 4]
            dr, dc = self._move_deltas[direction]
            next_r, next_c = s.row + dr, s.col + dc
            if 0 <= next_r < s.map_height and 0 <= next_c < s.map_width:
                if s.occupancy[next_r][next_c] == 1:  # FREE
                    if (next_r, next_c) not in s.agent_occupancy:
                        s.exploration_target = direction
                        s.exploration_target_step = s.step_count
                        return self._actions.move.Move(direction)

        return self._actions.noop.Noop()

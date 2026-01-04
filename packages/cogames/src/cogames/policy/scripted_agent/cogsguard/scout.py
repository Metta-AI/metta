"""
Scout role for CoGsGuard.

Scouts explore the map and patrol to discover objects.
With scout gear, they get +400 HP and +100 energy capacity.
"""

from __future__ import annotations

import random

from mettagrid.simulator import Action

from .policy import CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role


class ScoutAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Scout agent: explore and patrol the map."""

    ROLE = Role.SCOUT

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute scout behavior: explore and patrol."""
        # Scouts just explore extensively
        # They have high HP and energy so they can cover a lot of ground
        return self._patrol(s)

    def _patrol(self, s: CogsguardAgentState) -> Action:
        """Patrol by exploring in systematic directions."""
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

        # Pick new direction, prefer unexplored areas
        # Try to move towards map edges to cover more ground
        directions = ["north", "south", "east", "west"]

        # Bias towards edges based on current position
        center_r, center_c = s.map_height // 2, s.map_width // 2
        if s.row < center_r:
            # Closer to north, explore south more
            directions = ["south", "east", "west", "north"]
        elif s.row > center_r:
            directions = ["north", "east", "west", "south"]

        if s.col < center_c:
            directions = [d for d in directions if d != "west"] + ["west"]
        elif s.col > center_c:
            directions = [d for d in directions if d != "east"] + ["east"]

        random.shuffle(directions[:2])  # Shuffle top priorities

        for direction in directions:
            dr, dc = self._move_deltas[direction]
            next_r, next_c = s.row + dr, s.col + dc
            if 0 <= next_r < s.map_height and 0 <= next_c < s.map_width:
                if s.occupancy[next_r][next_c] == 1:  # FREE
                    if (next_r, next_c) not in s.agent_occupancy:
                        s.exploration_target = direction
                        s.exploration_target_step = s.step_count
                        return self._actions.move.Move(direction)

        return self._actions.noop.Noop()

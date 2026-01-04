"""
Scrambler role for CoGsGuard.

Scramblers find enemy-aligned supply depots and scramble them.
With scrambler gear, they get +200 HP.
"""

from __future__ import annotations

from typing import Optional

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role


class ScramblerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Scrambler agent: scramble enemy supply depots."""

    ROLE = Role.SCRAMBLER

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute scrambler behavior: find and scramble enemy depots."""
        # Check if we have heart (needed for scrambling)
        if s.heart < 1:
            # Need to get hearts
            return self._get_hearts(s)

        # Find an enemy depot to scramble
        target_depot = self._find_enemy_depot(s)

        if target_depot is None:
            # No known enemy depots, explore to find some
            return self._explore(s)

        # Navigate to depot
        if not is_adjacent((s.row, s.col), target_depot):
            return self._move_towards(s, target_depot, reach_adjacent=True)

        # Scramble the depot by bumping it
        return self._use_object_at(s, target_depot)

    def _get_hearts(self, s: CogsguardAgentState) -> Action:
        """Get hearts from chest or assembler."""
        # Try chest first
        chest_pos = s.stations.get("chest")
        if chest_pos is not None:
            if not is_adjacent((s.row, s.col), chest_pos):
                return self._move_towards(s, chest_pos, reach_adjacent=True)
            return self._use_object_at(s, chest_pos)

        # Try assembler
        assembler_pos = s.stations.get("assembler")
        if assembler_pos is not None:
            if not is_adjacent((s.row, s.col), assembler_pos):
                return self._move_towards(s, assembler_pos, reach_adjacent=True)
            return self._use_object_at(s, assembler_pos)

        return self._explore(s)

    def _find_enemy_depot(self, s: CogsguardAgentState) -> Optional[tuple[int, int]]:
        """Find an enemy-aligned supply depot to scramble."""
        # Look for enemy (clips) aligned supply depots
        for depot_pos, alignment in s.supply_depots:
            if alignment == "clips":
                return depot_pos

        # In cogsguard, chargers start as clips-aligned
        # If we haven't found enemy depots but know charger location, try that
        charger_pos = s.stations.get("charger")
        return charger_pos

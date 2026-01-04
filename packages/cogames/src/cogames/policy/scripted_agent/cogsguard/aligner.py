"""
Aligner role for CoGsGuard.

Aligners find neutral supply depots and align them to the cogs commons.
With aligner gear, they get +20 influence capacity.
"""

from __future__ import annotations

from typing import Optional

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role


class AlignerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Aligner agent: align supply depots to cogs."""

    ROLE = Role.ALIGNER

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute aligner behavior: find and align supply depots."""
        # Aligning requires: aligner gear + 1 influence + 1 heart
        # Check if we need resources first
        need_influence = s.influence < 1
        need_heart = s.heart < 1

        if DEBUG:
            print(f"[A{s.agent_id}] ALIGNER_EXEC: step={s.step_count}, influence={s.influence}, heart={s.heart}")

        if need_influence or need_heart:
            if DEBUG and s.step_count <= 100:
                print(f"[A{s.agent_id}] ALIGNER: Need resources - influence={need_influence}, heart={need_heart}")
            # Go to nexus to get influence (from AOE) and hearts (from commons)
            return self._get_resources(s, need_influence, need_heart)

        # Find a depot to align
        target_depot = self._find_alignable_depot(s)

        if target_depot is None:
            if DEBUG and s.step_count <= 100:
                print(f"[A{s.agent_id}] ALIGNER: No depot to align, exploring")
            return self._explore(s)

        # Navigate to depot
        if not is_adjacent((s.row, s.col), target_depot):
            if DEBUG and s.step_count <= 100:
                print(f"[A{s.agent_id}] ALIGNER: Moving to depot at {target_depot}")
            return self._move_towards(s, target_depot, reach_adjacent=True)

        # Align the depot by bumping it
        if DEBUG and s.step_count <= 100:
            print(f"[A{s.agent_id}] ALIGNER: ALIGNING depot at {target_depot}!")
        return self._use_object_at(s, target_depot)

    def _get_resources(self, s: CogsguardAgentState, need_influence: bool, need_heart: bool) -> Action:
        """Get influence and hearts from the nexus."""
        # Influence comes from AOE of aligned structures (nexus)
        # Hearts come from bumping the nexus (get_heart handler)
        assembler_pos = s.stations.get("assembler")
        if assembler_pos is None:
            return self._explore(s)

        # Navigate to assembler
        if not is_adjacent((s.row, s.col), assembler_pos):
            return self._move_towards(s, assembler_pos, reach_adjacent=True)

        # If we need hearts, bump the nexus to withdraw from commons
        if need_heart:
            return self._use_object_at(s, assembler_pos)

        # Just need influence - wait for AOE regeneration
        return self._actions.noop.Noop()

    def _find_alignable_depot(self, s: CogsguardAgentState) -> Optional[tuple[int, int]]:
        """Find a supply depot that can be aligned."""
        # Look for neutral (unaligned) supply depots
        # In cogsguard, chargers can be aligned
        for depot_pos, alignment in s.supply_depots:
            if alignment is None or alignment == "neutral":
                return depot_pos

        # If no known neutral depots, try the charger
        charger_pos = s.stations.get("charger")
        return charger_pos

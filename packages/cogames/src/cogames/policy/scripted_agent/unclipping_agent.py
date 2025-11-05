"""
UnclippingAgent - Extends BaselineAgent with unclipping capabilities.

This agent can detect clipped extractors and craft unclip items to restore them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from .baseline_agent import (
    BaselineAgent,
    CellType,
    ExtractorInfo,
    Phase,
    SimpleAgentState,
)

if TYPE_CHECKING:
    from cogames.policy import AgentPolicy
    from mettagrid.simulator import Simulation


@dataclass
class UnclippingAgentState(SimpleAgentState):
    """Extended state for unclipping agent."""

    # Unclip items inventory
    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

    # Unclip tracking
    blocked_by_clipped_extractor: Optional[tuple[int, int]] = None
    unclip_target_resource: Optional[str] = None  # Which resource is clipped


class UnclippingAgent(BaselineAgent):
    """
    Agent that can unclip extractors by crafting and using unclip items.

    Unclip item mapping:
    - decoder (from carbon) unclips oxygen extractors
    - modulator (from oxygen) unclips carbon extractors
    - resonator (from silicon) unclips germanium extractors
    - scrambler (from germanium) unclips silicon extractors
    """

    def __init__(self, simulation: "Simulation"):
        super().__init__(simulation)
        self._unclip_recipes = self._load_unclip_recipes()

    def _load_unclip_recipes(self) -> dict[str, str]:
        """
        Load unclip recipes: clipped_resource -> craft_resource.

        Returns mapping like {"oxygen": "carbon"} meaning to unclip oxygen,
        craft decoder from carbon.
        """
        # Standard mapping
        item_to_clipped_resource = {
            "decoder": "oxygen",
            "modulator": "carbon",
            "resonator": "germanium",
            "scrambler": "silicon",
        }

        item_to_craft_resource = {
            "decoder": "carbon",
            "modulator": "oxygen",
            "resonator": "silicon",
            "scrambler": "germanium",
        }

        # Build clipped_resource -> craft_resource mapping
        recipes = {}
        for item, clipped_res in item_to_clipped_resource.items():
            craft_res = item_to_craft_resource[item]
            recipes[clipped_res] = craft_res

        return recipes

    def _get_unclip_item_name(self, clipped_resource: str) -> Optional[str]:
        """Get the unclip item name for a clipped resource."""
        resource_to_item = {
            "oxygen": "decoder",
            "carbon": "modulator",
            "germanium": "resonator",
            "silicon": "scrambler",
        }
        return resource_to_item.get(clipped_resource)

    def _has_unclip_item(self, s: UnclippingAgentState) -> bool:
        """Check if agent has the unclip item for the blocked extractor."""
        if s.blocked_by_clipped_extractor is None:
            return False

        if s.unclip_target_resource is None:
            return False

        item_name = self._get_unclip_item_name(s.unclip_target_resource)
        if item_name is None:
            return False

        has_item = getattr(s, item_name, 0) > 0
        return has_item

    def _update_phase(self, s: UnclippingAgentState) -> None:
        """Override to add unclipping phase priorities."""
        # Priority 1: Recharge if energy low
        if s.energy < 30:
            if s.phase != Phase.RECHARGE:
                s.phase = Phase.RECHARGE
            return

        # Stay in RECHARGE until energy is fully restored (>= 90)
        if s.phase == Phase.RECHARGE:
            if s.energy >= 90:
                s.phase = Phase.GATHER
                s.target_position = None
            return

        # Priority 2: Deliver hearts if we have any
        if s.hearts > 0:
            if s.phase != Phase.DELIVER:
                s.phase = Phase.DELIVER
            return

        # Priority 3: Assemble if we have all resources
        can_assemble = (
            s.carbon >= self._heart_recipe["carbon"]
            and s.oxygen >= self._heart_recipe["oxygen"]
            and s.germanium >= self._heart_recipe["germanium"]
            and s.silicon >= self._heart_recipe["silicon"]
        )

        if can_assemble:
            if s.phase != Phase.ASSEMBLE:
                s.phase = Phase.ASSEMBLE
            return

        # Priority 4: Unclip if blocked and have unclip item
        if s.blocked_by_clipped_extractor is not None and self._has_unclip_item(s):
            if s.phase != Phase.UNCLIP:
                s.phase = Phase.UNCLIP
            return

        # Priority 5: Craft unclip item if blocked but don't have item
        # Only transition to CRAFT_UNCLIP if we have EXTRA craft resource (beyond what's needed for heart)
        # This prevents us from using our only carbon to craft decoder, then being stuck without carbon for hearts
        if s.blocked_by_clipped_extractor is not None and not self._has_unclip_item(s):
            craft_resource = self._unclip_recipes.get(s.unclip_target_resource) if s.unclip_target_resource else None

            if craft_resource:
                current_amount = getattr(s, craft_resource, 0)
                needed_for_heart = self._heart_recipe.get(craft_resource, 0)
                needed_for_craft = 1  # Need 1 unit to craft the unclip item
                total_needed = needed_for_heart + needed_for_craft

                has_enough = current_amount >= total_needed

                if has_enough:
                    if s.phase != Phase.CRAFT_UNCLIP:
                        s.phase = Phase.CRAFT_UNCLIP
                    return
                else:
                    # Stay in GATHER to collect more craft resource
                    if s.phase != Phase.GATHER:
                        s.phase = Phase.GATHER
                    return

        # Priority 6: Default to GATHER
        if s.phase != Phase.GATHER:
            s.phase = Phase.GATHER
            s.target_position = None

    def _find_any_needed_extractor(self, s: UnclippingAgentState) -> Optional[tuple[ExtractorInfo, str]]:
        """
        Override to detect clipped extractors.

        Returns (extractor, resource_type) or None if no extractors available.
        Sets s.blocked_by_clipped_extractor if all extractors are clipped.
        """
        deficits = self._calculate_deficits(s)

        # If blocked and need craft resource, treat it as a deficit
        if s.blocked_by_clipped_extractor is not None and s.unclip_target_resource:
            craft_resource = self._unclip_recipes.get(s.unclip_target_resource)
            if craft_resource:
                current_amount = getattr(s, craft_resource, 0)
                needed_for_heart = self._heart_recipe.get(craft_resource, 0)
                total_needed = needed_for_heart + 1  # +1 for crafting decoder
                if current_amount < total_needed:
                    deficits[craft_resource] = total_needed - current_amount

        # Check for needed resources
        for resource_type in ["carbon", "oxygen", "germanium", "silicon"]:
            deficit = deficits.get(resource_type, 0)
            if deficit <= 0:
                continue

            extractors = self._extractors.get(resource_type, [])
            if not extractors:
                continue

            # Filter available (not clipped, not depleted, not unreachable)
            if s.unreachable_extractors is None:
                s.unreachable_extractors = {}

            available = [
                e
                for e in extractors
                if not e.clipped and e.remaining_uses > 0 and s.unreachable_extractors.get(e.position, 0) < 5
            ]

            if available:
                # Found available extractor
                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                nearest = min(available, key=lambda e: distance(e.position))
                return (nearest, resource_type)

            # No available extractors - check if any are clipped (not just depleted!)
            clipped = [e for e in extractors if e.clipped and e.remaining_uses > 0]
            if clipped:
                # Found clipped extractor that we need
                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                nearest_clipped = min(clipped, key=lambda e: distance(e.position))
                s.blocked_by_clipped_extractor = nearest_clipped.position
                s.unclip_target_resource = resource_type
                return None

            # Check if we just have depleted extractors (can't unclip those)
            depleted = [e for e in extractors if e.remaining_uses == 0]
            if depleted:
                return None

        return None

    def _execute_phase(self, s: UnclippingAgentState) -> int:
        """Override to handle CRAFT_UNCLIP and UNCLIP phases."""
        if s.phase == Phase.GATHER:
            return self._do_gather(s)
        elif s.phase == Phase.ASSEMBLE:
            return self._do_assemble(s)
        elif s.phase == Phase.DELIVER:
            return self._do_deliver(s)
        elif s.phase == Phase.RECHARGE:
            return self._do_recharge(s)
        elif s.phase == Phase.CRAFT_UNCLIP:
            return self._do_craft_unclip(s)
        elif s.phase == Phase.UNCLIP:
            return self._do_unclip(s)
        return self._NOOP

    def _do_craft_unclip(self, s: UnclippingAgentState) -> int:
        """Craft unclip item at assembler."""
        if s.unclip_target_resource is None:
            s.phase = Phase.GATHER
            return self._NOOP

        # Get craft resource needed
        craft_resource = self._unclip_recipes.get(s.unclip_target_resource)
        if craft_resource is None:
            s.phase = Phase.GATHER
            return self._NOOP

        # Check if we have enough craft resource (need 1)
        current_amount = getattr(s, craft_resource, 0)
        if current_amount < 1:
            # Need to gather craft resource first
            s.phase = Phase.GATHER
            return self._NOOP

        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: self._stations["assembler"] is not None, reason="Need assembler for crafting"
        )
        if explore_action is not None:
            return explore_action

        # Change glyph to "gear" for crafting unclip items
        if s.current_glyph != "gear":
            vibe_action = self._change_vibe_actions["gear"]
            s.current_glyph = "gear"
            return vibe_action

        # Move to assembler and use it
        assembler = self._stations["assembler"]
        if assembler is None:
            return self._NOOP

        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            return self._move_into_cell(s, assembler)

        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_unclip(self, s: UnclippingAgentState) -> int:
        """Use unclip item on clipped extractor."""
        if s.blocked_by_clipped_extractor is None:
            s.phase = Phase.GATHER
            s.unclip_target_resource = None
            return self._NOOP

        # Navigate to clipped extractor
        target = s.blocked_by_clipped_extractor
        tr, tc = target
        dr = abs(s.row - tr)
        dc = abs(s.col - tc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Adjacent to clipped extractor - move into it to unclip
            # Clear blocked state after unclipping
            s.blocked_by_clipped_extractor = None
            s.unclip_target_resource = None
            return self._move_into_cell(s, target)

        # Not adjacent yet, move towards it
        return self._move_towards(s, target, reach_adjacent=True)


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class UnclippingPolicyImpl:
    """Implementation that wraps UnclippingAgent."""

    def __init__(self, simulation: "Simulation"):
        self._agent = UnclippingAgent(simulation)
        self._sim = simulation

    def agent_state(self, agent_id: int = 0) -> UnclippingAgentState:
        """Get initial state for an agent."""
        # Make sure agent states are initialized
        if agent_id not in self._agent._agent_states:
            state = UnclippingAgentState(
                agent_id=agent_id,
                map_height=self._agent._map_h,
                map_width=self._agent._map_w,
                occupancy=[[CellType.FREE.value] * self._agent._map_w for _ in range(self._agent._map_h)],
            )
            # Initialize mutable defaults
            state.unreachable_extractors = {}
            state.agent_occupancy = set()
            self._agent._agent_states[agent_id] = state
        return self._agent._agent_states[agent_id]

    def step_with_state(self, obs, state: UnclippingAgentState):
        """Compute action and return updated state."""
        from mettagrid.simulator.interface import Action

        # The state passed in tells us which agent this is
        agent_id = state.agent_id
        # Update the shared agent state
        self._agent._agent_states[agent_id] = state
        # Compute action (returns integer index)
        action_idx = self._agent.step(agent_id, obs)
        # Convert to Action object
        action = Action(name=self._agent._action_names[action_idx])
        # Return action and updated state
        return action, self._agent._agent_states[agent_id]


class UnclippingPolicy:
    """Policy class for unclipping agent.

    This policy requires a Simulation object for accessing grid_objects()
    to get absolute agent positions. Pass it via reset(simulation=sim).
    """

    def __init__(self):
        """Initialize policy (simulation will be provided via reset)."""
        self._sim = None
        self._impl = None
        self._agent_policies: Dict[int, "AgentPolicy"] = {}

    def reset(self, simulation: "Simulation" = None) -> None:
        """Reset all agent states.

        Args:
            simulation: The Simulation object (needed for grid_objects access)
        """
        if simulation is None:
            raise RuntimeError("UnclippingPolicy requires simulation parameter in reset()")

        self._sim = simulation
        self._impl = UnclippingPolicyImpl(simulation)
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int):
        """Get an AgentPolicy instance for a specific agent."""
        if self._impl is None:
            raise RuntimeError("Policy not initialized - call reset(simulation=sim) first")

        # Create agent policies lazily
        if agent_id not in self._agent_policies:
            from cogames.policy import StatefulAgentPolicy

            self._agent_policies[agent_id] = StatefulAgentPolicy(self._impl, agent_id)
        return self._agent_policies[agent_id]

"""
UnclippingAgent - Extends SimpleBaselineAgent with unclipping capabilities.

This agent can detect clipped extractors and craft unclip items to restore them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from mettagrid.config.vibes import VIBE_BY_NAME
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action

from .baseline_agent import (
    BaselineAgentPolicyImpl,
    BaselineHyperparameters,
    ExtractorInfo,
    Phase,
    SharedAgentState,
    SimpleAgentState,
)

if TYPE_CHECKING:
    from mettagrid.simulator import Simulation


@dataclass
class UnclippingHyperparameters(BaselineHyperparameters):
    """Extends baseline hyperparameters with unclipping-specific parameters."""

    # Unclipping strategy
    unclip_priority_order: tuple[str, ...] = ("oxygen", "silicon", "carbon", "germanium")  # Order to unclip resources
    craft_unclip_items_early: bool = True  # Craft unclip items proactively vs on-demand


@dataclass
class UnclippingAgentState(SimpleAgentState):
    """Extended state for unclipping agent.

    Note: decoder, modulator, resonator, scrambler are already defined in SimpleAgentState.
    """

    # Unclip tracking
    blocked_by_clipped_extractor: Optional[tuple[int, int]] = None
    unclip_target_resource: Optional[str] = None  # Which resource is clipped


class UnclippingAgentPolicyImpl(BaselineAgentPolicyImpl):
    """
    Agent that can unclip extractors by crafting and using unclip items.

    Unclip item mapping:
    - decoder (from carbon) unclips oxygen extractors
    - modulator (from oxygen) unclips carbon extractors
    - resonator (from silicon) unclips germanium extractors
    - scrambler (from germanium) unclips silicon extractors
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        shared_state: SharedAgentState,
        agent_id: int,
        hyperparams: UnclippingHyperparameters,
    ):
        super().__init__(policy_env_info, shared_state, agent_id, hyperparams)
        self._unclip_recipes = self._load_unclip_recipes()

    def initial_agent_state(self, simulation: Optional["Simulation"]) -> UnclippingAgentState:
        """Create initial state for unclipping agent."""
        assert simulation is not None
        return UnclippingAgentState(
            agent_id=self._agent_id,
            simulation=simulation,
            shared_state=self._shared_state,
            map_height=simulation.map_height,
            map_width=simulation.map_width,
            occupancy=[[1] * simulation.map_width for _ in range(simulation.map_height)],
            agent_occupancy=set(),
        )

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
        # When blocked by clipped extractor, prioritize unclipping over heart assembly
        # Rationale: Can't make hearts anyway without access to the clipped resource
        if s.blocked_by_clipped_extractor is not None and not self._has_unclip_item(s):
            craft_resource = self._unclip_recipes.get(s.unclip_target_resource) if s.unclip_target_resource else None

            if craft_resource:
                current_amount = getattr(s, craft_resource, 0)

                # If we have at least 1 unit of craft resource, craft the unclip item
                if current_amount >= 1:
                    if s.phase != Phase.CRAFT_UNCLIP:
                        s.phase = Phase.CRAFT_UNCLIP
                    return
                else:
                    # Stay in GATHER to collect craft resource
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

            # If we're blocked by this resource being clipped, skip it and gather craft resource instead
            if s.blocked_by_clipped_extractor is not None and s.unclip_target_resource == resource_type:
                continue

            extractors = s.shared_state.extractors.get(resource_type, [])
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

    def _execute_phase(self, s: UnclippingAgentState) -> Action:
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
        return self._actions.noop.Noop()

    def _do_craft_unclip(self, s: UnclippingAgentState) -> Action:
        """Craft unclip item at assembler."""
        if s.unclip_target_resource is None:
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        # Get craft resource needed
        craft_resource = self._unclip_recipes.get(s.unclip_target_resource)
        if craft_resource is None:
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        # Check if we have enough craft resource (need 1)
        current_amount = getattr(s, craft_resource, 0)
        if current_amount < 1:
            # Need to gather craft resource first
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: s.shared_state.stations["assembler"] is not None, reason="Need assembler for crafting"
        )
        if explore_action is not None:
            return explore_action

        # Change glyph to "gear" for crafting unclip items
        if s.current_glyph != "gear":
            vibe_action = self._actions.change_vibe.ChangeVibe(VIBE_BY_NAME["gear"])
            s.current_glyph = "gear"
            return vibe_action

        # Move to assembler and use it
        assembler = s.shared_state.stations["assembler"]
        if assembler is None:
            return self._actions.noop.Noop()

        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            return self._move_into_cell(s, assembler)

        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_unclip(self, s: UnclippingAgentState) -> Action:
        """Use unclip item on clipped extractor."""
        if s.blocked_by_clipped_extractor is None:
            s.phase = Phase.GATHER
            s.unclip_target_resource = None
            return self._actions.noop.Noop()

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
# Policy Wrapper Class
# ============================================================================


class UnclippingPolicy(MultiAgentPolicy):
    """Multi-agent policy wrapper for UnclippingAgent.

    This class wraps UnclippingAgent to work with the policy interface.
    It handles multiple agents, each with their own UnclippingAgent instance.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, hyperparams: Optional[UnclippingHyperparameters] = None):
        super().__init__(policy_env_info)
        self._shared_state = SharedAgentState()
        self._agent_policies: dict[int, StatefulAgentPolicy[UnclippingAgentState]] = {}
        self._hyperparams = hyperparams or UnclippingHyperparameters()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[UnclippingAgentState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                UnclippingAgentPolicyImpl(self._policy_env_info, self._shared_state, agent_id, self._hyperparams),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]

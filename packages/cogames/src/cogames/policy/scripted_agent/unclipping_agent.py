"""
UnclippingAgent - Extends SimpleBaselineAgent with unclipping capabilities.

This agent can detect clipped extractors and craft unclip items to restore them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, cast

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

from .baseline_agent import BaselineAgentPolicyImpl
from .types import (
    BaselineHyperparameters,
    ExtractorInfo,
    Phase,
    SimpleAgentState,
)
from .utils import use_object_at

if TYPE_CHECKING:
    pass


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

    # Discovered unclip recipes from observations (resource_type -> (unclip_item, craft_recipe))
    # e.g., "oxygen" -> ("decoder", {"carbon": 1})
    unclip_recipes: dict[str, tuple[str, dict[str, int]]] = field(default_factory=dict)

    # Craft recipes for unclip items (item_name -> craft_recipe)
    # e.g., "decoder" -> {"carbon": 1}
    # Initialized from assembler protocols
    unclip_craft_recipes: dict[str, dict[str, int]] = field(default_factory=dict)


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
        agent_id: int,
        hyperparams: UnclippingHyperparameters,
    ):
        super().__init__(policy_env_info, agent_id, hyperparams)
        self._priority_order = hyperparams.unclip_priority_order

    def initial_agent_state(self) -> UnclippingAgentState:
        """Create initial state for unclipping agent."""
        # Get the base state from parent class
        base_state = super().initial_agent_state()

        # Initialize unclip recipes from assembler protocols
        # Map: unclip_item -> craft_recipe (e.g., "decoder" -> {"carbon": 1})
        unclip_craft_recipes = {}
        for protocol in self._policy_env_info.assembler_protocols:
            for output_item, output_amount in protocol.output_resources.items():
                if output_amount > 0 and output_item in ("decoder", "modulator", "resonator", "scrambler"):
                    craft_recipe = dict(protocol.input_resources)
                    craft_recipe.pop("energy", None)
                    unclip_craft_recipes[output_item] = craft_recipe

        # Create unclipping state by extending base state
        # Convert base_state dict to UnclippingAgentState with additional fields
        return UnclippingAgentState(
            **base_state.__dict__,
            unclip_recipes={},  # Will be discovered from clipped extractor observations
            unclip_craft_recipes=unclip_craft_recipes,  # Store craft recipes separately
        )

    def _discover_unclip_recipe(self, s: UnclippingAgentState, parsed_observation) -> None:
        """
        Discover unclip recipes from clipped extractor observations.

        When we observe a clipped extractor, its protocol_outputs tell us what unclip item is needed.
        We then need to observe the assembler with the correct vibe to learn how to craft that item.
        """
        for _pos, obj_state in parsed_observation.nearby_objects.items():
            obj_name = obj_state.name.lower()
            # Check if this is a clipped extractor
            if "extractor" in obj_name and obj_state.clipped > 0:
                # Extract resource type from name (e.g., "oxygen_extractor" -> "oxygen")
                resource_type = obj_name.replace("_extractor", "")
                # Check if we already know this recipe
                if resource_type in s.unclip_recipes:
                    continue

                # Read the unclip item from protocol_inputs (unclipping protocols have inputs, not outputs!)
                if obj_state.protocol_inputs:
                    # The clipped extractor's protocol_inputs shows what unclip item is needed
                    for item_name, amount in obj_state.protocol_inputs.items():
                        if amount > 0 and item_name in ("decoder", "modulator", "resonator", "scrambler"):
                            # Get the craft recipe from our initialized recipes
                            craft_recipe = s.unclip_craft_recipes.get(item_name, {})
                            s.unclip_recipes[resource_type] = (item_name, craft_recipe)
                            break

            # No need to discover craft recipes from assembler - we already have them from init!

    def _get_unclip_item_name(self, s: UnclippingAgentState, clipped_resource: str) -> Optional[str]:
        recipe = s.unclip_recipes.get(clipped_resource)
        if recipe is None:
            return None
        item_name, _ = recipe
        return item_name

    def _has_unclip_item(self, s: UnclippingAgentState) -> bool:
        """Check if agent has the unclip item for the blocked extractor."""
        if s.blocked_by_clipped_extractor is None:
            return False

        if s.unclip_target_resource is None:
            return False

        item_name = self._get_unclip_item_name(s, s.unclip_target_resource)
        if item_name is None:
            return False

        has_item = getattr(s, item_name, 0) > 0
        return has_item

    def _get_unclip_info(
        self, s: UnclippingAgentState, resource: Optional[str]
    ) -> Optional[tuple[str, dict[str, int]]]:
        """Get unclip item name and craft recipe for a resource type."""
        if resource is None:
            return None
        return s.unclip_recipes.get(resource)

    def _clear_unclip_state(self, s: UnclippingAgentState) -> None:
        s.blocked_by_clipped_extractor = None
        s.unclip_target_resource = None

    def _set_unclip_state(self, s: UnclippingAgentState, resource_type: str, extractor: ExtractorInfo) -> None:
        # Only set unclip state if we've discovered the recipe for this resource
        if resource_type not in s.unclip_recipes:
            return
        s.blocked_by_clipped_extractor = extractor.position
        s.unclip_target_resource = resource_type

    def _get_vibe_for_phase(self, phase: Phase, state: UnclippingAgentState) -> str:
        """Override to set correct vibe for CRAFT_UNCLIP and UNCLIP phases."""
        # For crafting unclip items at the assembler, use "gear" vibe
        if phase == Phase.CRAFT_UNCLIP:
            return "gear"

        # For unclipping extractors, use "gear" vibe
        if phase == Phase.UNCLIP:
            return "gear"

        # Otherwise use baseline logic
        return super()._get_vibe_for_phase(phase, state)

    def _update_phase(self, s: UnclippingAgentState) -> None:
        """Override to add unclipping phase priorities before gathering."""
        old_phase = s.phase

        # Priority 1-2: Recharge and Deliver (handled by parent)
        if s.energy < self._hyperparams.recharge_threshold_low or s.phase == Phase.RECHARGE or s.hearts > 0:
            super()._update_phase(s)
            if old_phase != s.phase:
                s.cached_path = None
                s.cached_path_target = None
            return

        # Priority 3: Assemble if possible (and not currently blocked by clipped resource)
        heart_recipe = s.heart_recipe or {}
        can_assemble = all(
            getattr(s, res, 0) >= heart_recipe.get(res, 0) for res in ("carbon", "oxygen", "germanium", "silicon")
        )

        if can_assemble and s.blocked_by_clipped_extractor is None:
            if s.phase != Phase.ASSEMBLE:
                s.phase = Phase.ASSEMBLE
                s.pending_use_resource = None
                s.pending_use_amount = 0
                s.waiting_at_extractor = None
            if old_phase != s.phase:
                s.cached_path = None
                s.cached_path_target = None
            return

        # Priority 4: Unclipping workflow (before gathering)
        if s.blocked_by_clipped_extractor is not None and s.unclip_target_resource is not None:
            # First, check if the extractor is still clipped
            target_pos = s.blocked_by_clipped_extractor
            extractors = s.extractors.get(s.unclip_target_resource, [])
            target_extractor = None
            for ext in extractors:
                if ext.position == target_pos:
                    target_extractor = ext
                    break

            # If extractor is no longer clipped or not found, clear unclip state
            if target_extractor is None or not target_extractor.clipped:
                self._clear_unclip_state(s)
                # Fall through to normal phase logic
            else:
                # Still clipped, continue unclipping workflow
                info = self._get_unclip_info(s, s.unclip_target_resource)

                if info is not None:
                    item_name, craft_recipe = info
                    item_count = getattr(s, item_name, 0)

                    if item_count > 0:
                        if s.phase != Phase.UNCLIP:
                            s.phase = Phase.UNCLIP
                        return
                    elif craft_recipe and all(getattr(s, res, 0) >= amt for res, amt in craft_recipe.items()):
                        if s.phase != Phase.CRAFT_UNCLIP:
                            s.phase = Phase.CRAFT_UNCLIP
                        return
                    else:
                        # Need to gather craft resources
                        if s.phase != Phase.GATHER:
                            s.phase = Phase.GATHER
                        if old_phase != s.phase:
                            s.cached_path = None
                            s.cached_path_target = None
                        return

        # Priority 5: Default to GATHER (handled by parent)
        super()._update_phase(s)
        if old_phase != s.phase:
            s.cached_path = None
            s.cached_path_target = None

    def _find_any_needed_extractor(self, s: UnclippingAgentState) -> Optional[tuple[ExtractorInfo, str]]:
        """
        Override to detect clipped extractors and trigger unclipping workflow.
        """
        # Try baseline logic first
        result = super()._find_any_needed_extractor(s)
        if result is not None:
            self._clear_unclip_state(s)
            return result

        deficits = self._calculate_deficits(s)

        def distance(pos: tuple[int, int]) -> int:
            return abs(pos[0] - s.row) + abs(pos[1] - s.col)

        # If we are already blocked, ensure we gather craft resources if needed
        if s.blocked_by_clipped_extractor is not None:
            info = self._get_unclip_info(s, s.unclip_target_resource)
            if info is not None:
                item_name, craft_recipe = info
                if getattr(s, item_name, 0) > 0:
                    return None  # Ready to unclip

                # Need craft resources before crafting
                if craft_recipe:
                    for craft_resource, needed_amount in craft_recipe.items():
                        if getattr(s, craft_resource, 0) < needed_amount:
                            craft_extractors = s.extractors.get(craft_resource, [])
                            available = [e for e in craft_extractors if not e.clipped and e.remaining_uses > 0]
                            if available:
                                nearest = min(available, key=lambda e: distance(e.position))
                                return (nearest, craft_resource)

                            clipped = [e for e in craft_extractors if e.clipped and e.remaining_uses > 0]
                            if clipped:
                                nearest = min(clipped, key=lambda e: distance(e.position))
                                self._set_unclip_state(s, craft_resource, nearest)
                            return None

        # Look for deficits that are blocked by clipped extractors
        for resource_type in self._priority_order:
            if deficits.get(resource_type, 0) <= 0:
                continue

            extractors = s.extractors.get(resource_type, [])
            if not extractors:
                continue

            available = [e for e in extractors if not e.clipped and e.remaining_uses > 0]
            if available:
                nearest = min(available, key=lambda e: distance(e.position))
                return (nearest, resource_type)

            clipped = [e for e in extractors if e.clipped and e.remaining_uses > 0]
            if clipped:
                nearest = min(clipped, key=lambda e: distance(e.position))
                self._set_unclip_state(s, resource_type, nearest)
                return None

        # No special action
        self._clear_unclip_state(s)
        return None

    def step_with_state(
        self, obs: AgentObservation, state: UnclippingAgentState
    ) -> tuple[Action, UnclippingAgentState]:
        """Override to discover unclip recipes from observations."""
        # First, let the base class handle observation parsing and state updates
        action, updated_state = super().step_with_state(obs, state)
        state = cast(UnclippingAgentState, updated_state)

        # Now discover unclip recipes from the parsed observation
        # We need to parse again to get the nearby_objects
        parsed = self.parse_observation(state, obs)
        self._discover_unclip_recipe(state, parsed)

        return action, state

    def _execute_phase(self, s: UnclippingAgentState) -> Action:
        """Override to handle CRAFT_UNCLIP and UNCLIP phases."""
        if s.phase == Phase.CRAFT_UNCLIP:
            return self._do_craft_unclip(s)
        elif s.phase == Phase.UNCLIP:
            return self._do_unclip(s)
        # All other phases handled by parent (GATHER, ASSEMBLE, DELIVER, RECHARGE)
        return super()._execute_phase(s)

    def _do_craft_unclip(self, s: UnclippingAgentState) -> Action:
        """Craft unclip item at assembler."""
        info = self._get_unclip_info(s, s.unclip_target_resource)
        if info is None:
            self._clear_unclip_state(s)
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        item_name, craft_recipe = info

        # Check if we have all craft resources
        if craft_recipe and not all(getattr(s, res, 0) >= amt for res, amt in craft_recipe.items()):
            # Need to gather craft resources first
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: s.stations.get("assembler") is not None, reason="Need assembler for crafting"
        )
        if explore_action is not None:
            return explore_action

        # Vibe is automatically set by _get_vibe_for_phase to the input resource (e.g., "carbon" for decoder)

        assembler = s.stations.get("assembler")
        if assembler is None:
            return self._actions.noop.Noop()

        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            return use_object_at(
                s, assembler, actions=self._actions, move_deltas=self._move_deltas, using_for=f"craft_{item_name}"
            )

        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_unclip(self, s: UnclippingAgentState) -> Action:
        """Use unclip item on clipped extractor."""
        if s.blocked_by_clipped_extractor is None:
            s.phase = Phase.GATHER
            self._clear_unclip_state(s)
            return self._actions.noop.Noop()

        info = self._get_unclip_info(s, s.unclip_target_resource)
        if info is None:
            self._clear_unclip_state(s)
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        item_name, _ = info
        if getattr(s, item_name, 0) <= 0:
            # Lost the item before reaching extractor
            self._clear_unclip_state(s)
            s.phase = Phase.GATHER
            return self._actions.noop.Noop()

        # Navigate to clipped extractor
        target = s.blocked_by_clipped_extractor
        tr, tc = target
        dr = abs(s.row - tr)
        dc = abs(s.col - tc)
        is_at_target = dr == 0 and dc == 0
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_at_target:
            # Already on the extractor - it should be unclipped now
            # Wait for next step to verify and clear state
            return self._actions.noop.Noop()

        if is_adjacent:
            # Adjacent to clipped extractor - use it to unclip (like using any other object)
            action = use_object_at(
                s,
                target,
                actions=self._actions,
                move_deltas=self._move_deltas,
                using_for=f"unclip_{s.unclip_target_resource}",
            )
            # Don't clear unclip state yet - wait until next step to verify it worked
            # The state will be cleared in _update_phase when we see the extractor is unclipped
            return action

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

    short_names = ["ladybug"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        hyperparams: Optional[UnclippingHyperparameters] = None,
    ):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[UnclippingAgentState]] = {}
        self._hyperparams = hyperparams or UnclippingHyperparameters()

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[UnclippingAgentState]:
        if agent_id not in self._agent_policies:
            # UnclippingAgentPolicyImpl uses UnclippingAgentState but inherits from
            # BaselineAgentPolicyImpl typed with SimpleAgentState, requiring a cast
            policy = cast(
                StatefulAgentPolicy[UnclippingAgentState],
                StatefulAgentPolicy(
                    UnclippingAgentPolicyImpl(self._policy_env_info, agent_id, self._hyperparams),
                    self._policy_env_info,
                    agent_id=agent_id,
                ),
            )
            self._agent_policies[agent_id] = policy
        return self._agent_policies[agent_id]

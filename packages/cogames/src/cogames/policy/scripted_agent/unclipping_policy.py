"""Policy wrapper for UnclippingAgentImpl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cogames.policy.scripted_agent.simple_baseline_agent import (
    CellType,
    ExtractorInfo,
    Phase,
    SimpleAgentState,
    SimpleBaselineAgentImpl,
)
from mettagrid.config.vibes import VIBE_BY_NAME
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation


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


class UnclippingAgentImpl(SimpleBaselineAgentImpl):
    """Internal implementation for unclipping agent."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        simulation: Optional[Simulation],
    ):
        """Initialize unclipping agent implementation."""
        super().__init__(policy_env_info, agent_id, simulation)
        # Load unclip recipes
        self._unclip_recipes = self._load_unclip_recipes()
        print(f"[UnclippingAgent] Initialized with recipes: {self._unclip_recipes}")

    def _load_unclip_recipes(self) -> dict[str, str]:
        """Load unclip recipes: clipped_resource -> craft_resource."""
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
        return getattr(s, item_name, 0) > 0

    def agent_state(self, simulation: Simulation | None = None) -> UnclippingAgentState:
        """Get initial state for unclipping agent."""
        if simulation is None:
            return UnclippingAgentState(
                agent_id=self._agent_id,
                id_map=None,
                map_height=0,
                map_width=0,
                occupancy=None,
                simulation=None,
            )

        map_height = simulation.map_height
        map_width = simulation.map_width
        occupancy = [[CellType.FREE.value] * map_width for _ in range(map_height)]
        print(f"[UnclippingAgent] Initialized for map {map_height}x{map_width}")
        return UnclippingAgentState(
            agent_id=self._agent_id,
            id_map=simulation.id_map,
            simulation=simulation,
            map_height=map_height,
            map_width=map_width,
            occupancy=occupancy,
        )

    def step_with_state(
        self, obs: AgentObservation, state: UnclippingAgentState
    ) -> tuple[Action, UnclippingAgentState]:
        """Get action and update state."""
        action = self._step_impl(state, obs)
        return action, state

    def _update_phase(self, s: UnclippingAgentState) -> None:
        """Override to add unclipping phase priorities."""
        # Priority 1: Recharge if energy low
        if s.energy < 30:
            if s.phase != Phase.RECHARGE:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> RECHARGE (energy={s.energy})")
                s.phase = Phase.RECHARGE
            return

        # Stay in RECHARGE until energy is fully restored (>= 90)
        if s.phase == Phase.RECHARGE:
            if s.energy >= 90:
                print(f"[Agent {s.agent_id}] Phase: RECHARGE -> GATHER (energy={s.energy})")
                s.phase = Phase.GATHER
                s.target_position = None
            return

        # Priority 2: Deliver hearts if we have any
        if s.hearts > 0:
            if s.phase != Phase.DELIVER:
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> DELIVER ({s.hearts} hearts)")
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
                print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> ASSEMBLE (all resources ready)")
                s.phase = Phase.ASSEMBLE
            return

        # Priority 4: Unclip if blocked and have unclip item
        if s.blocked_by_clipped_extractor is not None and self._has_unclip_item(s):
            if s.phase != Phase.UNCLIP:
                print(
                    f"[Agent {s.agent_id}] Phase: {s.phase.name} -> UNCLIP "
                    f"(have unclip item for {s.unclip_target_resource})"
                )
                s.phase = Phase.UNCLIP
            return

        # Priority 5: Craft unclip item if blocked but don't have item
        if s.blocked_by_clipped_extractor is not None and not self._has_unclip_item(s):
            if s.phase != Phase.CRAFT_UNCLIP:
                print(
                    f"[Agent {s.agent_id}] Phase: {s.phase.name} -> CRAFT_UNCLIP "
                    f"(need to craft for {s.unclip_target_resource})"
                )
                s.phase = Phase.CRAFT_UNCLIP
            return

        # Priority 6: Default to GATHER
        if s.phase != Phase.GATHER:
            print(f"[Agent {s.agent_id}] Phase: {s.phase.name} -> GATHER (need resources)")
            s.phase = Phase.GATHER
            s.target_position = None

    def _find_any_needed_extractor(self, s: UnclippingAgentState) -> Optional[tuple[ExtractorInfo, str]]:
        """Override to detect clipped extractors."""
        deficits = self._calculate_deficits(s)

        for resource_type in ["carbon", "oxygen", "germanium", "silicon"]:
            if deficits.get(resource_type, 0) <= 0:
                continue

            extractors = s.extractors.get(resource_type, [])
            if not extractors:
                continue

            # Filter available (not clipped, not depleted, not unreachable)
            if s.unreachable_extractors is None:
                s.unreachable_extractors = {}

            available = [
                e
                for e in extractors
                if not e.clipped
                and e.remaining_uses > 0
                and s.unreachable_extractors.get(e.position, 0) < 5
            ]

            if available:

                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                nearest = min(available, key=lambda e: distance(e.position))
                return (nearest, resource_type)

            # No available extractors - check if any are clipped or depleted
            clipped = [e for e in extractors if e.clipped or e.remaining_uses == 0]
            if clipped:

                def distance(pos: tuple[int, int]) -> int:
                    return abs(pos[0] - s.row) + abs(pos[1] - s.col)

                nearest_clipped = min(clipped, key=lambda e: distance(e.position))
                s.blocked_by_clipped_extractor = nearest_clipped.position
                s.unclip_target_resource = resource_type
                print(
                    f"[Agent {s.agent_id}] All {resource_type} extractors clipped! "
                    f"Need to unclip at {nearest_clipped.position}"
                )
                return None

        return None

    def _execute_phase(self, state: UnclippingAgentState) -> Action:
        """Override to handle CRAFT_UNCLIP and UNCLIP phases."""
        if state.phase == Phase.GATHER:
            return self._do_gather(state)
        elif state.phase == Phase.ASSEMBLE:
            return self._do_assemble(state)
        elif state.phase == Phase.DELIVER:
            return self._do_deliver(state)
        elif state.phase == Phase.RECHARGE:
            return self._do_recharge(state)
        elif state.phase == Phase.CRAFT_UNCLIP:
            return self._do_craft_unclip(state)
        elif state.phase == Phase.UNCLIP:
            return self._do_unclip(state)
        return Action(name=self._policy_env_info.actions.noop.Noop().name)

    def _do_craft_unclip(self, s: UnclippingAgentState) -> Action:
        """Craft unclip item at assembler."""
        if s.unclip_target_resource is None:
            print(f"[Agent {s.agent_id}] CRAFT_UNCLIP: No target resource, returning to GATHER")
            s.phase = Phase.GATHER
            return Action(name=self._policy_env_info.actions.noop.Noop().name)

        craft_resource = self._unclip_recipes.get(s.unclip_target_resource)
        if craft_resource is None:
            print(f"[Agent {s.agent_id}] CRAFT_UNCLIP: No recipe for {s.unclip_target_resource}")
            s.phase = Phase.GATHER
            return Action(name=self._policy_env_info.actions.noop.Noop().name)

        current_amount = getattr(s, craft_resource, 0)
        if current_amount < 1:
            print(f"[Agent {s.agent_id}] CRAFT_UNCLIP: Need {craft_resource} to craft, have {current_amount}")
            return self._do_gather(s)

        explore_action = self._explore_until(
            s, condition=lambda: s.stations["assembler"] is not None, reason="Need assembler for crafting"
        )
        if explore_action is not None:
            return explore_action

        if s.current_glyph != "gear":
            # Get vibe action from policy_env_info
            gear_vibe = VIBE_BY_NAME["gear"]
            vibe_action_name = self._policy_env_info.actions.change_vibe.ChangeVibe(gear_vibe).name
            print(f"[Agent {s.agent_id}] Changing glyph to 'gear' for crafting (action {vibe_action_name})")
            s.current_glyph = "gear"
            return Action(name=vibe_action_name)

        assembler = s.stations["assembler"]
        if assembler is None:
            return Action(name=self._policy_env_info.actions.noop.Noop().name)
        ar, ac = assembler
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            print(f"[Agent {s.agent_id}] Crafting unclip item at assembler")
            return self._move_into_cell(s, assembler)

        return self._move_towards(s, assembler, reach_adjacent=True)

    def _do_unclip(self, s: UnclippingAgentState) -> Action:
        """Use unclip item on clipped extractor."""
        if s.blocked_by_clipped_extractor is None:
            print(f"[Agent {s.agent_id}] UNCLIP: No blocked extractor, returning to GATHER")
            s.phase = Phase.GATHER
            s.unclip_target_resource = None
            return Action(name=self._policy_env_info.actions.noop.Noop().name)

        target = s.blocked_by_clipped_extractor
        tr, tc = target
        dr = abs(s.row - tr)
        dc = abs(s.col - tc)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            print(f"[Agent {s.agent_id}] Unclipping extractor at {target}")
            s.blocked_by_clipped_extractor = None
            s.unclip_target_resource = None
            return self._move_into_cell(s, target)

        return self._move_towards(s, target, reach_adjacent=True)


class UnclippingPolicy(MultiAgentPolicy):
    """Policy class for unclipping agent."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._policy_env_info = policy_env_info
        self._agent_policies: list[AgentPolicy] = [
            StatefulAgentPolicy(UnclippingAgentImpl(self._policy_env_info, agent_id, simulation=None), agent_id)
            for agent_id in range(self._policy_env_info.num_agents)
        ]

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get policy for a specific agent."""
        return self._agent_policies[agent_id]

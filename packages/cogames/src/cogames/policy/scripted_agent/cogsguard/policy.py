"""
CoGsGuard Scripted Agent - Vibe-based multi-agent policy.

Agents use vibes to determine their behavior:
- default: do nothing (noop)
- gear: pick a role at random, change vibe to that role
- miner/scout/aligner/scrambler: get gear if needed, then execute role behavior
- heart: do nothing (noop)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from cogames.policy.scripted_agent.pathfinding import compute_goal_cells, shortest_path
from cogames.policy.scripted_agent.pathfinding import is_traversable as path_is_traversable
from cogames.policy.scripted_agent.pathfinding import is_within_bounds as path_is_within_bounds
from cogames.policy.scripted_agent.types import CellType, ObjectState, ParsedObservation
from cogames.policy.scripted_agent.utils import change_vibe_action, is_adjacent, is_station, is_wall
from cogames.policy.scripted_agent.utils import parse_observation as utils_parse_observation
from mettagrid.config.mettagrid_config import CardinalDirection
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action

from .types import (
    ROLE_TO_STATION,
    CogsguardAgentState,
    CogsguardPhase,
    Role,
    StructureInfo,
    StructureType,
)

# Vibe names for role selection
ROLE_VIBES = ["scout", "miner", "aligner", "scrambler"]
VIBE_TO_ROLE = {
    "miner": Role.MINER,
    "scout": Role.SCOUT,
    "aligner": Role.ALIGNER,
    "scrambler": Role.SCRAMBLER,
}

if TYPE_CHECKING:
    from mettagrid.simulator.interface import AgentObservation

# Debug flag - set to True to see detailed agent behavior
DEBUG = True


class CogsguardAgentPolicyImpl(StatefulPolicyImpl[CogsguardAgentState]):
    """Base policy implementation for CoGsGuard agents.

    Handles common behavior like gear acquisition. Role-specific behavior
    is implemented by overriding execute_role().
    """

    # Subclasses set this
    ROLE: Role = Role.MINER

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        role: Role,
    ):
        self._agent_id = agent_id
        self._role = role
        self._policy_env_info = policy_env_info

        # Observation grid half-ranges
        self._obs_hr = policy_env_info.obs_height // 2
        self._obs_wr = policy_env_info.obs_width // 2

        # Action lookup
        self._actions = policy_env_info.actions
        self._move_deltas = {
            "north": (-1, 0),
            "south": (1, 0),
            "east": (0, 1),
            "west": (0, -1),
        }

        # Feature name sets for observation parsing
        self._spatial_feature_names = {"tag", "cooldown_remaining", "clipped", "remaining_uses"}
        self._agent_feature_key_by_name = {"agent:group": "agent_group", "agent:frozen": "agent_frozen"}
        self._protocol_input_prefix = "protocol_input:"
        self._protocol_output_prefix = "protocol_output:"

        # Cache tag names on first use
        self._tag_names: dict[int, str] = {}

    def initial_agent_state(self) -> CogsguardAgentState:
        """Initialize state for this agent.

        IMPORTANT: Positions are tracked RELATIVE to the agent's starting position.
        - Agent starts at (0, 0) in internal coordinates
        - All discovered object positions are relative to this origin
        - The actual map size doesn't matter - we only use relative offsets
        - Occupancy grid is centered at (grid_size/2, grid_size/2) to allow negative relative positions
        """
        self._tag_names = self._policy_env_info.tag_id_to_name

        # Use a grid large enough for typical exploration range
        # Grid center is the agent's starting position (0, 0) in relative coords
        # But stored at grid_center to allow negative relative positions
        grid_size = 200
        grid_center = grid_size // 2

        return CogsguardAgentState(
            agent_id=self._agent_id,
            role=self._role,
            map_height=grid_size,
            map_width=grid_size,
            occupancy=[[CellType.FREE.value] * grid_size for _ in range(grid_size)],
            explored=[[False] * grid_size for _ in range(grid_size)],
            # Start at (0, 0) relative - stored at grid center for negative offset support
            row=grid_center,
            col=grid_center,
            stations={},
        )

    def step_with_state(self, obs: AgentObservation, s: CogsguardAgentState) -> tuple[Action, CogsguardAgentState]:
        """Main step function."""
        s.step_count += 1
        s.current_obs = obs
        s.agent_occupancy.clear()

        # Read inventory
        self._read_inventory(s, obs)

        # Update position from last action
        self._update_agent_position(s)

        # Parse observation
        parsed = self._parse_observation(s, obs)

        # Update map knowledge
        self._update_occupancy_and_discover(s, parsed)

        # Update phase
        self._update_phase(s)

        # Execute current phase
        action = self._execute_phase(s)

        # Debug logging
        if DEBUG and s.step_count <= 50:  # Only first 50 steps per agent
            gear_status = "HAS_GEAR" if s.has_gear() else "NO_GEAR"
            nexus_pos = s.stations.get("assembler", "NOT_FOUND")
            print(
                f"[A{s.agent_id}] Step {s.step_count}: vibe={s.current_vibe} role={s.role.value} | "
                f"Phase={s.phase.value} | {gear_status} | "
                f"Energy={s.energy} | "
                f"Pos=({s.row},{s.col}) | "
                f"Nexus@{nexus_pos} | "
                f"Action={action.name}"
            )

        s.last_action = action
        return action, s

    def _read_inventory(self, s: CogsguardAgentState, obs: AgentObservation) -> None:
        """Read inventory, vibe, and last executed action from observation."""
        inv = {}
        vibe_id = 0  # Default vibe ID
        last_action_id: Optional[int] = None
        center_r, center_c = self._obs_hr, self._obs_wr
        for tok in obs.tokens:
            if tok.location == (center_r, center_c):
                feature_name = tok.feature.name
                if feature_name.startswith("inv:"):
                    resource_name = feature_name[4:]
                    inv[resource_name] = tok.value
                elif feature_name == "vibe":
                    vibe_id = tok.value
                elif feature_name == "last_action":
                    last_action_id = tok.value

        s.energy = inv.get("energy", 0)
        s.carbon = inv.get("carbon", 0)
        s.oxygen = inv.get("oxygen", 0)
        s.germanium = inv.get("germanium", 0)
        s.silicon = inv.get("silicon", 0)
        s.heart = inv.get("heart", 0)
        s.influence = inv.get("influence", 0)
        s.hp = inv.get("hp", 100)

        # Gear items
        s.miner = inv.get("miner", 0)
        s.scout = inv.get("scout", 0)
        s.aligner = inv.get("aligner", 0)
        s.scrambler = inv.get("scrambler", 0)

        # Read vibe name from vibe ID using policy_env_info
        s.current_vibe = self._get_vibe_name(vibe_id)

        # Read last executed action from observation
        # This tells us what the simulator actually did, not what we intended
        if last_action_id is not None:
            action_names = self._policy_env_info.action_names
            if 0 <= last_action_id < len(action_names):
                s.last_action_executed = action_names[last_action_id]
            else:
                s.last_action_executed = None
        else:
            s.last_action_executed = None

    def _get_vibe_name(self, vibe_id: int) -> str:
        """Convert vibe ID to vibe name."""
        # Get vibe names from the change_vibe action config
        change_vibe_cfg = getattr(self._actions, "change_vibe", None)
        if change_vibe_cfg is not None:
            vibes = getattr(change_vibe_cfg, "vibes", [])
            if 0 <= vibe_id < len(vibes):
                return vibes[vibe_id].name
        return "default"

    def _update_agent_position(self, s: CogsguardAgentState) -> None:
        """Update position based on last action that was ACTUALLY EXECUTED.

        IMPORTANT: Position is ONLY updated when OUR intended action matches
        the executed action. This ensures:
        1. Position doesn't update when moves fail (executed=noop, intended=move_X)
        2. Position doesn't update when a human takes over and moves the cog
           (the human's moves are not what we intended)

        This keeps internal position consistent even during human control,
        so when control returns to the agent, its internal map remains valid.
        """
        # Use last_action_executed from observation, NOT last_action (our intent)
        executed_action = s.last_action_executed
        intended_action = s.last_action.name if s.last_action else None

        # Debug: Log when intended != executed (action failed, delayed, or human control)
        if DEBUG and s.step_count <= 100:
            if intended_action and executed_action and intended_action != executed_action:
                print(
                    f"[A{s.agent_id}] ACTION_MISMATCH: intended={intended_action}, "
                    f"executed={executed_action} (action failed/delayed or human control)"
                )

        # ONLY update position when:
        # 1. The executed action is a move
        # 2. The executed action matches what WE intended (not human control)
        # 3. We're not interacting with an object this step
        if (
            executed_action
            and executed_action.startswith("move_")
            and intended_action == executed_action  # Only if WE intended this move
            and not s.using_object_this_step
        ):
            direction = executed_action[5:]  # Remove "move_" prefix
            if direction in self._move_deltas:
                dr, dc = self._move_deltas[direction]
                s.row += dr
                s.col += dc

        s.using_object_this_step = False

        # Track position history
        current_pos = (s.row, s.col)
        s.position_history.append(current_pos)
        if len(s.position_history) > 30:
            s.position_history.pop(0)

    def _parse_observation(self, s: CogsguardAgentState, obs: AgentObservation) -> ParsedObservation:
        """Parse observation into structured format."""
        return utils_parse_observation(
            s,  # type: ignore[arg-type]  # CogsguardAgentState is compatible with SimpleAgentState
            obs,
            obs_hr=self._obs_hr,
            obs_wr=self._obs_wr,
            spatial_feature_names=self._spatial_feature_names,
            agent_feature_key_by_name=self._agent_feature_key_by_name,
            protocol_input_prefix=self._protocol_input_prefix,
            protocol_output_prefix=self._protocol_output_prefix,
            tag_names=self._tag_names,
        )

    def _update_occupancy_and_discover(self, s: CogsguardAgentState, parsed: ParsedObservation) -> None:
        """Update occupancy map and discover objects."""
        if s.row < 0:
            return

        # Mark all observed cells as FREE and explored
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                r, c = obs_r - self._obs_hr + s.row, obs_c - self._obs_wr + s.col
                if 0 <= r < s.map_height and 0 <= c < s.map_width:
                    s.occupancy[r][c] = CellType.FREE.value
                    s.explored[r][c] = True

        # Process discovered objects
        if DEBUG and s.step_count == 1:
            print(f"[A{s.agent_id}] Nearby objects: {[obj.name for obj in parsed.nearby_objects.values()]}")

        for pos, obj_state in parsed.nearby_objects.items():
            r, c = pos
            obj_name = obj_state.name.lower()

            # Walls are obstacles
            if is_wall(obj_name):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._update_structure(s, pos, obj_name, StructureType.WALL, obj_state)
                continue

            # Track other agents
            if obj_name == "agent" and obj_state.agent_id != s.agent_id:
                s.agent_occupancy.add((r, c))
                continue

            # Discover gear stations
            for _role, station_name in ROLE_TO_STATION.items():
                if is_station(obj_name, station_name) or station_name in obj_name:
                    s.occupancy[r][c] = CellType.OBSTACLE.value
                    struct_type = self._get_station_type(station_name)
                    self._update_structure(s, pos, obj_name, struct_type, obj_state)
                    # Legacy: update stations dict
                    if station_name not in s.stations or s.stations[station_name] is None:
                        s.stations[station_name] = pos
                        if DEBUG:
                            print(f"[A{s.agent_id}] DISCOVERED {station_name} at {pos}")
                    break

            # Discover supply depots (charger in cogsguard)
            if "supply_depot" in obj_name or "charger" in obj_name:
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._update_structure(s, pos, obj_name, StructureType.CHARGER, obj_state)
                # Legacy: update stations dict and supply_depots list
                if "charger" not in s.stations or s.stations["charger"] is None:
                    s.stations["charger"] = pos
                    if DEBUG:
                        print(f"[A{s.agent_id}] DISCOVERED charger/supply_depot at {pos}")
                self._discover_supply_depot(s, pos, obj_state)

            # Discover assembler (the resource deposit point)
            # In cogsguard, the assembler object has tag name "main_nexus"
            if "assembler" in obj_name or obj_name == "main_nexus":
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._update_structure(s, pos, obj_name, StructureType.ASSEMBLER, obj_state)
                # Legacy: update stations dict
                if "assembler" not in s.stations or s.stations["assembler"] is None:
                    s.stations["assembler"] = pos
                    if DEBUG:
                        print(f"[A{s.agent_id}] DISCOVERED assembler at {pos}")

            # Discover chest (for hearts) - NOT resource extractors, just "chest"
            # Must check this isn't a resource extractor (carbon_chest, etc)
            resources = ["carbon", "oxygen", "germanium", "silicon"]
            is_resource_chest = any(f"{res}_" in obj_name or f"{res}chest" in obj_name for res in resources)
            if obj_name == "chest" or ("chest" in obj_name and not is_resource_chest):
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._update_structure(s, pos, obj_name, StructureType.CHEST, obj_state)
                # Track chest for heart acquisition
                if "chest" not in s.stations or s.stations["chest"] is None:
                    s.stations["chest"] = pos
                    if DEBUG:
                        print(f"[A{s.agent_id}] DISCOVERED chest at {pos}")

            # Discover extractors (in cogsguard they're named {resource}_chest)
            for resource in ["carbon", "oxygen", "germanium", "silicon"]:
                if f"{resource}_extractor" in obj_name or f"{resource}_chest" in obj_name:
                    s.occupancy[r][c] = CellType.OBSTACLE.value
                    self._update_structure(s, pos, obj_name, StructureType.EXTRACTOR, obj_state, resource_type=resource)
                    # Legacy: update extractors dict
                    if resource in s.extractors:
                        self._discover_extractor(s, pos, resource, obj_state)
                        if DEBUG and s.step_count <= 30:
                            print(f"[A{s.agent_id}] DISCOVERED {resource} extractor at {pos}")
                    break

    def _get_station_type(self, station_name: str) -> StructureType:
        """Convert station name to StructureType."""
        mapping = {
            "miner_station": StructureType.MINER_STATION,
            "scout_station": StructureType.SCOUT_STATION,
            "aligner_station": StructureType.ALIGNER_STATION,
            "scrambler_station": StructureType.SCRAMBLER_STATION,
        }
        return mapping.get(station_name, StructureType.UNKNOWN)

    def _update_structure(
        self,
        s: CogsguardAgentState,
        pos: tuple[int, int],
        obj_name: str,
        structure_type: StructureType,
        obj_state: ObjectState,
        resource_type: Optional[str] = None,
    ) -> None:
        """Update or create a structure in the structures map."""
        # Derive alignment from clipped field, object name, and structure type
        clipped = obj_state.clipped > 0
        alignment = self._derive_alignment(obj_name, clipped, structure_type)

        # Calculate inventory amount for extractors
        # Key insight: empty dict {} on FIRST observation = no info yet (assume full)
        # Empty dict {} on SUBSEQUENT observation = depleted (0 resources)
        inventory_amount = 999  # Default: unknown/full
        is_new_structure = pos not in s.structures

        if structure_type == StructureType.EXTRACTOR:
            if resource_type and resource_type in obj_state.inventory:
                # We have actual inventory info
                inventory_amount = obj_state.inventory[resource_type]
            elif obj_state.inventory:
                # Sum all inventory if resource type not specified
                inventory_amount = sum(obj_state.inventory.values())
            elif not is_new_structure:
                # Empty dict for KNOWN extractor = depleted
                # (For new extractors, keep default 999)
                inventory_amount = 0
        elif obj_state.inventory:
            # Non-extractors: use inventory if present
            inventory_amount = sum(obj_state.inventory.values())

        if pos in s.structures:
            # Update existing structure
            struct = s.structures[pos]
            struct.last_seen_step = s.step_count
            struct.cooldown_remaining = obj_state.cooldown_remaining
            struct.remaining_uses = obj_state.remaining_uses
            struct.clipped = clipped
            struct.alignment = alignment
            struct.inventory_amount = inventory_amount
        else:
            # Create new structure
            s.structures[pos] = StructureInfo(
                position=pos,
                structure_type=structure_type,
                name=obj_name,
                last_seen_step=s.step_count,
                resource_type=resource_type,
                cooldown_remaining=obj_state.cooldown_remaining,
                remaining_uses=obj_state.remaining_uses,
                clipped=clipped,
                alignment=alignment,
                inventory_amount=inventory_amount,
            )
            if DEBUG:
                print(
                    f"[A{s.agent_id}] STRUCTURE: Added {structure_type.value} at {pos} "
                    f"(alignment={alignment}, inv={inventory_amount})"
                )

    def _derive_alignment(
        self, obj_name: str, clipped: bool, structure_type: Optional[StructureType] = None
    ) -> Optional[str]:
        """Derive alignment from object name, clipped status, and structure type.

        In CoGsGuard:
        - Assembler/nexus = cogs-aligned
        - Charger/supply_depot = clips-aligned (unless converted)
        """
        obj_lower = obj_name.lower()
        # Check if name contains alignment info
        if "cogs" in obj_lower or "cogs_" in obj_lower:
            return "cogs"
        if "clips" in obj_lower or "clips_" in obj_lower:
            return "clips"
        # Clipped field indicates clips alignment
        if clipped:
            return "clips"
        # Structure type defaults:
        # - Assembler/nexus defaults to cogs (main cogs building)
        # - Charger/supply_depot defaults to clips (enemy buildings to scramble)
        if structure_type == StructureType.ASSEMBLER:
            if "nexus" in obj_lower or "assembler" in obj_lower:
                return "cogs"
        if structure_type == StructureType.CHARGER:
            # Chargers start as clips-aligned (enemy), scrambler converts them
            return "clips"
        return None  # Unknown/neutral

    def _discover_supply_depot(self, s: CogsguardAgentState, pos: tuple[int, int], obj_state: ObjectState) -> None:
        """Track a supply depot with its alignment (legacy)."""
        # Derive alignment from clipped status
        alignment = "clips" if obj_state.clipped > 0 else None

        # Check if already tracked
        for i, (depot_pos, _) in enumerate(s.supply_depots):
            if depot_pos == pos:
                # Update alignment
                s.supply_depots[i] = (pos, alignment)
                return
        # Add new depot
        s.supply_depots.append((pos, alignment))

    def _discover_extractor(
        self,
        s: CogsguardAgentState,
        pos: tuple[int, int],
        resource_type: str,
        obj_state: ObjectState,
    ) -> None:
        """Track a discovered extractor (legacy)."""
        from cogames.policy.scripted_agent.types import ExtractorInfo

        for existing in s.extractors[resource_type]:
            if existing.position == pos:
                existing.cooldown_remaining = obj_state.cooldown_remaining
                existing.clipped = obj_state.clipped > 0
                existing.remaining_uses = obj_state.remaining_uses
                existing.last_seen_step = s.step_count
                return

        s.extractors[resource_type].append(
            ExtractorInfo(
                position=pos,
                resource_type=resource_type,
                last_seen_step=s.step_count,
                cooldown_remaining=obj_state.cooldown_remaining,
                clipped=obj_state.clipped > 0,
                remaining_uses=obj_state.remaining_uses,
            )
        )

    def _update_phase(self, s: CogsguardAgentState) -> None:
        """Update agent phase based on current vibe.

        Vibe-based state machine:
        - default/heart: do nothing
        - gear: pick random role, change vibe to that role
        - role vibe (scout/miner/aligner/scrambler): get gear first, then execute role
        """
        vibe = s.current_vibe

        # Role vibes: scout, miner, aligner, scrambler
        if vibe in VIBE_TO_ROLE:
            # Update role based on vibe
            s.role = VIBE_TO_ROLE[vibe]
            # Always try to get gear first, then execute role
            if s.has_gear():
                s.phase = CogsguardPhase.EXECUTE_ROLE
            elif s.step_count > 30:
                # After 30 steps, proceed without gear to bootstrap economy
                # Miners can mine with reduced capacity, others can still contribute
                s.phase = CogsguardPhase.EXECUTE_ROLE
            else:
                s.phase = CogsguardPhase.GET_GEAR
        else:
            # For default, heart, gear vibes - handled in _execute_phase
            s.phase = CogsguardPhase.GET_GEAR  # Will be overridden

    def _execute_phase(self, s: CogsguardAgentState) -> Action:
        """Execute action for current phase based on vibe.

        Vibe-based behavior:
        - default: do nothing (noop)
        - gear: pick random role, change vibe to that role
        - role vibe: get gear then execute role
        - heart: do nothing (noop)
        """
        vibe = s.current_vibe

        # Default vibe: do nothing (wait for external vibe change)
        if vibe == "default":
            return self._actions.noop.Noop()

        # Heart vibe: do nothing
        if vibe == "heart":
            return self._actions.noop.Noop()

        # Gear vibe: pick a role and change vibe to it
        if vibe == "gear":
            selected_role = random.choice(ROLE_VIBES)
            if DEBUG:
                print(f"[A{s.agent_id}] GEAR_VIBE: Picking role vibe: {selected_role}")
            return change_vibe_action(selected_role, actions=self._actions)

        # Role vibes: execute the role behavior
        if vibe in VIBE_TO_ROLE:
            if s.phase == CogsguardPhase.GET_GEAR:
                return self._do_get_gear(s)
            elif s.phase == CogsguardPhase.EXECUTE_ROLE:
                return self.execute_role(s)

        return self._actions.noop.Noop()

    def _do_recharge(self, s: CogsguardAgentState) -> Action:
        """Recharge by standing near the main nexus (cogs-aligned, has energy AOE).

        IMPORTANT: If energy is very low, we can't even move to the nexus!
        In that case, just wait (noop) and hope AOE regeneration eventually helps,
        or try a single step towards the nexus if we can afford it.
        """
        # The main_nexus is cogs-aligned and has AOE that gives energy to cogs agents
        # The supply_depot is clips-aligned and won't give energy to cogs agents
        nexus_pos = s.stations.get("assembler")
        if nexus_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] RECHARGE: No nexus found, exploring")
            return self._explore(s)

        # Just need to be near the nexus (within AOE range), not adjacent
        dist = abs(s.row - nexus_pos[0]) + abs(s.col - nexus_pos[1])
        aoe_range = 10  # AOE range from recipe

        if dist <= aoe_range:
            if DEBUG and s.step_count % 20 == 0:
                print(f"[A{s.agent_id}] RECHARGE: Near nexus (dist={dist}), waiting for AOE (energy={s.energy})")
            return self._actions.noop.Noop()

        # Check if we have enough energy to move at all
        # If energy is too low, just wait and hope for passive regen or AOE
        if s.energy < s.MOVE_ENERGY_COST:
            if DEBUG and s.step_count % 20 == 0:
                print(
                    f"[A{s.agent_id}] RECHARGE: Energy critically low ({s.energy}), "
                    f"can't move to nexus at dist={dist}, waiting for regen"
                )
            return self._actions.noop.Noop()

        # If we have some energy but not much, try to move one step at a time towards nexus
        # This is more conservative - don't commit to a long path if we might not make it
        if s.energy < s.MOVE_ENERGY_COST * 3:
            if DEBUG and s.step_count % 10 == 0:
                print(
                    f"[A{s.agent_id}] RECHARGE: Low energy ({s.energy}), "
                    f"taking single step towards nexus at {nexus_pos}"
                )
            # Simple single-step movement towards nexus
            dr = nexus_pos[0] - s.row
            dc = nexus_pos[1] - s.col
            if abs(dr) >= abs(dc):
                # Move vertically
                if dr > 0:
                    return self._actions.move.Move("south")
                else:
                    return self._actions.move.Move("north")
            else:
                # Move horizontally
                if dc > 0:
                    return self._actions.move.Move("east")
                else:
                    return self._actions.move.Move("west")

        if DEBUG and s.step_count % 20 == 0:
            print(f"[A{s.agent_id}] RECHARGE: Moving to nexus at {nexus_pos} from ({s.row},{s.col}), dist={dist}")
        return self._move_towards(s, nexus_pos, reach_adjacent=True)

    def _do_get_gear(self, s: CogsguardAgentState) -> Action:
        """Find gear station and equip gear."""
        station_name = s.get_gear_station_name()
        station_pos = s.stations.get(station_name)

        if DEBUG and s.step_count <= 10:
            print(f"[A{s.agent_id}] GET_GEAR: station={station_name} pos={station_pos} all={list(s.stations.keys())}")

        # Explore until we find the station
        if station_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] GET_GEAR: No {station_name} found, exploring")
            return self._explore(s)

        # Navigate to station
        adj = is_adjacent((s.row, s.col), station_pos)
        if DEBUG and s.step_count <= 60:
            print(f"[A{s.agent_id}] GET_GEAR: pos=({s.row},{s.col}), station={station_pos}, adjacent={adj}")
        if not adj:
            return self._move_towards(s, station_pos, reach_adjacent=True)

        # Bump station to get gear
        if DEBUG:
            print(f"[A{s.agent_id}] GET_GEAR: Adjacent to {station_name}, bumping it!")
        return self._use_object_at(s, station_pos)

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute role-specific behavior. Override in subclasses."""
        if s.step_count <= 100:
            print(f"[A{s.agent_id}] BASE_EXECUTE_ROLE: class={type(self).__name__}, role={s.role}")
        return self._explore(s)

    # =========================================================================
    # Navigation utilities
    # =========================================================================

    def _use_object_at(self, s: CogsguardAgentState, target_pos: tuple[int, int]) -> Action:
        """Use an object by moving into its cell."""
        tr, tc = target_pos
        if s.row == tr and s.col == tc:
            return self._actions.noop.Noop()

        dr = tr - s.row
        dc = tc - s.col

        # Check agent collision
        if (tr, tc) in s.agent_occupancy:
            return self._actions.noop.Noop()

        # Mark that we're using an object
        s.using_object_this_step = True

        if dr == -1:
            return self._actions.move.Move("north")
        if dr == 1:
            return self._actions.move.Move("south")
        if dc == 1:
            return self._actions.move.Move("east")
        if dc == -1:
            return self._actions.move.Move("west")

        return self._actions.noop.Noop()

    def _explore(self, s: CogsguardAgentState) -> Action:
        """Explore systematically - cycle through cardinal directions."""
        # Check for location loop (agents blocking each other back and forth)
        if self._is_in_location_loop(s):
            action = self._break_location_loop(s)
            if action:
                return action
            # If can't break loop, fall through to normal exploration

        # Start with east since gear stations are typically east of hub
        direction_cycle: list[CardinalDirection] = ["east", "south", "west", "north"]

        if DEBUG and s.step_count <= 30:
            print(f"[A{s.agent_id}] EXPLORE: target={s.exploration_target}, step={s.step_count}")

        if s.exploration_target is not None and isinstance(s.exploration_target, str):
            steps_in_direction = s.step_count - s.exploration_target_step
            if steps_in_direction < 8:  # Explore 8 steps before turning (faster cycles)
                dr, dc = self._move_deltas.get(s.exploration_target, (0, 0))
                next_r, next_c = s.row + dr, s.col + dc
                if path_is_traversable(s, next_r, next_c, CellType):  # type: ignore[arg-type]
                    return self._actions.move.Move(s.exploration_target)  # type: ignore[arg-type]

        # Pick next direction in the cycle (don't randomize)
        current_dir = s.exploration_target
        if current_dir in direction_cycle:
            idx = direction_cycle.index(current_dir)
            next_idx = (idx + 1) % 4
        else:
            # Always start with east (index 0) since gear stations are east of hub
            next_idx = 0

        # Try directions starting from next_idx
        for i in range(4):
            direction = direction_cycle[(next_idx + i) % 4]
            dr, dc = self._move_deltas[direction]
            next_r, next_c = s.row + dr, s.col + dc
            traversable = path_is_traversable(s, next_r, next_c, CellType)  # type: ignore[arg-type]
            if DEBUG and s.step_count <= 10:
                in_bounds = 0 <= next_r < s.map_height and 0 <= next_c < s.map_width
                cell_val = s.occupancy[next_r][next_c] if in_bounds else -1
                agent_occ = (next_r, next_c) in s.agent_occupancy
                print(
                    f"[A{s.agent_id}] EXPLORE_DIR: {direction} -> ({next_r},{next_c}) "
                    f"trav={traversable} cell={cell_val} agent={agent_occ}"
                )
            if traversable:
                s.exploration_target = direction
                s.exploration_target_step = s.step_count
                return self._actions.move.Move(direction)

        if DEBUG and s.step_count <= 10:
            print(f"[A{s.agent_id}] EXPLORE: All directions blocked, returning noop")
        return self._actions.noop.Noop()

    def _move_towards(
        self,
        s: CogsguardAgentState,
        target: tuple[int, int],
        *,
        reach_adjacent: bool = False,
    ) -> Action:
        """Pathfind toward a target."""
        # Check for location loop (agents blocking each other back and forth)
        if self._is_in_location_loop(s):
            action = self._break_location_loop(s)
            if action:
                return action
            # If can't break loop, fall through to normal pathfinding

        start = (s.row, s.col)
        if start == target and not reach_adjacent:
            return self._actions.noop.Noop()

        goal_cells = compute_goal_cells(s, target, reach_adjacent, CellType)  # type: ignore[arg-type]
        if not goal_cells:
            if DEBUG:
                print(f"[A{s.agent_id}] PATHFIND: No goal cells for {target}")
            return self._actions.noop.Noop()

        # Check cached path
        path = None
        if s.cached_path and s.cached_path_target == target and s.cached_path_reach_adjacent == reach_adjacent:
            next_pos = s.cached_path[0]
            if path_is_traversable(s, next_pos[0], next_pos[1], CellType):  # type: ignore[arg-type]
                path = s.cached_path

        # Compute new path if needed
        if path is None:
            path = shortest_path(s, start, goal_cells, False, CellType)  # type: ignore[arg-type]
            s.cached_path = path.copy() if path else None
            s.cached_path_target = target
            s.cached_path_reach_adjacent = reach_adjacent

        if not path:
            if DEBUG:
                print(f"[A{s.agent_id}] PATHFIND: No path to {target}, exploring instead")
            return self._explore(s)

        next_pos = path[0]

        # Advance cached path
        if s.cached_path:
            s.cached_path = s.cached_path[1:]
            if not s.cached_path:
                s.cached_path = None
                s.cached_path_target = None

        # Convert to action
        dr = next_pos[0] - s.row
        dc = next_pos[1] - s.col

        # Check agent collision
        if (next_pos[0], next_pos[1]) in s.agent_occupancy:
            return self._try_random_direction(s) or self._actions.noop.Noop()

        if dr == -1 and dc == 0:
            return self._actions.move.Move("north")
        elif dr == 1 and dc == 0:
            return self._actions.move.Move("south")
        elif dr == 0 and dc == 1:
            return self._actions.move.Move("east")
        elif dr == 0 and dc == -1:
            return self._actions.move.Move("west")

        return self._actions.noop.Noop()

    def _try_random_direction(self, s: CogsguardAgentState) -> Optional[Action]:
        """Try to move in a random free direction."""
        directions: list[CardinalDirection] = ["north", "south", "east", "west"]
        random.shuffle(directions)
        for direction in directions:
            dr, dc = self._move_deltas[direction]
            nr, nc = s.row + dr, s.col + dc
            if path_is_within_bounds(s, nr, nc) and s.occupancy[nr][nc] == CellType.FREE.value:  # type: ignore[arg-type]
                if (nr, nc) not in s.agent_occupancy:
                    return self._actions.move.Move(direction)
        return None

    def _is_in_location_loop(self, s: CogsguardAgentState) -> bool:
        """Detect if agent is stuck in a back-and-forth location loop.

        Detects patterns like A→B→A→B→A (oscillating between 2 positions 3+ times).
        Returns True if such a loop is detected.
        """
        history = s.position_history
        # Need at least 5 positions to detect A→B→A→B→A pattern
        if len(history) < 5:
            return False

        # Check last 6 positions for oscillation pattern
        recent = history[-6:] if len(history) >= 6 else history

        # Count unique positions in recent history
        unique_positions = set(recent)

        # If only 2 unique positions in last 6 moves, we're oscillating
        if len(unique_positions) <= 2 and len(recent) >= 5:
            if DEBUG:
                print(f"[A{s.agent_id}] LOOP_DETECTED: Oscillating between {unique_positions}")
            return True

        return False

    def _break_location_loop(self, s: CogsguardAgentState) -> Optional[Action]:
        """Try to break out of a location loop by moving in a random direction.

        Clears cached path to force re-pathing after breaking the loop.
        """
        if DEBUG:
            print(f"[A{s.agent_id}] BREAKING_LOOP: Attempting random move to escape")

        # Clear cached path to force fresh pathfinding
        s.cached_path = None
        s.cached_path_target = None

        # Clear position history to reset loop detection
        s.position_history.clear()

        return self._try_random_direction(s)


# =============================================================================
# Policy wrapper
# =============================================================================


class CogsguardPolicy(MultiAgentPolicy):
    """Multi-agent policy for CoGsGuard with vibe-based role selection.

    Agents use vibes to determine their behavior:
    - default: do nothing
    - gear: pick a random role, change vibe to that role
    - miner/scout/aligner/scrambler: get gear then execute role
    - heart: do nothing

    Initial vibe counts can be specified via URI query parameters:
        metta://policy/cogsguard?miner=4&scrambler=2&gear=1

    Vibes are assigned to agents in order. If counts don't sum to num_agents,
    remaining agents get "gear" vibe (which picks a random role).
    """

    short_names = ["cogsguard"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        **vibe_counts: int,
    ):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[CogsguardAgentState]] = {}

        # Build initial vibe assignment from URI params (e.g., ?scrambler=1&miner=4)
        counts = {k: v for k, v in vibe_counts.items() if isinstance(v, int)}

        # Build list of vibes to assign to agents
        self._initial_vibes: list[str] = []
        for vibe_name in ["scrambler", "aligner", "miner", "scout"]:  # Role vibes first
            self._initial_vibes.extend([vibe_name] * counts.get(vibe_name, 0))
        # Add gear vibes (agents will pick random role)
        self._initial_vibes.extend(["gear"] * counts.get("gear", 0))

        if DEBUG:
            print(f"[CogsguardPolicy] Initial vibe assignment: {self._initial_vibes}")

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[CogsguardAgentState]:
        if agent_id not in self._agent_policies:
            # Create a multi-role implementation that can handle any role
            # The actual role is determined by vibe at runtime
            # Assign initial target vibe based on agent_id and configured counts
            target_vibe: Optional[str] = None
            if agent_id < len(self._initial_vibes):
                target_vibe = self._initial_vibes[agent_id]
            # Agents without assigned vibes stay on "default" (noop)

            impl = CogsguardMultiRoleImpl(self._policy_env_info, agent_id, initial_target_vibe=target_vibe)
            self._agent_policies[agent_id] = StatefulAgentPolicy(impl, self._policy_env_info, agent_id=agent_id)

        return self._agent_policies[agent_id]


class CogsguardMultiRoleImpl(CogsguardAgentPolicyImpl):
    """Multi-role implementation that delegates to role-specific behavior based on vibe."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        initial_target_vibe: Optional[str] = None,
    ):
        # Initialize with MINER as default, but role will be updated based on vibe
        super().__init__(policy_env_info, agent_id, Role.MINER)

        # Target vibe to switch to at start (if specified)
        self._initial_target_vibe = initial_target_vibe
        self._initial_vibe_set = False

        # Lazy-load role implementations
        self._role_impls: dict[Role, CogsguardAgentPolicyImpl] = {}

    def _execute_phase(self, s: CogsguardAgentState) -> Action:
        """Execute action for current phase, handling initial vibe assignment.

        Overrides base class to:
        1. Handle initial vibe assignment from URI params
        2. Skip the hardcoded "agent 0 = scrambler" logic when initial vibe is configured
        """
        # If we have a target vibe and haven't switched yet, do it first
        if self._initial_target_vibe and not self._initial_vibe_set:
            if s.current_vibe != self._initial_target_vibe:
                if DEBUG:
                    print(
                        f"[A{s.agent_id}] INITIAL_VIBE: Switching from {s.current_vibe} to {self._initial_target_vibe}"
                    )
                return change_vibe_action(self._initial_target_vibe, actions=self._actions)
            self._initial_vibe_set = True

        # If initial target vibe was configured, skip the hardcoded agent 0 scrambler logic
        # by directly handling the vibe-based behavior here
        if self._initial_target_vibe:
            return self._execute_vibe_behavior(s)

        # Continue with normal phase execution (includes agent 0 scrambler logic)
        return super()._execute_phase(s)

    def _execute_vibe_behavior(self, s: CogsguardAgentState) -> Action:
        """Execute vibe-based behavior without the hardcoded agent 0 scrambler override."""
        vibe = s.current_vibe

        # Default vibe: do nothing (wait for external vibe change)
        if vibe == "default":
            return self._actions.noop.Noop()

        # Heart vibe: do nothing
        if vibe == "heart":
            return self._actions.noop.Noop()

        # Gear vibe: pick a role and change vibe to it
        if vibe == "gear":
            selected_role = random.choice(ROLE_VIBES)
            if DEBUG:
                print(f"[A{s.agent_id}] GEAR_VIBE: Picking role vibe: {selected_role}")
            return change_vibe_action(selected_role, actions=self._actions)

        # Role vibes: execute the role behavior
        if vibe in VIBE_TO_ROLE:
            if s.phase == CogsguardPhase.GET_GEAR:
                return self._do_get_gear(s)
            elif s.phase == CogsguardPhase.EXECUTE_ROLE:
                return self.execute_role(s)

        return self._actions.noop.Noop()

    def _get_role_impl(self, role: Role) -> CogsguardAgentPolicyImpl:
        """Get or create role-specific implementation."""
        if role not in self._role_impls:
            from .aligner import AlignerAgentPolicyImpl
            from .miner import MinerAgentPolicyImpl
            from .scout import ScoutAgentPolicyImpl
            from .scrambler import ScramblerAgentPolicyImpl

            impl_class = {
                Role.MINER: MinerAgentPolicyImpl,
                Role.SCOUT: ScoutAgentPolicyImpl,
                Role.ALIGNER: AlignerAgentPolicyImpl,
                Role.SCRAMBLER: ScramblerAgentPolicyImpl,
            }[role]

            self._role_impls[role] = impl_class(self._policy_env_info, self._agent_id, role)

        return self._role_impls[role]

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Delegate to role-specific implementation based on current role (set from vibe)."""
        role_impl = self._get_role_impl(s.role)
        return role_impl.execute_role(s)

"""
CoGsGuard Scripted Agent - Role-based multi-agent policy.

Agents are randomly assigned roles (Miner, Scout, Aligner, Scrambler) and:
1. Find their gear station and equip role-specific gear
2. Execute role-specific behavior
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from cogames.policy.scripted_agent.pathfinding import compute_goal_cells, shortest_path
from cogames.policy.scripted_agent.pathfinding import is_traversable as path_is_traversable
from cogames.policy.scripted_agent.pathfinding import is_within_bounds as path_is_within_bounds
from cogames.policy.scripted_agent.types import CellType, ObjectState, ParsedObservation
from cogames.policy.scripted_agent.utils import is_adjacent, is_station, is_wall
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

if TYPE_CHECKING:
    from mettagrid.simulator.interface import AgentObservation

# Debug flag - set to True to see detailed agent behavior
DEBUG = False


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
        """Initialize state for this agent."""
        self._tag_names = self._policy_env_info.tag_id_to_name

        map_size = 200
        center = map_size // 2

        return CogsguardAgentState(
            agent_id=self._agent_id,
            role=self._role,
            map_height=map_size,
            map_width=map_size,
            occupancy=[[CellType.FREE.value] * map_size for _ in range(map_size)],
            row=center,
            col=center,
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
                f"[A{s.agent_id}] Step {s.step_count}: {s.role.value} | "
                f"Phase={s.phase.value} | {gear_status} | "
                f"Energy={s.energy} | "
                f"Pos=({s.row},{s.col}) | "
                f"Nexus@{nexus_pos} | "
                f"Action={action.name}"
            )

        s.last_action = action
        return action, s

    def _read_inventory(self, s: CogsguardAgentState, obs: AgentObservation) -> None:
        """Read inventory from observation."""
        inv = {}
        center_r, center_c = self._obs_hr, self._obs_wr
        for tok in obs.tokens:
            if tok.location == (center_r, center_c):
                feature_name = tok.feature.name
                if feature_name.startswith("inv:"):
                    resource_name = feature_name[4:]
                    inv[resource_name] = tok.value

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

    def _update_agent_position(self, s: CogsguardAgentState) -> None:
        """Update position based on last action."""
        # Update position if last action was a move and we weren't using an object
        if s.last_action and s.last_action.name.startswith("move_") and not s.using_object_this_step:
            direction = s.last_action.name[5:]  # Remove "move_" prefix
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

        # Mark all observed cells as FREE initially
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                r, c = obs_r - self._obs_hr + s.row, obs_c - self._obs_wr + s.col
                if 0 <= r < s.map_height and 0 <= c < s.map_width:
                    s.occupancy[r][c] = CellType.FREE.value

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

            # Discover assembler/nexus
            if "assembler" in obj_name or "nexus" in obj_name:
                s.occupancy[r][c] = CellType.OBSTACLE.value
                self._update_structure(s, pos, obj_name, StructureType.ASSEMBLER, obj_state)
                # Legacy: update stations dict
                if "assembler" not in s.stations or s.stations["assembler"] is None:
                    s.stations["assembler"] = pos
                    if DEBUG:
                        print(f"[A{s.agent_id}] DISCOVERED assembler/nexus at {pos}")

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
        if pos in s.structures:
            # Update existing structure
            struct = s.structures[pos]
            struct.last_seen_step = s.step_count
            struct.cooldown_remaining = obj_state.cooldown_remaining
            struct.remaining_uses = obj_state.remaining_uses
            struct.clipped = obj_state.clipped > 0
            # TODO: Parse alignment from obj_state when available
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
                clipped=obj_state.clipped > 0,
            )
            if DEBUG:
                print(f"[A{s.agent_id}] STRUCTURE: Added {structure_type.value} at {pos}")

    def _discover_supply_depot(self, s: CogsguardAgentState, pos: tuple[int, int], obj_state: ObjectState) -> None:
        """Track a supply depot with its alignment (legacy)."""
        # Check if already tracked
        for _i, (depot_pos, _) in enumerate(s.supply_depots):
            if depot_pos == pos:
                # Update alignment if known
                # TODO: Parse alignment from obj_state when available
                return
        # Add new depot
        s.supply_depots.append((pos, None))

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
        """Update agent phase based on current state."""
        # Energy auto-regens near cogs nexus/chargers, so don't worry about it
        # Low energy just makes you slower, not a critical issue
        if s.has_gear():
            s.phase = CogsguardPhase.EXECUTE_ROLE
        elif s.role == Role.MINER and s.step_count > 30:
            # Miners can work without gear (with reduced cargo capacity of 4 vs 44)
            # This prevents them from getting stuck forever waiting for gear
            # when the commons doesn't have enough resources
            # After 30 steps, start mining with base capacity to bootstrap the economy
            s.phase = CogsguardPhase.EXECUTE_ROLE
        else:
            s.phase = CogsguardPhase.GET_GEAR

    def _execute_phase(self, s: CogsguardAgentState) -> Action:
        """Execute action for current phase."""
        if s.phase == CogsguardPhase.GET_GEAR:
            return self._do_get_gear(s)
        elif s.phase == CogsguardPhase.EXECUTE_ROLE:
            return self.execute_role(s)
        return self._actions.noop.Noop()

    def _do_recharge(self, s: CogsguardAgentState) -> Action:
        """Recharge by standing near the main nexus (cogs-aligned, has energy AOE)."""
        # The main_nexus is cogs-aligned and has AOE that gives energy to cogs agents
        # The supply_depot is clips-aligned and won't give energy to cogs agents
        nexus_pos = s.stations.get("assembler")
        if nexus_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] RECHARGE: No nexus found, exploring")
            return self._explore(s)

        # Just need to be near the nexus (within AOE range), not adjacent
        dist = abs(s.row - nexus_pos[0]) + abs(s.col - nexus_pos[1])
        if dist <= 5:  # Within AOE range
            if DEBUG:
                print(f"[A{s.agent_id}] RECHARGE: Near nexus (dist={dist}), waiting for AOE")
            return self._actions.noop.Noop()

        if DEBUG:
            print(f"[A{s.agent_id}] RECHARGE: Moving to nexus at {nexus_pos} from ({s.row},{s.col})")
        return self._move_towards(s, nexus_pos, reach_adjacent=True)

    def _do_get_gear(self, s: CogsguardAgentState) -> Action:
        """Find gear station and equip gear."""
        station_name = s.get_gear_station_name()
        station_pos = s.stations.get(station_name)

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
            if path_is_traversable(s, next_r, next_c, CellType):  # type: ignore[arg-type]
                s.exploration_target = direction
                s.exploration_target_step = s.step_count
                return self._actions.move.Move(direction)

        return self._actions.noop.Noop()

    def _move_towards(
        self,
        s: CogsguardAgentState,
        target: tuple[int, int],
        *,
        reach_adjacent: bool = False,
    ) -> Action:
        """Pathfind toward a target."""
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


# =============================================================================
# Policy wrapper
# =============================================================================


class CogsguardPolicy(MultiAgentPolicy):
    """Multi-agent policy for CoGsGuard with random role assignment."""

    short_names = ["cogsguard"]

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
    ):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[CogsguardAgentState]] = {}

        # Randomly assign roles to agents
        self._agent_roles: dict[int, Role] = {}
        roles = list(Role)
        for i in range(policy_env_info.num_agents):
            self._agent_roles[i] = roles[i % len(roles)]

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[CogsguardAgentState]:
        if agent_id not in self._agent_policies:
            role = self._agent_roles.get(agent_id, Role.MINER)

            # Import role implementations
            from .aligner import AlignerAgentPolicyImpl
            from .miner import MinerAgentPolicyImpl
            from .scout import ScoutAgentPolicyImpl
            from .scrambler import ScramblerAgentPolicyImpl

            # Create role-specific implementation
            impl_class = {
                Role.MINER: MinerAgentPolicyImpl,
                Role.SCOUT: ScoutAgentPolicyImpl,
                Role.ALIGNER: AlignerAgentPolicyImpl,
                Role.SCRAMBLER: ScramblerAgentPolicyImpl,
            }[role]

            impl = impl_class(self._policy_env_info, agent_id, role)
            self._agent_policies[agent_id] = StatefulAgentPolicy(impl, self._policy_env_info, agent_id=agent_id)

        return self._agent_policies[agent_id]

"""
Data types and structures for CoGsGuard scripted agents.

This module contains state, enums, and type definitions for role-based agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from mettagrid.simulator import Action

if TYPE_CHECKING:
    from mettagrid.simulator.interface import AgentObservation

from cogames.policy.scripted_agent.types import ExtractorInfo


class Role(Enum):
    """Agent roles in CoGsGuard."""

    MINER = "miner"
    SCOUT = "scout"
    ALIGNER = "aligner"
    SCRAMBLER = "scrambler"


class CogsguardPhase(Enum):
    """Phases for CoGsGuard agents."""

    GET_GEAR = "get_gear"  # Find and equip role-specific gear
    EXECUTE_ROLE = "execute_role"  # Execute role-specific behavior
    RECHARGE = "recharge"  # Recharge energy at charger


class StructureType(Enum):
    """Types of structures in the game."""

    ASSEMBLER = "assembler"  # Resource deposit point
    CHARGER = "charger"  # Supply depot
    MINER_STATION = "miner_station"
    SCOUT_STATION = "scout_station"
    ALIGNER_STATION = "aligner_station"
    SCRAMBLER_STATION = "scrambler_station"
    EXTRACTOR = "extractor"  # Resource extractor/chest
    CHEST = "chest"  # Heart acquisition point
    WALL = "wall"
    UNKNOWN = "unknown"


@dataclass
class StructureInfo:
    """Information about a discovered structure."""

    position: tuple[int, int]
    structure_type: StructureType
    name: str  # Original object name

    # Common attributes
    last_seen_step: int = 0

    # Alignment (for depots/assemblers): "cogs", "clips", or None (neutral)
    alignment: Optional[str] = None

    # Extractor-specific attributes
    resource_type: Optional[str] = None  # carbon, oxygen, germanium, silicon
    remaining_uses: int = 999
    cooldown_remaining: int = 0
    clipped: bool = False  # True if owned by clips
    inventory_amount: int = 999  # Current resource amount in extractor inventory

    def is_usable_extractor(self) -> bool:
        """Check if this is a usable extractor (not depleted, not clipped, has resources)."""
        return (
            self.structure_type == StructureType.EXTRACTOR
            and self.remaining_uses > 0
            and self.inventory_amount > 0
            and not self.clipped
        )


# Map roles to their gear station names
ROLE_TO_STATION = {
    Role.MINER: "miner_station",
    Role.SCOUT: "scout_station",
    Role.ALIGNER: "aligner_station",
    Role.SCRAMBLER: "scrambler_station",
}

# Map roles to the gear item name in inventory
ROLE_TO_GEAR = {
    Role.MINER: "miner",
    Role.SCOUT: "scout",
    Role.ALIGNER: "aligner",
    Role.SCRAMBLER: "scrambler",
}


@dataclass
class CogsguardAgentState:
    """State for a CoGsGuard agent."""

    agent_id: int
    role: Role

    # Current phase
    phase: CogsguardPhase = CogsguardPhase.GET_GEAR

    # Current vibe (read from observation)
    current_vibe: str = "default"

    step_count: int = 0

    # Position tracking (origin-relative)
    row: int = 0
    col: int = 0
    energy: int = 100

    # Map knowledge
    map_height: int = 200
    map_width: int = 200
    occupancy: list[list[int]] = field(default_factory=list)
    # Track which cells have been observed (explored)
    explored: list[list[bool]] = field(default_factory=list)

    # === Unified structure map ===
    # All discovered structures: position -> StructureInfo
    structures: dict[tuple[int, int], StructureInfo] = field(default_factory=dict)

    # Legacy fields (kept for backward compatibility, populated from structures)
    # Discovered stations: station_name -> position
    stations: dict[str, tuple[int, int] | None] = field(default_factory=dict)

    # Discovered extractors by resource type
    extractors: dict[str, list[ExtractorInfo]] = field(
        default_factory=lambda: {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
    )

    # Discovered supply depots: list of (position, alignment) tuples
    # alignment: "cogs", "clips", or None (neutral)
    supply_depots: list[tuple[tuple[int, int], Optional[str]]] = field(default_factory=list)

    # Inventory - gear items
    miner: int = 0
    scout: int = 0
    aligner: int = 0
    scrambler: int = 0

    # Inventory - resources
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    heart: int = 0
    influence: int = 0
    hp: int = 100

    @property
    def cargo_capacity(self) -> int:
        """Current cargo capacity based on miner gear.

        Base capacity is 4. Miner gear adds 40.
        This is dynamic - if gear is lost, capacity drops.
        """
        return 4 + (40 if self.miner > 0 else 0)

    @property
    def total_cargo(self) -> int:
        """Total resources currently carried."""
        return self.carbon + self.oxygen + self.germanium + self.silicon

    # Track last action for position updates
    last_action: Action = field(default_factory=lambda: Action(name="noop"))

    # Track what action the simulator actually executed (from observation)
    last_action_executed: Optional[str] = None

    # Navigation state
    target_position: Optional[tuple[int, int]] = None
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None
    cached_path_reach_adjacent: bool = False

    # Exploration state
    exploration_target: Optional[str] = None
    exploration_target_step: int = 0

    # Agent collision detection
    agent_occupancy: set[tuple[int, int]] = field(default_factory=set)

    # Position history for stuck detection
    position_history: list[tuple[int, int]] = field(default_factory=list)

    # Object interaction tracking
    using_object_this_step: bool = False

    # Current observation reference
    current_obs: Optional[AgentObservation] = None

    # Track chargers we've worked on (position -> last interaction step)
    # Used by aligners/scramblers to avoid getting stuck on the same charger
    worked_chargers: dict[tuple[int, int], int] = field(default_factory=dict)

    # Scrambler-specific tracking for heart acquisition timeout
    _heart_wait_start: int = 0
    _last_heart_count: int = 0

    # Action retry tracking
    _pending_action_type: Optional[str] = None  # "scramble", "align", "mine"
    _pending_action_target: Optional[tuple[int, int]] = None
    _action_retry_count: int = 0
    _pre_action_heart: int = 0  # Heart count before action attempt
    _pre_action_cargo: int = 0  # Cargo count before action attempt

    # Miner gear acquisition tracking
    _gear_attempt_step: int = 0  # Step when we last tried to get gear
    _resources_deposited_since_gear_attempt: int = 0  # Resources deposited since last gear attempt
    _gear_attempts_failed: int = 0  # Count of failed gear acquisition attempts (for scrambler)

    # Energy costs (can be overridden based on game config)
    MOVE_ENERGY_COST: int = 2  # Default move energy cost

    def has_gear(self) -> bool:
        """Check if agent has their role's gear equipped."""
        gear_name = ROLE_TO_GEAR[self.role]
        return getattr(self, gear_name, 0) > 0

    def get_gear_station_name(self) -> str:
        """Get the station name for this agent's role."""
        return ROLE_TO_STATION[self.role]

    # === Structure query methods ===

    def get_structures_by_type(self, structure_type: StructureType) -> list[StructureInfo]:
        """Get all structures of a given type."""
        return [s for s in self.structures.values() if s.structure_type == structure_type]

    def get_structure_at(self, pos: tuple[int, int]) -> Optional[StructureInfo]:
        """Get structure at a specific position."""
        return self.structures.get(pos)

    def get_nearest_structure(
        self,
        structure_type: StructureType,
        exclude: Optional[tuple[int, int]] = None,
    ) -> Optional[StructureInfo]:
        """Find the nearest structure of a given type."""
        best: Optional[StructureInfo] = None
        best_dist = float("inf")

        for struct in self.structures.values():
            if struct.structure_type != structure_type:
                continue
            if exclude and struct.position == exclude:
                continue

            dist = abs(struct.position[0] - self.row) + abs(struct.position[1] - self.col)
            if dist < best_dist:
                best = struct
                best_dist = dist

        return best

    def get_usable_extractors(self) -> list[StructureInfo]:
        """Get all usable extractors (not depleted, not clipped)."""
        return [s for s in self.structures.values() if s.is_usable_extractor()]

    def get_nearest_usable_extractor(self, exclude: Optional[tuple[int, int]] = None) -> Optional[StructureInfo]:
        """Find nearest usable extractor."""
        best: Optional[StructureInfo] = None
        best_dist = float("inf")

        for struct in self.structures.values():
            if not struct.is_usable_extractor():
                continue
            if exclude and struct.position == exclude:
                continue

            dist = abs(struct.position[0] - self.row) + abs(struct.position[1] - self.col)
            if dist < best_dist:
                best = struct
                best_dist = dist

        return best

    # === Action retry helpers ===

    def has_enough_energy_for_moves(self, num_moves: int) -> bool:
        """Check if agent has enough energy to make num_moves moves."""
        return self.energy >= num_moves * self.MOVE_ENERGY_COST

    def start_action_attempt(self, action_type: str, target: tuple[int, int]) -> None:
        """Start tracking an action attempt for retry purposes."""
        self._pending_action_type = action_type
        self._pending_action_target = target
        self._action_retry_count = 0
        self._pre_action_heart = self.heart
        self._pre_action_cargo = self.total_cargo

    def check_action_success(self) -> bool:
        """Check if the last action attempt succeeded based on state changes.

        Returns True if action succeeded or no action was pending.

        For moves: checks if last_action_executed matches last_action (intended).
        For scramble/align: checks if heart count decreased.
        For mine: checks if cargo increased.
        """
        if self._pending_action_type is None:
            return True

        action_type = self._pending_action_type

        # First check: did our intended action actually execute?
        # If we intended to move but executed noop, the action failed
        intended = self.last_action.name if self.last_action else None
        executed = self.last_action_executed
        if intended and executed and intended != executed:
            # Action failed at the move level - don't clear, allow retry
            return False

        # Check based on action type
        if action_type in ("scramble", "align"):
            # These actions consume 1 heart on success
            if self.heart < self._pre_action_heart:
                # Heart was consumed - action succeeded
                self.clear_pending_action()
                return True
            return False

        elif action_type == "mine":
            # Mining increases cargo
            if self.total_cargo > self._pre_action_cargo:
                self.clear_pending_action()
                return True
            return False

        # Unknown action type - assume success
        self.clear_pending_action()
        return True

    def increment_retry(self) -> int:
        """Increment retry count and return current count."""
        self._action_retry_count += 1
        return self._action_retry_count

    def clear_pending_action(self) -> None:
        """Clear pending action tracking."""
        self._pending_action_type = None
        self._pending_action_target = None
        self._action_retry_count = 0

    def should_retry_action(self, max_retries: int = 3) -> bool:
        """Check if we should retry the pending action."""
        if self._pending_action_type is None:
            return False
        return self._action_retry_count < max_retries

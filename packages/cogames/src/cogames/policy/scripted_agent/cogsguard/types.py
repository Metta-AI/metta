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

    ASSEMBLER = "assembler"  # Main nexus (cogs)
    CHARGER = "charger"  # Supply depot
    MINER_STATION = "miner_station"
    SCOUT_STATION = "scout_station"
    ALIGNER_STATION = "aligner_station"
    SCRAMBLER_STATION = "scrambler_station"
    EXTRACTOR = "extractor"  # Resource extractor/chest
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

    def is_usable_extractor(self) -> bool:
        """Check if this is a usable extractor (not depleted, not clipped)."""
        return self.structure_type == StructureType.EXTRACTOR and self.remaining_uses > 0 and not self.clipped


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

    step_count: int = 0

    # Position tracking (origin-relative)
    row: int = 0
    col: int = 0
    energy: int = 100

    # Map knowledge
    map_height: int = 200
    map_width: int = 200
    occupancy: list[list[int]] = field(default_factory=list)

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

    # Track last action for position updates
    last_action: Action = field(default_factory=lambda: Action(name="noop"))

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

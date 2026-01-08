"""
Data types and structures for scripted agents.

This module contains all the dataclasses, enums, and type definitions
used by the baseline and unclipping agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation


@dataclass
class BaselineHyperparameters:
    """Hyperparameters controlling baseline agent behavior."""

    # Energy management (recharge timing)
    recharge_threshold_low: int = 35  # Enter RECHARGE phase when energy < this
    recharge_threshold_high: int = 85  # Exit RECHARGE phase when energy >= this

    # Stuck detection and escape
    stuck_detection_enabled: bool = True  # Enable loop detection
    stuck_escape_distance: int = 12  # Minimum distance for escape target

    # Exploration parameters
    position_history_size: int = 30  # Size of position history buffer
    exploration_area_check_window: int = 30  # Steps to check for stuck area
    exploration_area_size_threshold: int = 7  # Max area size (height/width) to trigger escape
    exploration_escape_duration: int = 10  # Steps to navigate to assembler when stuck
    exploration_direction_persistence: int = 10  # Steps to persist in one direction
    exploration_assembler_distance_threshold: int = 10  # Min distance from assembler to trigger escape


# Hyperparameter Presets for Ensemble Creation
BASELINE_HYPERPARAMETER_PRESETS = {
    "default": BaselineHyperparameters(
        recharge_threshold_low=35,  # Moderate energy management
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=40,  # Thorough exploration: longer history
        exploration_area_check_window=35,  # Thorough exploration: longer check window
        exploration_area_size_threshold=9,  # Thorough exploration: larger area tolerance
        exploration_escape_duration=8,  # Thorough exploration: shorter escape duration
        exploration_direction_persistence=18,  # Thorough exploration: longer persistence
        exploration_assembler_distance_threshold=12,  # Thorough exploration: larger distance threshold
    ),
    "conservative": BaselineHyperparameters(
        recharge_threshold_low=50,  # Recharge early
        recharge_threshold_high=95,  # Stay charged
        stuck_detection_enabled=True,
        stuck_escape_distance=8,  # Shorter escape distance
        position_history_size=30,
        exploration_area_check_window=30,
        exploration_area_size_threshold=7,
        exploration_escape_duration=10,
        exploration_direction_persistence=10,
        exploration_assembler_distance_threshold=10,
    ),
    "aggressive": BaselineHyperparameters(
        recharge_threshold_low=20,  # Low energy tolerance
        recharge_threshold_high=80,  # Don't wait for full charge
        stuck_detection_enabled=True,
        stuck_escape_distance=15,  # Longer escape distance
        position_history_size=30,
        exploration_area_check_window=30,
        exploration_area_size_threshold=7,
        exploration_escape_duration=10,
        exploration_direction_persistence=10,
        exploration_assembler_distance_threshold=10,
    ),
}


class CellType(Enum):
    """Occupancy map cell states."""

    FREE = 1  # Passable (can walk through)
    OBSTACLE = 2  # Impassable (walls, stations, extractors)


class Phase(Enum):
    """Goal-driven phases for the baseline agent."""

    GATHER = "gather"  # Collect resources (explore if needed)
    ASSEMBLE = "assemble"  # Make hearts at assembler
    DELIVER = "deliver"  # Deposit hearts to chest
    RECHARGE = "recharge"  # Recharge energy at charger
    CRAFT_UNCLIP = "craft_unclip"  # Craft unclip items at assembler
    UNCLIP = "unclip"  # Unclip extractors


@dataclass
class ExtractorInfo:
    """Tracks a discovered extractor with full state."""

    position: tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon"
    last_seen_step: int
    times_used: int = 0

    # Extractor state from observations
    cooldown_remaining: int = 0  # Steps until ready
    clipped: bool = False  # Is it depleted?
    remaining_uses: int = 999  # How many uses left


@dataclass
class ObjectState:
    """State of a single object at a position."""

    name: str

    # Extractor/station features
    cooldown_remaining: int = 0
    clipped: int = 0
    remaining_uses: int = 999

    # Protocol details (recipes for assemblers/extractors)
    protocol_inputs: dict[str, int] = field(default_factory=dict)
    protocol_outputs: dict[str, int] = field(default_factory=dict)

    # Agent features (when object is another agent)
    agent_id: int = -1  # Which agent (-1 if not an agent) - NOT in observations, kept for API
    agent_group: int = -1  # Team/group
    agent_frozen: int = 0  # Is frozen?


@dataclass
class ParsedObservation:
    """Parsed observation data in a clean format."""

    # Agent state
    row: int
    col: int
    energy: int

    # Inventory
    carbon: int
    oxygen: int
    germanium: int
    silicon: int
    hearts: int
    decoder: int
    modulator: int
    resonator: int
    scrambler: int

    # Nearby objects with full state (position -> ObjectState)
    nearby_objects: dict[tuple[int, int], ObjectState]


@dataclass
class SimpleAgentState:
    """State for a single agent."""

    agent_id: int

    phase: Phase = Phase.GATHER  # Start gathering immediately
    step_count: int = 0

    # Current position (origin-relative, starting at (0, 0))
    row: int = 0
    col: int = 0
    energy: int = 100

    # Per-agent discovered extractors and stations (no shared state, each agent tracks independently)
    extractors: dict[str, list[ExtractorInfo]] = field(
        default_factory=lambda: {"carbon": [], "oxygen": [], "germanium": [], "silicon": []}
    )
    stations: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"assembler": None, "chest": None, "charger": None}
    )

    # Inventory
    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    hearts: int = 0
    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

    # Current target
    target_position: Optional[tuple[int, int]] = None
    target_resource: Optional[str] = None

    # Map knowledge
    map_height: int = 0
    map_width: int = 0
    occupancy: list[list[int]] = field(default_factory=list)  # 1=free, 2=obstacle (initialized in reset)

    # Track last action for position updates
    last_action: Action = field(default_factory=lambda: Action(name="noop"))

    # Current glyph (vibe) for interacting with assembler
    current_glyph: str = "default"

    # Discovered assembler recipe (dynamically discovered from observations)
    heart_recipe: Optional[dict[str, int]] = None

    # Extractor activation state
    waiting_at_extractor: Optional[tuple[int, int]] = None
    wait_steps: int = 0
    pending_use_resource: Optional[str] = None
    pending_use_amount: int = 0

    # Directional exploration state
    exploration_target: Optional[str] = None  # Current direction ("north", "south", "east", "west")
    exploration_target_step: int = 0  # When we set the direction
    exploration_escape_until_step: int = 0  # If > 0, we're in escape mode until this step

    # Agent positions (for collision detection)
    agent_occupancy: set[tuple[int, int]] = field(default_factory=set)

    # Stuck detection
    position_history: list[tuple[int, int]] = field(default_factory=list)  # Last 30 positions
    stuck_loop_detected: bool = False
    stuck_escape_step: int = 0

    # Path caching for efficient navigation (per-agent)
    cached_path: Optional[list[tuple[int, int]]] = None
    cached_path_target: Optional[tuple[int, int]] = None
    cached_path_reach_adjacent: bool = False
    using_object_this_step: bool = False  # Flag to prevent position update when using objects

    # Current observation (for collision detection and state updates)
    current_obs: Optional[AgentObservation] = None

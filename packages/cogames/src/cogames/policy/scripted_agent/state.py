"""Dataclasses capturing per-agent runtime state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from .phase_controller import GamePhase


@dataclass
class AgentState:
    agent_id: int = 0
    current_phase: GamePhase = GamePhase.GATHER_GERMANIUM
    phase_history: List[GamePhase] = field(default_factory=list)
    phase_runtime: Dict[str, Dict[str, int]] = field(default_factory=dict)
    resource_order: List[str] = field(default_factory=lambda: ["germanium", "silicon", "carbon", "oxygen"])
    current_glyph: str = "default"
    active_resource_target: Optional[str] = None
    heart_requirements: Dict[str, int] = field(default_factory=dict)
    is_leader: bool = False

    carbon: int = 0
    oxygen: int = 0
    germanium: int = 0
    silicon: int = 0
    energy: int = 100
    last_energy: int = 100
    energy_delta: int = 0
    heart: int = 0

    decoder: int = 0
    modulator: int = 0
    resonator: int = 0
    scrambler: int = 0

    hearts_assembled: int = 0
    wait_counter: int = 0
    just_deposited: bool = False

    agent_row: int = -1
    agent_col: int = -1
    home_base_row: int = -1
    home_base_col: int = -1
    assembler_discovered: bool = False
    chest_discovered: bool = False

    waiting_since_step: int = -1
    wait_target: Optional[Tuple[int, int]] = None
    unclip_target: Optional[Tuple[int, int]] = None
    unclip_recipes: Dict[str, str] = field(default_factory=dict)

    blocked_by_clipped_extractor: Optional[Tuple[int, int]] = None
    explore_goal: Optional[str] = None

    phase_entry_step: int = 0
    phase_entry_inventory: Dict[str, int] = field(default_factory=dict)
    unobtainable_resources: Set[str] = field(default_factory=set)
    resource_gathering_start: Dict[str, int] = field(default_factory=dict)
    resource_progress_tracking: Dict[str, int] = field(default_factory=dict)
    phase_visit_count: Dict[str, int] = field(default_factory=dict)

    step_count: int = 0
    last_heart: int = 0
    stuck_counter: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0

    occupancy_map: Optional[List[List[int]]] = None
    prev_pos: Optional[Tuple[int, int]] = None
    last_action_idx: Optional[int] = None
    last_attempt_was_use: bool = False
    visited_cells: Set[Tuple[int, int]] = field(default_factory=set)
    visit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)

    hub_probe_targets: Deque[Tuple[int, int]] = field(default_factory=deque)
    current_probe_target: Optional[Tuple[int, int]] = None
    hub_probe_initialized: bool = False

    # Occupancy tracking & cached exploration helpers
    occupancy_revision: int = 0
    frontier_cache: List[Tuple[int, int]] = field(default_factory=list)
    frontier_cache_revision: int = -1

    # Navigation caching
    nav_target: Optional[Tuple[int, int]] = None
    nav_path: Deque[Tuple[int, int]] = field(default_factory=deque)

    recharge_last_energy: int = -1
    recharge_ticks_without_gain: int = 0
    recharge_total_gained: int = 0

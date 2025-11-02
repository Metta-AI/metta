"""Shared constants and lookup tables for the scripted agent policy."""

from __future__ import annotations

from typing import Dict

from .phase_controller import GamePhase


class C:
    """Global numeric knobs used across the scripted policy."""

    # Inventory requirements
    REQ_CARBON: int = 20
    REQ_OXYGEN: int = 20
    REQ_SILICON: int = 50
    REQ_ENERGY: int = 20

    # Heart detection
    HEART_FEATURE: str = "inv:heart"
    HEART_SENTINEL_FIRST_FIELD: int = 85  # 0x55

    # Observation window (must match env)
    OBS_H: int = 11
    OBS_W: int = 11
    OBS_HR: int = OBS_H // 2
    OBS_WR: int = OBS_W // 2

    # Occupancy encoding
    OCC_UNKNOWN, OCC_FREE, OCC_OBSTACLE = 0, 1, 2

    # Scoring weights
    W_DISTANCE: float = 0.7
    W_EFFICIENCY: float = 0.3

    # Thresholds / heuristics
    DEFAULT_DEPLETION_THRESHOLD: float = 0.25
    LOW_DEPLETION_THRESHOLD: float = 0.25
    CLIPPED_SCORE_PENALTY: float = 0.5
    WAIT_IF_COOLDOWN_LEQ: int = 3  # try-use when <= this
    ROTATE_COOLDOWN_LT: int = 3  # consider alternatives if remaining < this
    ALT_ROTATE_RADIUS: int = 7
    PATIENCE_STEPS: int = 12  # how long to idle when waiting on cooldown
    RECHARGE_BUFFER: float = 5.0
    RECHARGE_IDLE_TOLERANCE: int = 3
    GATHER_BUFFER_SMALL: float = 10.0
    GATHER_BUFFER_LARGE: float = 5.0
    TASK_ENERGY_SILICON: int = 50

    # Planner knobs
    USE_ASTAR: bool = True
    ASTAR_THRESHOLD: int = 20
    FRONTIER_RADIUS_AROUND_HOME: int = 50
    FRONTIER_SPAWN_RADIUS: int = 30

    # Default (learned) cooldown fallbacks by resource
    DEFAULT_COOLDOWNS: Dict[str, int] = {
        "germanium": 0,
        "silicon": 0,
        "carbon": 10,
        "oxygen": 100,
        "charger": 10,
    }


class FeatureNames:
    """Pointers into the observation feature array."""

    TYPE_ID = "type_id"
    CONVERTING = "converting"
    COOLDOWN_REMAINING = "cooldown_remaining"
    CLIPPED = "clipped"
    REMAINING_USES = "remaining_uses"
    INV_CARBON = "inv:carbon"
    INV_OXYGEN = "inv:oxygen"
    INV_GERMANIUM = "inv:germanium"
    INV_SILICON = "inv:silicon"
    INV_ENERGY = "inv:energy"
    GLOBAL_LAST_REWARD = "global:last_reward"


class StationMaps:
    """Lookup tables translating stations ↔ glyphs/resources."""

    STATION_TO_GLYPH: Dict[str, str] = {
        "charger": "charger",
        "carbon_extractor": "carbon",
        "oxygen_extractor": "oxygen",
        "germanium_extractor": "germanium",
        "silicon_extractor": "silicon",
        "assembler": "heart",
        "chest": "chest",
    }

    STATION_TO_RESOURCE: Dict[str, str] = {
        "carbon_extractor": "carbon",
        "oxygen_extractor": "oxygen",
        "germanium_extractor": "germanium",
        "silicon_extractor": "silicon",
        "charger": "charger",
    }

    PHASE_TO_STATION: Dict[GamePhase, str | None] = {
        GamePhase.GATHER_GERMANIUM: "germanium_extractor",
        GamePhase.GATHER_SILICON: "silicon_extractor",
        GamePhase.GATHER_CARBON: "carbon_extractor",
        GamePhase.GATHER_OXYGEN: "oxygen_extractor",
        GamePhase.ASSEMBLE_HEART: "assembler",
        GamePhase.DEPOSIT_HEART: "chest",
        GamePhase.RECHARGE: "charger",
        GamePhase.UNCLIP_STATION: None,
        GamePhase.CRAFT_DECODER: "assembler",
    }

    PHASE_TO_GLYPH: Dict[GamePhase, str] = {
        GamePhase.ASSEMBLE_HEART: "heart",
        GamePhase.CRAFT_DECODER: "gear",
    }

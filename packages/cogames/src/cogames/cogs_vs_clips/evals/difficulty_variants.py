"""
Difficulty Variants for CoGames Missions

This module defines difficulty levels that can be applied to any mission to create
varied challenges. Each difficulty level modifies:
- max_uses (extractor depletion)
- efficiency (resource output per use)
- energy_regen (passive energy recovery)

The goal is to force agents to:
1. Explore wider to find multiple extractors
2. Learn about efficiency/depletion through observation
3. Adapt strategies based on resource availability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import Mission, MissionVariant

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

RESOURCE_KEYS = ("carbon", "oxygen", "germanium", "silicon")

# Solvability floors (non-breaking; keep extreme playable)
EFFICIENCY_FLOOR = 10
CHARGER_EFFICIENCY_FLOOR = 50
# Allow zero to persist for difficulties that force no passive regen
ENERGY_REGEN_FLOOR = 0


class DifficultyLevel(BaseModel):
    """Configuration for a difficulty level."""

    name: str = Field(description="Difficulty name (easy, medium, hard, brutal, etc.)")
    description: str = Field(description="What makes this difficulty challenging")
    allow_agent_scaling: bool = Field(default=True, description="Whether agent-count scaling helpers should run")

    # Extractor max_uses multipliers (relative to mission baseline)
    carbon_max_uses_mult: float = Field(default=1.0)
    oxygen_max_uses_mult: float = Field(default=1.0)
    germanium_max_uses_mult: float = Field(default=1.0)
    silicon_max_uses_mult: float = Field(default=1.0)

    # Extractor efficiency multipliers (relative to mission baseline)
    carbon_eff_mult: float = Field(default=1.0)
    oxygen_eff_mult: float = Field(default=1.0)
    germanium_eff_mult: float = Field(default=1.0)
    silicon_eff_mult: float = Field(default=1.0)
    charger_eff_mult: float = Field(default=1.0)

    # Energy regen multiplier (relative to mission baseline)
    energy_regen_mult: float = Field(default=1.0)

    # Absolute overrides (if set, ignore multipliers)
    carbon_max_uses_override: int | None = Field(default=None)
    oxygen_max_uses_override: int | None = Field(default=None)
    germanium_max_uses_override: int | None = Field(default=None)
    silicon_max_uses_override: int | None = Field(default=None)

    carbon_eff_override: int | None = Field(default=None)
    oxygen_eff_override: int | None = Field(default=None)
    germanium_eff_override: int | None = Field(default=None)
    silicon_eff_override: int | None = Field(default=None)
    charger_eff_override: int | None = Field(default=None)

    energy_regen_override: int | None = Field(default=None)
    move_energy_cost_override: int | None = Field(default=None)
    energy_capacity_override: int | None = Field(default=None)
    cargo_capacity_override: int | None = Field(default=None)
    max_steps_override: int | None = Field(default=None)

    # Clipping configuration
    clip_rate: float = Field(default=0.0, description="Probability per step that extractors get clipped")
    clip_target: str | None = Field(
        default=None, description="Specific extractor to clip (carbon/oxygen/germanium/silicon/charger)"
    )
    clip_immune_extractor: str | None = Field(default=None, description="Extractor that stays immune to clipping")


# =============================================================================
# Standard Difficulty Levels
# =============================================================================

STORY_MODE = DifficultyLevel(
    name="story_mode",
    description="Abundant energy/resource output so scripted agents can clear missions reliably",
    carbon_max_uses_override=12,
    oxygen_max_uses_override=12,
    germanium_max_uses_override=12,
    silicon_max_uses_override=12,
    carbon_eff_override=140,
    oxygen_eff_override=140,
    germanium_eff_override=140,
    silicon_eff_override=140,
    charger_eff_override=150,
    energy_regen_override=2,
    allow_agent_scaling=False,
)

STANDARD = DifficultyLevel(
    name="standard",
    description="Baseline mission parameters (legacy medium)",
)

HARD = DifficultyLevel(
    name="hard",
    description="Tight extractor budgets and no passive regen",
    carbon_max_uses_override=4,
    oxygen_max_uses_override=4,
    germanium_max_uses_override=6,
    silicon_max_uses_override=3,
    carbon_eff_override=80,
    oxygen_eff_override=65,
    germanium_eff_override=75,
    silicon_eff_override=70,
    charger_eff_override=80,
    energy_regen_override=0,
    move_energy_cost_override=3,
    allow_agent_scaling=False,
)

BRUTAL = DifficultyLevel(
    name="brutal",
    description="Extreme scarcity, reduced inventories, perfection required",
    carbon_max_uses_override=2,
    oxygen_max_uses_override=2,
    germanium_max_uses_override=3,
    silicon_max_uses_override=2,
    carbon_eff_override=55,
    oxygen_eff_override=45,
    germanium_eff_override=50,
    silicon_eff_override=50,
    charger_eff_override=60,
    energy_regen_override=0,
    move_energy_cost_override=3,
    energy_capacity_override=70,
    cargo_capacity_override=80,
    allow_agent_scaling=False,
)

SINGLE_USE = DifficultyLevel(
    name="single_use",
    description="Every extractor can be used exactly once - no second chances",
    carbon_max_uses_override=1,
    oxygen_max_uses_override=1,
    germanium_max_uses_override=1,
    silicon_max_uses_override=1,
    charger_eff_override=120,
    energy_regen_override=1,
    allow_agent_scaling=False,
)

SPEED_RUN = DifficultyLevel(
    name="speed_run",
    description="Short clock, cheap movement, efficient extraction",
    carbon_max_uses_override=6,
    oxygen_max_uses_override=6,
    germanium_max_uses_override=6,
    silicon_max_uses_override=6,
    carbon_eff_override=160,
    oxygen_eff_override=160,
    germanium_eff_override=160,
    silicon_eff_override=160,
    charger_eff_override=160,
    energy_regen_override=2,
    move_energy_cost_override=1,
    max_steps_override=600,
    allow_agent_scaling=True,
)

ENERGY_CRISIS = DifficultyLevel(
    name="energy_crisis",
    description="Zero passive regen and weak chargers - plan every move",
    charger_eff_override=50,
    energy_regen_override=0,
    allow_agent_scaling=False,
)

# =============================================================================
# Clipping Difficulty Variants
# =============================================================================

CLIPPED_OXYGEN = DifficultyLevel(
    name="clipped_oxygen",
    description="Oxygen extractor starts clipped - craft decoder from carbon to unclip",
    clip_rate=0.0,
    clip_target="oxygen",
    clip_immune_extractor="carbon_extractor",
)

CLIPPED_CARBON = DifficultyLevel(
    name="clipped_carbon",
    description="Carbon extractor starts clipped - craft modulator from oxygen to unclip",
    clip_rate=0.0,
    clip_target="carbon",
    clip_immune_extractor="oxygen_extractor",
)

CLIPPED_GERMANIUM = DifficultyLevel(
    name="clipped_germanium",
    description="Germanium extractor starts clipped - craft resonator from silicon to unclip",
    clip_rate=0.0,
    clip_target="germanium",
    clip_immune_extractor="silicon_extractor",
)

CLIPPED_SILICON = DifficultyLevel(
    name="clipped_silicon",
    description="Silicon extractor starts clipped - craft scrambler from germanium to unclip",
    clip_rate=0.0,
    clip_target="silicon",
    clip_immune_extractor="germanium_extractor",
)

CLIPPING_CHAOS = DifficultyLevel(
    name="clipping_chaos",
    description="Random extractors clip over time - must craft unclip items reactively",
    clip_rate=0.15,
    clip_target=None,
)

HARD_CLIPPED_OXYGEN = DifficultyLevel(
    name="hard_clipped_oxygen",
    description="Hard mode + oxygen starts clipped",
    carbon_max_uses_override=4,
    oxygen_max_uses_override=4,
    germanium_max_uses_override=6,
    silicon_max_uses_override=3,
    carbon_eff_override=80,
    oxygen_eff_override=65,
    germanium_eff_override=75,
    silicon_eff_override=70,
    charger_eff_override=80,
    energy_regen_override=0,
    move_energy_cost_override=3,
    clip_rate=0.0,
    clip_target="oxygen",
    clip_immune_extractor="carbon_extractor",
    allow_agent_scaling=False,
)


# =============================================================================
# Difficulty Registry
# =============================================================================

CANONICAL_DIFFICULTY_ORDER = [
    "story_mode",
    "standard",
    "hard",
    "brutal",
    "single_use",
    "speed_run",
    "energy_crisis",
    "clipped_oxygen",
    "clipped_carbon",
    "clipped_germanium",
    "clipped_silicon",
    "clipping_chaos",
    "hard_clipped_oxygen",
]

DIFFICULTY_LEVELS: dict[str, DifficultyLevel] = {
    "story_mode": STORY_MODE,
    "standard": STANDARD,
    "hard": HARD,
    "brutal": BRUTAL,
    "single_use": SINGLE_USE,
    "speed_run": SPEED_RUN,
    "energy_crisis": ENERGY_CRISIS,
    "clipped_oxygen": CLIPPED_OXYGEN,
    "clipped_carbon": CLIPPED_CARBON,
    "clipped_germanium": CLIPPED_GERMANIUM,
    "clipped_silicon": CLIPPED_SILICON,
    "clipping_chaos": CLIPPING_CHAOS,
    "hard_clipped_oxygen": HARD_CLIPPED_OXYGEN,
}

DifficultyName = Literal[
    "story_mode",
    "standard",
    "hard",
    "brutal",
    "single_use",
    "speed_run",
    "energy_crisis",
    "clipped_oxygen",
    "clipped_carbon",
    "clipped_germanium",
    "clipped_silicon",
    "clipping_chaos",
    "hard_clipped_oxygen",
]


def get_difficulty(name: DifficultyName) -> DifficultyLevel:
    """Get a difficulty level by name."""
    return DIFFICULTY_LEVELS[name]


def _apply_clipping(mission, difficulty: DifficultyLevel) -> None:
    """Apply clipping configuration from a difficulty level to a mission.

    This sets clip_rate, marks target extractors as start_clipped, and adds
    unclipping recipes and gear crafting recipes as needed.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Set clip_rate on mission
    if difficulty.clip_rate > 0.0:
        mission.clip_rate = difficulty.clip_rate

    # If no target specified, nothing to clip at start
    if not difficulty.clip_target or difficulty.clip_target == "none":
        return

    target = difficulty.clip_target

    # Set the specific station to start clipped
    try:
        if target == "carbon":
            mission.carbon_extractor.start_clipped = True
            logger.info("Set carbon_extractor.start_clipped = True")
        elif target == "oxygen":
            mission.oxygen_extractor.start_clipped = True
            logger.info(
                f"Set oxygen_extractor.start_clipped = True (current value: {mission.oxygen_extractor.start_clipped})"
            )
        elif target == "germanium":
            mission.germanium_extractor.start_clipped = True
            logger.info("Set germanium_extractor.start_clipped = True")
        elif target == "silicon":
            mission.silicon_extractor.start_clipped = True
            logger.info("Set silicon_extractor.start_clipped = True")
        elif target == "charger":
            mission.charger.start_clipped = True
            logger.info("Set charger.start_clipped = True")
    except Exception as e:
        # Some missions may not expose all station configs
        logger.error(f"Failed to set start_clipped on {target}: {e}")
        pass

    # Determine gear and resource mapping for unclipping
    gear_by_target: dict[str, tuple[str, str]] = {
        "carbon": ("modulator", "oxygen"),
        "oxygen": ("decoder", "carbon"),
        "germanium": ("resonator", "silicon"),
        "silicon": ("scrambler", "germanium"),
    }

    if target not in gear_by_target:
        return

    required_gear, resource_for_gear = gear_by_target[target]

    # Determine which extractor should be immune
    immune_extractor_name = difficulty.clip_immune_extractor or f"{resource_for_gear}_extractor"

    try:
        from cogames.cogs_vs_clips.mission_utils import _add_make_env_modifier
        from mettagrid.config.mettagrid_config import MettaGridConfig, ProtocolConfig
    except ImportError:
        logger.warning("Cannot import mission utilities for clipping config")
        return

    def _filter_unclip(cfg: MettaGridConfig) -> None:
        """Filter unclipping protocols to only the required gear."""
        if cfg.game.clipper is None:
            logger.warning("_filter_unclip: clipper is None")
            return
        try:
            original_count = (
                len(cfg.game.clipper.unclipping_protocols) if hasattr(cfg.game.clipper, "unclipping_protocols") else 0
            )
            cfg.game.clipper.unclipping_protocols = [
                r for r in cfg.game.clipper.unclipping_protocols if r.input_resources == {required_gear: 1}
            ]
            new_count = len(cfg.game.clipper.unclipping_protocols)
            logger.info(
                f"_filter_unclip: filtered unclipping protocols from {original_count} to {new_count} "
                f"(keeping {required_gear})"
            )
        except Exception as e:
            logger.error(f"_filter_unclip failed: {e}")
            pass

    def _tweak_assembler(cfg: MettaGridConfig) -> None:
        """Add gear crafting protocol to assembler."""
        print(f"[_tweak_assembler] Called with resource_for_gear={resource_for_gear}, required_gear={required_gear}")
        asm = cfg.game.objects.get("assembler")
        if asm is None:
            print("[_tweak_assembler] assembler not found")
            logger.warning("_tweak_assembler: assembler not found")
            return
        try:
            print(f"[_tweak_assembler] Current protocols ({len(asm.protocols)}):")
            for i, p in enumerate(asm.protocols):
                print(f"  [{i}] vibes={p.vibes}, in={p.input_resources}, out={p.output_resources}")

            protocol = ProtocolConfig(
                vibes=["gear"], input_resources={resource_for_gear: 1}, output_resources={required_gear: 1}
            )
            print(
                f"[_tweak_assembler] Created protocol: vibes=['gear'], "
                f"input={resource_for_gear}:1, output={required_gear}:1"
            )
            # Check if this protocol already exists
            if not any(p.vibes == ["gear"] and p.output_resources == {required_gear: 1} for p in asm.protocols):
                # APPEND to end (highest priority) instead of prepending (lowest priority)
                # Note: protocols are in "reverse order of priority" per AssemblerConfig
                asm.protocols = [*asm.protocols, protocol]
                print(
                    f"[_tweak_assembler] ✓ Added gear protocol {resource_for_gear} -> {required_gear} "
                    f"at END (highest priority)"
                )
                logger.info(f"_tweak_assembler: Added gear protocol {resource_for_gear} -> {required_gear}")
            else:
                print(f"[_tweak_assembler] gear protocol {resource_for_gear} -> {required_gear} already exists")
                logger.info(f"_tweak_assembler: gear protocol {resource_for_gear} -> {required_gear} already exists")
        except Exception as e:
            print(f"[_tweak_assembler] ERROR: {e}")
            logger.error(f"_tweak_assembler failed: {e}")
            pass

    def _ensure_gear_resource_immune(cfg: MettaGridConfig) -> None:
        """Make the extractor for the gear resource immune."""
        obj = cfg.game.objects.get(immune_extractor_name)
        if obj is None:
            return
        try:
            if hasattr(obj, "clip_immune"):
                obj.clip_immune = True
            if hasattr(obj, "start_clipped"):
                obj.start_clipped = False
        except Exception:
            pass

    def _ensure_critical_stations_immune(cfg: MettaGridConfig) -> None:
        """Make charger, assembler, and chest immune when clipping is active."""
        for station_name in ["charger", "assembler", "chest"]:
            obj = cfg.game.objects.get(station_name)
            if obj is None:
                continue
            try:
                if hasattr(obj, "clip_immune"):
                    obj.clip_immune = True
                if hasattr(obj, "start_clipped"):
                    obj.start_clipped = False
            except Exception:
                pass

    mission = _add_make_env_modifier(mission, _filter_unclip)
    logger.info("Added _filter_unclip modifier")
    mission = _add_make_env_modifier(mission, _tweak_assembler)
    mission = _add_make_env_modifier(mission, _ensure_gear_resource_immune)
    logger.info("Added _ensure_gear_resource_immune modifier")
    mission = _add_make_env_modifier(mission, _ensure_critical_stations_immune)
    logger.info("Added _ensure_critical_stations_immune modifier")


def apply_difficulty(
    mission,
    difficulty: DifficultyLevel,
) -> None:
    """Apply a difficulty level to a mission instance.

    Modifies the mission's extractor configs and energy_regen in place.

    Args:
        mission: Mission instance to modify
        difficulty: DifficultyLevel to apply
    """
    # Apply max_uses (override if set, else multiply), then enforce floor of 1 if baseline > 0
    for res in RESOURCE_KEYS:
        extractor = getattr(mission, f"{res}_extractor")
        override_val = getattr(difficulty, f"{res}_max_uses_override")
        mult_val = getattr(difficulty, f"{res}_max_uses_mult")
        if override_val is not None:
            extractor.max_uses = override_val
        else:
            try:
                mu = int(extractor.max_uses)
                scaled = int(mu * mult_val)
                extractor.max_uses = max(1, scaled) if mu > 0 else scaled
            except Exception:
                # Best-effort; leave as-is on failure
                pass

    # Apply efficiency (override if set, else multiply)
    for res in RESOURCE_KEYS:
        extractor = getattr(mission, f"{res}_extractor")
        override_val = getattr(difficulty, f"{res}_eff_override")
        mult_val = getattr(difficulty, f"{res}_eff_mult")
        if override_val is not None:
            extractor.efficiency = override_val
        else:
            try:
                eff = int(extractor.efficiency)
                extractor.efficiency = int(eff * mult_val)
            except Exception:
                pass

    # Charger efficiency
    if difficulty.charger_eff_override is not None:
        mission.charger.efficiency = difficulty.charger_eff_override
    else:
        mission.charger.efficiency = int(mission.charger.efficiency * difficulty.charger_eff_mult)

    # Energy regen
    if difficulty.energy_regen_override is not None:
        mission.energy_regen_amount = difficulty.energy_regen_override
    else:
        mission.energy_regen_amount = max(0, int(mission.energy_regen_amount * difficulty.energy_regen_mult))

    # Mission-level overrides
    if difficulty.move_energy_cost_override is not None:
        mission.move_energy_cost = difficulty.move_energy_cost_override
    if difficulty.energy_capacity_override is not None:
        mission.energy_capacity = difficulty.energy_capacity_override
    if difficulty.cargo_capacity_override is not None:
        mission.cargo_capacity = difficulty.cargo_capacity_override
    if difficulty.max_steps_override is not None:
        try:
            from cogames.cogs_vs_clips.missions import _add_make_env_modifier

            def _override_max_steps(cfg):
                cfg.game.max_steps = difficulty.max_steps_override

            _add_make_env_modifier(mission, _override_max_steps)
        except Exception:
            pass

    # Apply clipping configuration
    if difficulty.clip_rate > 0.0 or difficulty.clip_target is not None:
        _apply_clipping(mission, difficulty)

    if not difficulty.allow_agent_scaling:
        return

    # Post-build agent-aware scaling and solvability floors
    # - Scale extractor max_uses roughly with num_agents
    # - Mildly scale efficiency with num_agents
    # - Enforce minimal floors to keep extreme solvable
    try:
        # Lazy import to avoid circular imports at module import time
        from cogames.cogs_vs_clips.missions import _add_make_env_modifier

        def _scale_for_agents(cfg):
            try:
                num_agents = int(getattr(cfg.game, "num_agents", 1))
            except Exception:
                num_agents = 1

            # Efficiency scale: +20% per extra agent, capped at 2.0x
            eff_scale = 1.0 + 0.2 * max(0, num_agents - 1)
            if eff_scale > 2.0:
                eff_scale = 2.0

            for res in RESOURCE_KEYS:
                key = f"{res}_extractor"
                obj = cfg.game.objects.get(key)
                if obj is None:
                    continue
                try:
                    if hasattr(obj, "max_uses") and obj.max_uses is not None:
                        mu = int(obj.max_uses)
                        if mu > 0 and num_agents > 1:
                            obj.max_uses = max(1, mu * num_agents)
                        else:
                            obj.max_uses = max(1, mu)
                    if hasattr(obj, "efficiency"):
                        eff = int(obj.efficiency)
                        obj.efficiency = max(EFFICIENCY_FLOOR, int(eff * eff_scale))
                except Exception:
                    pass

            # Charger floor to avoid energy starvation unless explicitly zero-regen
            ch = cfg.game.objects.get("charger")
            if ch is not None and hasattr(ch, "efficiency"):
                try:
                    ch.efficiency = max(CHARGER_EFFICIENCY_FLOOR, int(ch.efficiency))
                except Exception:
                    pass

            # Energy regen floor: if nonzero, keep at least 1
            try:
                if cfg.game.agent.inventory_regen_amounts.get("energy", 1) > 0:
                    cfg.game.agent.inventory_regen_amounts["energy"] = max(
                        ENERGY_REGEN_FLOOR,
                        int(cfg.game.agent.inventory_regen_amounts.get("energy", 1)),
                    )
            except Exception:
                pass

        _add_make_env_modifier(mission, _scale_for_agents)
    except Exception:
        # If modifier cannot be attached (e.g., import timing), skip agent-aware scaling
        pass


def list_difficulties() -> None:
    """Print all available difficulty levels."""
    print("\nAvailable Difficulty Levels")
    print("=" * 80)
    for name, diff in DIFFICULTY_LEVELS.items():
        print(f"\n{name.upper()}: {diff.description}")
        print(
            f"  Max uses mult: C={diff.carbon_max_uses_mult}, O={diff.oxygen_max_uses_mult}, "
            f"G={diff.germanium_max_uses_mult}, S={diff.silicon_max_uses_mult}"
        )
        print(
            f"  Efficiency mult: C={diff.carbon_eff_mult}, O={diff.oxygen_eff_mult}, "
            f"G={diff.germanium_eff_mult}, S={diff.silicon_eff_mult}"
        )
        print(f"  Energy regen mult: {diff.energy_regen_mult}")


# =============================================================================
# MissionVariant Wrappers (for --variant CLI flag)
# =============================================================================


def create_difficulty_variant(difficulty_level: DifficultyLevel) -> type[MissionVariant]:
    """Create a MissionVariant class for a difficulty level."""
    from cogames.cogs_vs_clips.mission import MissionVariant

    class DifficultyMissionVariant(MissionVariant):
        name: str = difficulty_level.name
        description: str = difficulty_level.description

        def apply(self, mission: Mission) -> Mission:  # type: ignore[override]
            apply_difficulty(mission, difficulty_level)
            return mission

    return DifficultyMissionVariant


# Create mission variant classes for each difficulty level
StoryModeVariant = create_difficulty_variant(STORY_MODE)
StandardVariant = create_difficulty_variant(STANDARD)
HardVariant = create_difficulty_variant(HARD)
BrutalVariant = create_difficulty_variant(BRUTAL)
SingleUseVariant = create_difficulty_variant(SINGLE_USE)
SpeedRunVariant = create_difficulty_variant(SPEED_RUN)
EnergyCrisisVariant = create_difficulty_variant(ENERGY_CRISIS)
ClippedOxygenVariant = create_difficulty_variant(CLIPPED_OXYGEN)
ClippedCarbonVariant = create_difficulty_variant(CLIPPED_CARBON)
ClippedGermaniumVariant = create_difficulty_variant(CLIPPED_GERMANIUM)
ClippedSiliconVariant = create_difficulty_variant(CLIPPED_SILICON)
ClippingChaosVariant = create_difficulty_variant(CLIPPING_CHAOS)
HardClippedOxygenVariant = create_difficulty_variant(HARD_CLIPPED_OXYGEN)

# Export variants for use with --variant CLI flag
DIFFICULTY_VARIANTS = [
    StoryModeVariant,
    StandardVariant,
    HardVariant,
    BrutalVariant,
    SingleUseVariant,
    SpeedRunVariant,
    EnergyCrisisVariant,
    ClippedOxygenVariant,
    ClippedCarbonVariant,
    ClippedGermaniumVariant,
    ClippedSiliconVariant,
    ClippingChaosVariant,
    HardClippedOxygenVariant,
]


if __name__ == "__main__":
    list_difficulties()

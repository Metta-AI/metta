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

import logging
from typing import override

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig, ProtocolConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

RESOURCE_KEYS = ("carbon", "oxygen", "germanium", "silicon")

# Solvability floors (non-breaking; keep extreme playable)
EFFICIENCY_FLOOR = 10
CHARGER_EFFICIENCY_FLOOR = 50
# Allow zero to persist for difficulties that force no passive regen
ENERGY_REGEN_FLOOR = 0


# =============================================================================
# Difficulty Registry
# =============================================================================


class DifficultyLevel(MissionVariant):
    """Configuration for a difficulty level."""

    name: str = Field(description="Difficulty name (easy, medium, hard, brutal, etc.)")
    description: str = Field(description="What makes this difficulty challenging", default="")

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

    @override
    def modify_mission(self, mission: Mission):
        """Apply a difficulty level to a mission instance.

        Modifies the mission's extractor configs and energy_regen in place.

        Args:
            mission: Mission instance to modify
            difficulty: DifficultyLevel to apply
        """
        # Apply max_uses (override if set, else multiply), then enforce floor of 1 if baseline > 0
        for res in RESOURCE_KEYS:
            extractor = getattr(mission, f"{res}_extractor")
            override_val = getattr(self, f"{res}_max_uses_override")
            mult_val = getattr(self, f"{res}_max_uses_mult")
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
            override_val = getattr(self, f"{res}_eff_override")
            mult_val = getattr(self, f"{res}_eff_mult")
            if override_val is not None:
                extractor.efficiency = override_val
            else:
                try:
                    eff = int(extractor.efficiency)
                    extractor.efficiency = int(eff * mult_val)
                except Exception:
                    pass

        # Charger efficiency
        if self.charger_eff_override is not None:
            mission.charger.efficiency = self.charger_eff_override
        else:
            mission.charger.efficiency = int(mission.charger.efficiency * self.charger_eff_mult)

        # Energy regen
        if self.energy_regen_override is not None:
            mission.energy_regen_amount = self.energy_regen_override
        else:
            mission.energy_regen_amount = max(0, int(mission.energy_regen_amount * self.energy_regen_mult))

        # Mission-level overrides
        if self.move_energy_cost_override is not None:
            mission.move_energy_cost = self.move_energy_cost_override
        if self.energy_capacity_override is not None:
            mission.energy_capacity = self.energy_capacity_override
        if self.cargo_capacity_override is not None:
            mission.cargo_capacity = self.cargo_capacity_override

        # Set clip_rate on mission
        if self.clip_rate > 0.0:
            mission.clip_rate = self.clip_rate

        # Apply clipping configuration
        clip_target = self.clip_target

        # Set the specific station to start clipped
        if clip_target == "carbon":
            mission.carbon_extractor.start_clipped = True
            logger.info("Set carbon_extractor.start_clipped = True")
        elif clip_target == "oxygen":
            mission.oxygen_extractor.start_clipped = True
            logger.info(
                f"Set oxygen_extractor.start_clipped = True (current value: {mission.oxygen_extractor.start_clipped})"
            )
        elif clip_target == "germanium":
            mission.germanium_extractor.start_clipped = True
            logger.info("Set germanium_extractor.start_clipped = True")
        elif clip_target == "silicon":
            mission.silicon_extractor.start_clipped = True
            logger.info("Set silicon_extractor.start_clipped = True")
        elif clip_target == "charger":
            mission.charger.start_clipped = True
            logger.info("Set charger.start_clipped = True")

    @override
    def modify_env(self, mission: Mission, env: MettaGridConfig):
        if self.max_steps_override is not None:
            env.game.max_steps = self.max_steps_override

        if not self.allow_agent_scaling:
            return

        # Post-build agent-aware scaling and solvability floors
        # - Scale extractor max_uses roughly with num_agents
        # - Mildly scale efficiency with num_agents
        # - Enforce minimal floors to keep extreme solvable
        num_agents = env.game.num_agents

        # Efficiency scale: +20% per extra agent, capped at 2.0x
        eff_scale = 1.0 + 0.2 * max(0, num_agents - 1)
        if eff_scale > 2.0:
            eff_scale = 2.0

        for res in RESOURCE_KEYS:
            key = f"{res}_extractor"
            obj = env.game.objects.get(key)
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
        ch = env.game.objects.get("charger")
        if ch is not None and hasattr(ch, "efficiency"):
            try:
                ch.efficiency = max(CHARGER_EFFICIENCY_FLOOR, int(ch.efficiency))
            except Exception:
                pass

        # Energy regen floor: if nonzero, keep at least 1
        try:
            if env.game.agent.inventory_regen_amounts.get("energy", 1) > 0:
                env.game.agent.inventory_regen_amounts["energy"] = max(
                    ENERGY_REGEN_FLOOR,
                    int(env.game.agent.inventory_regen_amounts.get("energy", 1)),
                )
        except Exception:
            pass

        # Clipping
        self._apply_clipping(env)

    def _apply_clipping(self, cfg: MettaGridConfig) -> None:
        target = self.clip_target

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

        def _filter_unclip() -> None:
            """Filter unclipping protocols to only the required gear."""
            if cfg.game.clipper is None:
                logger.warning("_filter_unclip: clipper is None")
                return
            try:
                original_count = (
                    len(cfg.game.clipper.unclipping_protocols)
                    if hasattr(cfg.game.clipper, "unclipping_protocols")
                    else 0
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

        def _tweak_assembler() -> None:
            """Add gear crafting protocol to assembler."""
            print(
                f"[_tweak_assembler] Called with resource_for_gear={resource_for_gear}, required_gear={required_gear}"
            )
            asm = cfg.game.objects.get("assembler")
            if not isinstance(asm, AssemblerConfig):
                raise TypeError("Expected 'assembler' to be AssemblerConfig")
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
                    logger.info(
                        f"_tweak_assembler: gear protocol {resource_for_gear} -> {required_gear} already exists"
                    )
            except Exception as e:
                print(f"[_tweak_assembler] ERROR: {e}")
                logger.error(f"_tweak_assembler failed: {e}")
                pass

        def _ensure_gear_resource_immune() -> None:
            """Make the extractor for the gear resource immune."""
            # Determine which extractor should be immune
            immune_extractor_name = self.clip_immune_extractor or f"{resource_for_gear}_extractor"

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

        def _ensure_critical_stations_immune() -> None:
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

        _filter_unclip()
        logger.info("Added _filter_unclip modifier")
        _tweak_assembler()
        _ensure_gear_resource_immune()
        logger.info("Added _ensure_gear_resource_immune modifier")
        _ensure_critical_stations_immune()
        logger.info("Added _ensure_critical_stations_immune modifier")


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

# Export variants for use with --variant CLI flag.
# Ordered in canonical difficulty order.
DIFFICULTY_VARIANTS: list[DifficultyLevel] = [
    STORY_MODE,
    STANDARD,
    HARD,
    BRUTAL,
    SINGLE_USE,
    SPEED_RUN,
    ENERGY_CRISIS,
    CLIPPED_OXYGEN,
    CLIPPED_CARBON,
    CLIPPED_GERMANIUM,
    CLIPPED_SILICON,
    CLIPPING_CHAOS,
    HARD_CLIPPED_OXYGEN,
]


def get_difficulty(name: str) -> DifficultyLevel:
    """Get a difficulty level by name."""
    return next(difficulty for difficulty in DIFFICULTY_VARIANTS if difficulty.name == name)


def list_difficulties() -> None:
    """Print all available difficulty levels."""
    print("\nAvailable Difficulty Levels")
    print("=" * 80)
    for diff in DIFFICULTY_VARIANTS:
        print(f"\n{diff.name.upper()}: {diff.description}")
        print(
            f"  Max uses mult: C={diff.carbon_max_uses_mult}, O={diff.oxygen_max_uses_mult}, "
            f"G={diff.germanium_max_uses_mult}, S={diff.silicon_max_uses_mult}"
        )
        print(
            f"  Efficiency mult: C={diff.carbon_eff_mult}, O={diff.oxygen_eff_mult}, "
            f"G={diff.germanium_eff_mult}, S={diff.silicon_eff_mult}"
        )
        print(f"  Energy regen mult: {diff.energy_regen_mult}")


if __name__ == "__main__":
    list_difficulties()

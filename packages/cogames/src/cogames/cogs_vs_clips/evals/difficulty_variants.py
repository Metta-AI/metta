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
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

RESOURCE_KEYS = ("carbon", "oxygen", "germanium", "silicon")

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
    clip_period: int = Field(default=0, description="Approximate period of time between clips")
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
        # Note: GermaniumExtractorConfig doesn't have max_uses field (it's hardcoded to 1 in station_cfg)
        for res in RESOURCE_KEYS:
            extractor = getattr(mission, f"{res}_extractor")
            # Skip if extractor doesn't have max_uses attribute (e.g., GermaniumExtractorConfig)
            if not hasattr(extractor, "max_uses"):
                continue
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

        # Set clip_period on mission
        if self.clip_period > 0:
            mission.clip_period = self.clip_period

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

        # Apply clipping first (before agent scaling)
        self._apply_clipping(env)

        if not self.allow_agent_scaling:
            return

        # Post-build agent-aware scaling: scale extractor max_uses roughly with num_agents
        num_agents = env.game.num_agents

        # Scale extractor resources for multi-agent scenarios
        for res in RESOURCE_KEYS:
            key = f"{res}_extractor"
            obj = env.game.objects.get(key)
            if not isinstance(obj, AssemblerConfig):
                continue

            # Scale max_uses by agent count (leave unlimited=0 as-is)
            if obj.max_uses > 0 and num_agents > 1:
                obj.max_uses = obj.max_uses * num_agents

        # Energy regen floor: if nonzero, keep at least 1
        default_regen = env.game.agent.inventory.regen_amounts.get("default", {})
        current_regen = default_regen.get("energy", 1)
        if current_regen > 0:
            if "default" not in env.game.agent.inventory.regen_amounts:
                env.game.agent.inventory.regen_amounts["default"] = {}
            env.game.agent.inventory.regen_amounts["default"]["energy"] = max(ENERGY_REGEN_FLOOR, current_regen)

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
                logger.warning("_filter_unclip: clipper is None, skipping")
                return

            original_count = len(cfg.game.clipper.unclipping_protocols)
            cfg.game.clipper.unclipping_protocols = [
                r for r in cfg.game.clipper.unclipping_protocols if r.input_resources == {required_gear: 1}
            ]
            new_count = len(cfg.game.clipper.unclipping_protocols)
            logger.info(
                f"_filter_unclip: filtered unclipping protocols from {original_count} to {new_count} "
                f"(keeping {required_gear})"
            )

        def _add_gear_protocol() -> None:
            """Add generic ['gear'] protocol to assembler for this specific clipping variant.

            C++ only allows ONE protocol per unique vibe list, so we add just the one needed.
            """
            from mettagrid.config.mettagrid_config import AssemblerConfig, ProtocolConfig

            asm = cfg.game.objects.get("assembler")
            if asm is None or not isinstance(asm, AssemblerConfig):
                return

            # Check if ['gear'] protocol already exists
            if any(p.vibes == ["gear"] for p in asm.protocols):
                return  # Already added

            # Add the ONE generic gear protocol for this variant
            protocol = ProtocolConfig(
                vibes=["gear"], input_resources={resource_for_gear: 1}, output_resources={required_gear: 1}
            )
            asm.protocols.append(protocol)

        def _ensure_gear_resource_immune() -> None:
            """Make the extractor for the gear resource immune to clipping."""
            immune_extractor_name = self.clip_immune_extractor or f"{resource_for_gear}_extractor"
            obj = cfg.game.objects.get(immune_extractor_name)
            if not isinstance(obj, AssemblerConfig):
                return
            obj.clip_immune = True
            obj.start_clipped = False

        def _ensure_critical_stations_immune() -> None:
            """Make charger, assembler, and chest immune to clipping."""
            for station_name in ["charger", "assembler", "chest"]:
                obj = cfg.game.objects.get(station_name)
                if not isinstance(obj, AssemblerConfig):
                    continue
                obj.clip_immune = True
                obj.start_clipped = False

        # Apply clipping modifiers
        _filter_unclip()
        _add_gear_protocol()
        _ensure_gear_resource_immune()
        _ensure_critical_stations_immune()


# =============================================================================
# Standard Difficulty Levels
# =============================================================================

STANDARD = DifficultyLevel(
    name="standard",
    description="Baseline mission parameters (legacy medium)",
)

HARD = DifficultyLevel(
    name="hard",
    description="Tight extractor budgets and minimal passive regen",
    carbon_max_uses_override=4,
    oxygen_max_uses_override=4,
    germanium_max_uses_override=6,
    silicon_max_uses_override=3,
    carbon_eff_override=85,
    oxygen_eff_override=65,
    germanium_eff_override=75,
    silicon_eff_override=70,
    charger_eff_override=100,
    energy_regen_override=1,  # Minimal regen prevents deadlock
    move_energy_cost_override=2,
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
    description="Minimal passive regen and weak chargers - plan every move",
    charger_eff_override=50,
    energy_regen_override=1,  # Minimal regen prevents deadlock
    allow_agent_scaling=False,
)

# =============================================================================
# Clipping Difficulty Variants
# =============================================================================

CLIPPED_OXYGEN = DifficultyLevel(
    name="clipped_oxygen",
    description="Oxygen extractor starts clipped - craft decoder from carbon to unclip",
    clip_period=0,
    clip_target="oxygen",
    clip_immune_extractor="carbon_extractor",
    allow_agent_scaling=False,
)

CLIPPED_CARBON = DifficultyLevel(
    name="clipped_carbon",
    description="Carbon extractor starts clipped - craft modulator from oxygen to unclip",
    clip_period=0,
    clip_target="carbon",
    clip_immune_extractor="oxygen_extractor",
    allow_agent_scaling=False,
)

CLIPPED_GERMANIUM = DifficultyLevel(
    name="clipped_germanium",
    description="Germanium extractor starts clipped - craft resonator from silicon to unclip",
    clip_period=0,
    clip_target="germanium",
    clip_immune_extractor="silicon_extractor",
    allow_agent_scaling=False,
)

CLIPPED_SILICON = DifficultyLevel(
    name="clipped_silicon",
    description="Silicon extractor starts clipped - craft scrambler from germanium to unclip",
    clip_period=0,
    clip_target="silicon",
    clip_immune_extractor="germanium_extractor",
    allow_agent_scaling=False,
)

CLIPPING_CHAOS = DifficultyLevel(
    name="clipping_chaos",
    description="Random extractors clip over time - must craft unclip items reactively",
    clip_period=7,
    clip_target=None,
    allow_agent_scaling=False,
)

# Export variants for use with --variant CLI flag.
# Ordered in canonical difficulty order.
DIFFICULTY_VARIANTS: list[DifficultyLevel] = [
    STANDARD,
    HARD,
    SINGLE_USE,
    SPEED_RUN,
    ENERGY_CRISIS,
    CLIPPED_OXYGEN,
    CLIPPED_CARBON,
    CLIPPED_GERMANIUM,
    CLIPPED_SILICON,
    CLIPPING_CHAOS,
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

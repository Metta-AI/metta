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

from typing import Literal

from pydantic import BaseModel, Field

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
]

DIFFICULTY_LEVELS: dict[str, DifficultyLevel] = {
    "story_mode": STORY_MODE,
    "standard": STANDARD,
    "hard": HARD,
    "brutal": BRUTAL,
    "single_use": SINGLE_USE,
    "speed_run": SPEED_RUN,
    "energy_crisis": ENERGY_CRISIS,
}

# Legacy aliases for backwards compatibility
_ALIAS_MAP = {
    "easy": "story_mode",
    "medium": "standard",
    "extreme": "brutal",
}
DIFFICULTY_LEVELS.update({alias: DIFFICULTY_LEVELS[target] for alias, target in _ALIAS_MAP.items()})

DifficultyName = Literal[
    "story_mode",
    "standard",
    "hard",
    "brutal",
    "single_use",
    "speed_run",
    "energy_crisis",
    "easy",
    "medium",
    "extreme",
]


def get_difficulty(name: DifficultyName) -> DifficultyLevel:
    """Get a difficulty level by name."""
    return DIFFICULTY_LEVELS[name]


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


if __name__ == "__main__":
    list_difficulties()

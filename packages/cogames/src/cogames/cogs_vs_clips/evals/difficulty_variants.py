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
ENERGY_REGEN_FLOOR = 1  # applied only when nonzero


class DifficultyLevel(BaseModel):
    """Configuration for a difficulty level."""

    name: str = Field(description="Difficulty name (easy, medium, hard, extreme)")
    description: str = Field(description="What makes this difficulty challenging")

    # Extractor max_uses multipliers (relative to mission baseline)
    carbon_max_uses_mult: float = Field(default=1.0, description="Carbon max_uses multiplier")
    oxygen_max_uses_mult: float = Field(default=1.0, description="Oxygen max_uses multiplier")
    germanium_max_uses_mult: float = Field(default=1.0, description="Germanium max_uses multiplier")
    silicon_max_uses_mult: float = Field(default=1.0, description="Silicon max_uses multiplier")

    # Extractor efficiency multipliers (relative to mission baseline)
    carbon_eff_mult: float = Field(default=1.0, description="Carbon efficiency multiplier")
    oxygen_eff_mult: float = Field(default=1.0, description="Oxygen efficiency multiplier")
    germanium_eff_mult: float = Field(default=1.0, description="Germanium efficiency multiplier")
    silicon_eff_mult: float = Field(default=1.0, description="Silicon efficiency multiplier")
    charger_eff_mult: float = Field(default=1.0, description="Charger efficiency multiplier")

    # Energy regen multiplier (relative to mission baseline)
    energy_regen_mult: float = Field(default=1.0, description="Energy regen multiplier")

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


# =============================================================================
# Standard Difficulty Levels
# =============================================================================

EASY = DifficultyLevel(
    name="easy",
    description="Abundant resources, high efficiency, good energy regen - learn the basics",
    # High max_uses - can reuse extractors many times
    carbon_max_uses_mult=2.0,
    oxygen_max_uses_mult=2.0,
    germanium_max_uses_mult=2.0,
    silicon_max_uses_mult=2.0,
    # High efficiency - gather resources quickly
    carbon_eff_mult=1.5,
    oxygen_eff_mult=1.5,
    germanium_eff_mult=1.5,
    silicon_eff_mult=1.5,
    charger_eff_mult=1.5,
    # Good energy regen
    energy_regen_mult=1.5,
)

MEDIUM = DifficultyLevel(
    name="medium",
    description="Standard difficulty - balanced resources and efficiency",
    # Standard multipliers (1.0 = use mission baseline)
    carbon_max_uses_mult=1.0,
    oxygen_max_uses_mult=1.0,
    germanium_max_uses_mult=1.0,
    silicon_max_uses_mult=1.0,
    carbon_eff_mult=1.0,
    oxygen_eff_mult=1.0,
    germanium_eff_mult=1.0,
    silicon_eff_mult=1.0,
    charger_eff_mult=1.0,
    energy_regen_mult=1.0,
)

HARD = DifficultyLevel(
    name="hard",
    description="Scarce resources, lower efficiency - must explore and adapt",
    # Moderate max_uses - extractors WILL deplete, agent MUST find multiple sources
    carbon_max_uses_mult=0.65,  # Forces finding 2+ extractors
    oxygen_max_uses_mult=0.65,
    germanium_max_uses_mult=0.75,  # Germanium slightly more generous (always scarce)
    silicon_max_uses_mult=0.65,
    # Lower efficiency - gathering is slower, rewards efficient extractors
    carbon_eff_mult=0.80,
    oxygen_eff_mult=0.80,
    germanium_eff_mult=0.80,
    silicon_eff_mult=0.80,
    charger_eff_mult=0.80,
    # Lower energy regen - energy management matters
    energy_regen_mult=0.80,
)

EXTREME = DifficultyLevel(
    name="extreme",
    description="Brutal scarcity, minimal efficiency - perfect play required",
    # Low max_uses - agent MUST find ALL extractors and use them optimally
    carbon_max_uses_mult=0.45,  # Very tight budget
    oxygen_max_uses_mult=0.45,
    germanium_max_uses_mult=0.55,  # Germanium slightly more generous
    silicon_max_uses_mult=0.45,
    # Low efficiency - every extraction counts, efficiency tracking critical
    carbon_eff_mult=0.65,
    oxygen_eff_mult=0.65,
    germanium_eff_mult=0.65,
    silicon_eff_mult=0.65,
    charger_eff_mult=0.70,
    # Low energy regen - strategic energy management required
    energy_regen_mult=0.65,
)

# Special difficulty: Forces single-use behavior
SINGLE_USE = DifficultyLevel(
    name="single_use",
    description="Every extractor can be used exactly once - no second chances",
    # Override all max_uses to 1
    carbon_max_uses_override=1,
    oxygen_max_uses_override=1,
    germanium_max_uses_override=1,
    silicon_max_uses_override=1,
    # Standard efficiency
    carbon_eff_mult=1.0,
    oxygen_eff_mult=1.0,
    germanium_eff_mult=1.0,
    silicon_eff_mult=1.0,
    charger_eff_mult=1.0,
    # Standard energy regen
    energy_regen_mult=1.0,
)

# Special difficulty: High efficiency but very limited uses
SPEED_RUN = DifficultyLevel(
    name="speed_run",
    description="High efficiency but very limited uses - optimize routing",
    # Very low max_uses
    carbon_max_uses_mult=0.2,
    oxygen_max_uses_mult=0.2,
    germanium_max_uses_mult=0.3,
    silicon_max_uses_mult=0.2,
    # High efficiency - gather quickly
    carbon_eff_mult=2.0,
    oxygen_eff_mult=2.0,
    germanium_eff_mult=2.0,
    silicon_eff_mult=2.0,
    charger_eff_mult=2.0,
    # High energy regen - focus on routing, not energy
    energy_regen_mult=2.0,
)

# Special difficulty: Energy crisis
ENERGY_CRISIS = DifficultyLevel(
    name="energy_crisis",
    description="Zero energy regen, limited charger efficiency - plan every move",
    # Standard max_uses
    carbon_max_uses_mult=1.0,
    oxygen_max_uses_mult=1.0,
    germanium_max_uses_mult=1.0,
    silicon_max_uses_mult=1.0,
    # Standard resource efficiency
    carbon_eff_mult=1.0,
    oxygen_eff_mult=1.0,
    germanium_eff_mult=1.0,
    silicon_eff_mult=1.0,
    # Low charger efficiency
    charger_eff_mult=0.6,
    # ZERO energy regen!
    energy_regen_override=0,
)


# =============================================================================
# Difficulty Registry
# =============================================================================

DIFFICULTY_LEVELS: dict[str, DifficultyLevel] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
    "extreme": EXTREME,
    "single_use": SINGLE_USE,
    "speed_run": SPEED_RUN,
    "energy_crisis": ENERGY_CRISIS,
}

DifficultyName = Literal["easy", "medium", "hard", "extreme", "single_use", "speed_run", "energy_crisis"]


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

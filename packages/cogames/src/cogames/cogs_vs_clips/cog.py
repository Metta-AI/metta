from __future__ import annotations

from pydantic import Field

from mettagrid.base_config import Config


class CogConfig(Config):
    """Configuration for cog agents in CogsGuard game mode."""

    # Inventory limits
    gear_limit: int = Field(default=1)
    hp_limit: int = Field(default=100)
    heart_limit: int = Field(default=10)
    energy_limit: int = Field(default=10)
    cargo_limit: int = Field(default=4)
    influence_limit: int = Field(default=0)

    # Inventory modifiers by gear type
    hp_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 400, "scrambler": 200})
    energy_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 100})
    cargo_modifiers: dict[str, int] = Field(default_factory=lambda: {"miner": 40})
    influence_modifiers: dict[str, int] = Field(default_factory=lambda: {"aligner": 20})

    # Initial inventory
    initial_energy: int = Field(default=100)
    initial_hp: int = Field(default=50)

    # Regen amounts
    energy_regen: int = Field(default=1)
    hp_regen: int = Field(default=-1)
    influence_regen: int = Field(default=-1)

    # Movement cost
    move_energy_cost: int = Field(default=3)

"""Action configuration classes for MettaGrid.

This module defines all action-related configurations including movement,
attacks, transfers, and vibe changes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, get_args

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config.vibes import VIBES, Vibe
from mettagrid.simulator import Action

# ===== Direction Types =====

Direction = Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
Directions = list(get_args(Direction))

# Order must match C++ expectations: north, south, west, east
CardinalDirection = Literal["north", "south", "west", "east"]
CardinalDirections = list(get_args(CardinalDirection))


# ===== Action Configuration Classes =====


class ActionConfig(Config):
    """Python action configuration."""

    action_handler: str
    enabled: bool = Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: dict[str, int] = Field(default_factory=dict)
    consumed_resources: dict[str, int] = Field(default_factory=dict)

    def actions(self) -> list[Action]:
        if self.enabled:
            return self._actions()
        return []

    @abstractmethod
    def _actions(self) -> list[Action]: ...


class NoopActionConfig(ActionConfig):
    """Noop action configuration."""

    action_handler: str = Field(default="noop")

    def _actions(self) -> list[Action]:
        return [self.Noop()]

    def Noop(self) -> Action:
        return Action(name="noop")


class MoveActionConfig(ActionConfig):
    """Move action configuration."""

    action_handler: str = Field(default="move")
    allowed_directions: list[Direction] = Field(default_factory=lambda: CardinalDirections)

    def _actions(self) -> list[Action]:
        return [self.Move(direction) for direction in self.allowed_directions]

    def Move(self, direction: Direction) -> Action:
        return Action(name=f"move_{direction}")


class ChangeVibeActionConfig(ActionConfig):
    """Change vibe action configuration."""

    action_handler: str = Field(default="change_vibe")
    vibes: list[Vibe] = Field(default_factory=lambda: list(VIBES))

    def _actions(self) -> list[Action]:
        return [self.ChangeVibe(vibe) for vibe in self.vibes]

    def ChangeVibe(self, vibe: Vibe) -> Action:
        return Action(name=f"change_vibe_{vibe.name}")


class AttackOutcome(Config):
    """Outcome configuration for successful attack."""

    actor_inv_delta: dict[str, int] = Field(
        default_factory=dict,
        description="Inventory changes for attacker. Maps resource name to delta.",
    )
    target_inv_delta: dict[str, int] = Field(
        default_factory=dict,
        description="Inventory changes for target. Maps resource name to delta.",
    )
    loot: list[str] = Field(
        default_factory=list,
        description="Resources to steal from target.",
    )
    freeze: int = Field(
        default=0,
        description="Freeze duration (0 = no freeze).",
    )


class AttackActionConfig(ActionConfig):
    """Python attack action configuration.

    Attack is triggered by moving onto another agent (when vibes match).
    No standalone attack actions are created.

    Enhanced attack system with armor/weapon modifiers:
    - defense_resources: Base resources needed to block an attack
    - armor_resources: Target's resources that reduce incoming damage (weighted)
    - weapon_resources: Attacker's resources that increase damage (weighted)
    - success: Outcome when attack succeeds (actor/target inventory changes, freeze)
    - vibe_bonus: Per-vibe armor bonus when vibing a matching resource

    Defense calculation:
    - weapon_power = sum(attacker_inventory[item] * weapon_weight)
    - armor_power = sum(target_inventory[item] * armor_weight) + vibe_bonus[target_vibe] if vibing
    - damage_bonus = max(weapon_power - armor_power, 0)
    - cost_to_defend = defense_resources + damage_bonus
    """

    action_handler: str = Field(default="attack")
    defense_resources: dict[str, int] = Field(default_factory=dict)
    armor_resources: dict[str, int] = Field(
        default_factory=dict,
        description="Resources on target that reduce damage. Maps resource name to weight.",
    )
    weapon_resources: dict[str, int] = Field(
        default_factory=dict,
        description="Resources on attacker that increase damage. Maps resource name to weight.",
    )
    success: AttackOutcome = Field(
        default_factory=AttackOutcome,
        description="Outcome when attack succeeds.",
    )
    vibes: list[str] = Field(
        default_factory=list,
        description="Vibe names that trigger attack on move (e.g., ['weapon'])",
    )
    vibe_bonus: dict[str, int] = Field(
        default_factory=dict,
        description="Per-vibe armor bonus. Maps vibe name to bonus amount.",
    )

    def _actions(self) -> list[Action]:
        # Attack only triggers via move, no standalone actions
        return []


class VibeTransfer(Config):
    """Configuration for resource transfers triggered by a specific vibe.

    When an agent with this vibe moves into another agent,
    the specified resource deltas are applied to both the actor and target.

    Example:
        VibeTransfer(
            vibe="battery",
            target={"energy": 50},      # target gains 50 energy
            actor={"energy": -50}       # actor loses 50 energy
        )
    """

    vibe: str
    target: dict[str, int] = Field(default_factory=dict)
    actor: dict[str, int] = Field(default_factory=dict)


class TransferActionConfig(ActionConfig):
    """Python transfer action configuration.

    Transfer is triggered by move when the agent's vibe matches a vibe in vibe_transfers.
    The vibe_transfers list specifies what resource effects happen for each vibe.
    """

    action_handler: str = Field(default="transfer")
    vibe_transfers: list[VibeTransfer] = Field(
        default_factory=list,
        description="List of vibe transfer configs specifying actor/target resource effects",
    )

    def _actions(self) -> list[Action]:
        # Transfer doesn't create standalone actions - it's triggered by move
        return []


class AlignActionConfig(ActionConfig):
    """Python align action configuration.

    Align is triggered by move when the agent's vibe matches a vibe in vibes.
    When triggered, attempts to align the target to the actor's collective.
    """

    action_handler: str = Field(default="align")
    vibes: list[str] = Field(
        default_factory=list,
        description="Vibe names that trigger align on move",
    )

    def _actions(self) -> list[Action]:
        # Align doesn't create standalone actions - it's triggered by move
        return []


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: NoopActionConfig = Field(default_factory=lambda: NoopActionConfig())
    move: MoveActionConfig = Field(default_factory=lambda: MoveActionConfig())
    attack: AttackActionConfig = Field(default_factory=lambda: AttackActionConfig(enabled=False))
    transfer: TransferActionConfig = Field(default_factory=lambda: TransferActionConfig(enabled=False))
    change_vibe: ChangeVibeActionConfig = Field(default_factory=lambda: ChangeVibeActionConfig())
    align: AlignActionConfig = Field(default_factory=lambda: AlignActionConfig(enabled=False))

    def actions(self) -> list[Action]:
        return sum(
            [
                action.actions()
                for action in [self.noop, self.move, self.attack, self.transfer, self.change_vibe, self.align]
            ],
            [],
        )

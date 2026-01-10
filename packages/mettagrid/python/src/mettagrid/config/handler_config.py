"""Handler configuration classes and helper functions.

This module provides a data-driven system for configuring handlers on GridObjects.
There are three types of handlers:
  - on_use: Triggered when agent uses/activates an object (context: actor=agent, target=object)
  - on_update: Triggered after mutations are applied (context: actor=null, target=object)
  - aoe: Triggered per-tick for objects within radius (context: actor=source, target=affected)

Handlers consist of filters (conditions that must be met) and mutations (effects that are applied).
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, Literal, Union

from pydantic import Discriminator, Field, Tag

from mettagrid.base_config import Config

# ===== Handler System =====
# Data-driven system for configuring what happens when handlers trigger


class HandlerTarget(StrEnum):
    """Target entity for filter/mutation operations."""

    ACTOR = auto()
    TARGET = auto()
    ACTOR_COLLECTIVE = auto()
    TARGET_COLLECTIVE = auto()


class AlignmentTarget(StrEnum):
    """Target entity for alignment checks (subset of HandlerTarget)."""

    ACTOR = auto()
    TARGET = auto()


class AlignmentCondition(StrEnum):
    """Conditions for alignment filter checks."""

    ALIGNED = auto()  # target has any collective
    UNALIGNED = auto()  # target has no collective
    SAME_COLLECTIVE = auto()  # target has same collective as actor
    DIFFERENT_COLLECTIVE = auto()  # target has different collective than actor (but is aligned)
    NOT_SAME_COLLECTIVE = auto()  # target is not aligned to actor (unaligned OR different_collective)


class Filter(Config):
    """Base class for handler filters. All filters in a handler must pass."""

    target: HandlerTarget = Field(
        default=HandlerTarget.ACTOR,
        description="Entity to check the filter against",
    )


class VibeFilter(Filter):
    """Filter that checks if the target entity has a specific vibe."""

    filter_type: Literal["vibe"] = "vibe"
    vibe: str = Field(description="Vibe name that must match")


class ResourceFilter(Filter):
    """Filter that checks if the target entity has required resources."""

    filter_type: Literal["resource"] = "resource"
    resources: dict[str, int] = Field(
        default_factory=dict,
        description="Minimum resource amounts required",
    )


class AlignmentFilter(Filter):
    """Filter that checks the alignment status of a target.

    Can check if target is aligned/unaligned, or if it's aligned to
    the same/different collective as the actor.
    """

    filter_type: Literal["alignment"] = "alignment"
    target: AlignmentTarget = Field(
        default=AlignmentTarget.TARGET,
        description="Entity to check the filter against (only actor/target for alignment)",
    )
    alignment: AlignmentCondition = Field(
        description=(
            "Alignment condition to check: "
            "'aligned' = target has any collective, "
            "'unaligned' = target has no collective, "
            "'same_collective' = target has same collective as actor, "
            "'different_collective' = target has different collective than actor (but is aligned), "
            "'not_same_collective' = target is not aligned to actor (unaligned OR different_collective)"
        ),
    )


AnyFilter = Annotated[
    Union[
        Annotated[VibeFilter, Tag("vibe")],
        Annotated[ResourceFilter, Tag("resource")],
        Annotated[AlignmentFilter, Tag("alignment")],
    ],
    Discriminator("filter_type"),
]


class AOEEffectConfig(Config):
    """Simplified AOE effect configuration.

    This provides a simpler interface for common AOE patterns compared to full handlers.
    AOEs apply resource deltas to entities within range, optionally filtered by alignment.
    """

    range: int = Field(ge=0, description="Radius of the AOE effect")
    resource_deltas: dict[str, int] = Field(
        default_factory=dict,
        description="Resource changes to apply to affected entities",
    )
    filters: list[AnyFilter] = Field(
        default_factory=list,
        description="Filters to determine which entities are affected",
    )


class Mutation(Config):
    """Base class for handler mutations."""

    pass


class ResourceDeltaMutation(Mutation):
    """Apply resource deltas to a target entity."""

    mutation_type: Literal["resource_delta"] = "resource_delta"
    target: HandlerTarget = Field(description="Entity to apply deltas to")
    deltas: dict[str, int] = Field(
        default_factory=dict,
        description="Resource changes (positive = gain, negative = lose)",
    )


class ResourceTransferMutation(Mutation):
    """Transfer resources from one entity to another."""

    mutation_type: Literal["resource_transfer"] = "resource_transfer"
    from_target: HandlerTarget = Field(description="Entity to take resources from")
    to_target: HandlerTarget = Field(description="Entity to give resources to")
    resources: dict[str, int] = Field(
        default_factory=dict,
        description="Resources to transfer (amount, -1 = all available)",
    )


class AlignTo(StrEnum):
    """Alignment target options for AlignmentMutation."""

    ACTOR_COLLECTIVE = auto()  # align to actor's collective
    NONE = auto()  # remove alignment


class AlignmentMutation(Mutation):
    """Update the collective alignment of a target."""

    mutation_type: Literal["alignment"] = "alignment"
    target: Literal["target"] = Field(
        default="target",
        description="Entity to align (only 'target' supported)",
    )
    align_to: AlignTo = Field(
        description="What to align the target to",
    )


class FreezeMutation(Mutation):
    """Freeze an entity for a duration."""

    mutation_type: Literal["freeze"] = "freeze"
    target: AlignmentTarget = Field(description="Entity to freeze (actor or target)")
    duration: int = Field(description="Freeze duration in ticks")


class ClearInventoryMutation(Mutation):
    """Clear all resources in a limit group from inventory (set to 0)."""

    mutation_type: Literal["clear_inventory"] = "clear_inventory"
    target: HandlerTarget = Field(description="Entity to clear inventory from")
    limit_name: str = Field(description="Name of the resource limit group to clear (e.g., 'gear')")


class AttackMutation(Mutation):
    """Combat mutation with weapon/armor/defense mechanics.

    Defense calculation:
    - weapon_power = sum(attacker_inventory[item] * weapon_weight)
    - armor_power = sum(target_inventory[item] * armor_weight) + vibe_bonus if vibing
    - damage_bonus = max(weapon_power - armor_power, 0)
    - cost_to_defend = defense_resources + damage_bonus

    If target can defend, defense resources are consumed and attack is blocked.
    Otherwise, on_success mutations are applied.
    """

    mutation_type: Literal["attack"] = "attack"
    defense_resources: dict[str, int] = Field(
        default_factory=dict,
        description="Resources target needs to block the attack",
    )
    armor_resources: dict[str, int] = Field(
        default_factory=dict,
        description="Target resources that reduce damage (resource -> weight)",
    )
    weapon_resources: dict[str, int] = Field(
        default_factory=dict,
        description="Attacker resources that increase damage (resource -> weight)",
    )
    vibe_bonus: dict[str, int] = Field(
        default_factory=dict,
        description="Per-vibe armor bonus when vibing a matching resource",
    )
    on_success: list["AnyMutation"] = Field(
        default_factory=list,
        description="Mutations to apply when attack succeeds",
    )


AnyMutation = Annotated[
    Union[
        Annotated[ResourceDeltaMutation, Tag("resource_delta")],
        Annotated[ResourceTransferMutation, Tag("resource_transfer")],
        Annotated[AlignmentMutation, Tag("alignment")],
        Annotated[FreezeMutation, Tag("freeze")],
        Annotated[ClearInventoryMutation, Tag("clear_inventory")],
        Annotated[AttackMutation, Tag("attack")],
    ],
    Discriminator("mutation_type"),
]

# Update forward references
AttackMutation.model_rebuild()


class Handler(Config):
    """Configuration for a handler on GridObject.

    Used for all three handler types:
      - on_use: Triggered when agent uses/activates this object
      - on_update: Triggered after mutations are applied to this object
      - aoe: Triggered per-tick for objects within radius

    For on_use handlers, the first handler where all filters pass has its mutations applied.
    For on_update and aoe handlers, all handlers where filters pass have their mutations applied.

    The handler name is provided as the dict key when defining handlers on a GridObject.
    """

    filters: list[AnyFilter] = Field(
        default_factory=list,
        description="All filters must pass for handler to trigger",
    )
    mutations: list[AnyMutation] = Field(
        default_factory=list,
        description="Mutations applied when handler triggers",
    )
    radius: int = Field(
        default=0,
        ge=0,
        description="AOE radius (L-infinity/Chebyshev distance). Only used for aoe handlers.",
    )


# ===== Helper Filter Functions =====
# Factory functions for creating common filter configurations


def isAligned() -> AlignmentFilter:
    """Filter: target is aligned to actor (same collective)."""
    return AlignmentFilter(target=AlignmentTarget.TARGET, alignment=AlignmentCondition.SAME_COLLECTIVE)


def hasCollective() -> AlignmentFilter:
    """Filter: target has any collective alignment."""
    return AlignmentFilter(target=AlignmentTarget.TARGET, alignment=AlignmentCondition.ALIGNED)


def isNeutral() -> AlignmentFilter:
    """Filter: target has no alignment (unaligned)."""
    return AlignmentFilter(target=AlignmentTarget.TARGET, alignment=AlignmentCondition.UNALIGNED)


def isNotAligned() -> AlignmentFilter:
    """Filter: target is NOT aligned to actor (unaligned OR different collective)."""
    return AlignmentFilter(target=AlignmentTarget.TARGET, alignment=AlignmentCondition.NOT_SAME_COLLECTIVE)


def isEnemy() -> AlignmentFilter:
    """Filter: target is aligned to a different collective than actor."""
    return AlignmentFilter(target=AlignmentTarget.TARGET, alignment=AlignmentCondition.DIFFERENT_COLLECTIVE)


def ActorHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: actor has at least the specified resources."""
    return ResourceFilter(target=HandlerTarget.ACTOR, resources=resources)


def TargetHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: target has at least the specified resources."""
    return ResourceFilter(target=HandlerTarget.TARGET, resources=resources)


def ActorCollectiveHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: actor's collective has at least the specified resources."""
    return ResourceFilter(target=HandlerTarget.ACTOR_COLLECTIVE, resources=resources)


def TargetCollectiveHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: target's collective has at least the specified resources."""
    return ResourceFilter(target=HandlerTarget.TARGET_COLLECTIVE, resources=resources)


# ===== Helper Mutation Functions =====
# Factory functions for creating common mutation configurations


def Align() -> AlignmentMutation:
    """Mutation: align target to actor's collective."""
    return AlignmentMutation(target="target", align_to=AlignTo.ACTOR_COLLECTIVE)


def RemoveAlignment() -> AlignmentMutation:
    """Mutation: remove target's alignment (set collective to none)."""
    return AlignmentMutation(target="target", align_to=AlignTo.NONE)


def Withdraw(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from target to actor.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(
        from_target=HandlerTarget.TARGET, to_target=HandlerTarget.ACTOR, resources=resources
    )


def Deposit(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from actor to target.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(
        from_target=HandlerTarget.ACTOR, to_target=HandlerTarget.TARGET, resources=resources
    )


def CollectiveDeposit(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from actor to actor's collective.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(
        from_target=HandlerTarget.ACTOR, to_target=HandlerTarget.ACTOR_COLLECTIVE, resources=resources
    )


def CollectiveWithdraw(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from actor's collective to actor.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(
        from_target=HandlerTarget.ACTOR_COLLECTIVE, to_target=HandlerTarget.ACTOR, resources=resources
    )


def UpdateTarget(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to target.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target=HandlerTarget.TARGET, deltas=deltas)


def UpdateActor(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to actor.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target=HandlerTarget.ACTOR, deltas=deltas)


def TargetCollectiveUpdate(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to target's collective.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target=HandlerTarget.TARGET_COLLECTIVE, deltas=deltas)


def ActorCollectiveUpdate(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to actor's collective.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target=HandlerTarget.ACTOR_COLLECTIVE, deltas=deltas)

from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, Union, get_args

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    SerializeAsAny,
    Tag,
    model_validator,
)

from mettagrid.base_config import Config
from mettagrid.config.id_map import IdMap
from mettagrid.config.obs_config import ObsConfig
from mettagrid.config.vibes import VIBES, Vibe
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.simulator import Action

# ===== Python Configuration Models =====

Direction = Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
Directions = list(get_args(Direction))

# Order must match C++ expectations: north, south, west, east
CardinalDirection = Literal["north", "south", "west", "east"]
CardinalDirections = list(get_args(CardinalDirection))


class AgentRewards(Config):
    """Agent reward configuration with separate inventory and stats rewards."""

    # inventory rewards get merged into stats rewards in the C++ environment. The advantage of using inventory rewards
    # is that it's easier for us to assert that these inventory items exist, and thus catch typos.
    inventory: dict[str, float] = Field(default_factory=dict)
    inventory_max: dict[str, float] = Field(default_factory=dict)
    # commons_inventory rewards agents based on the inventory of the commons they belong to
    commons_inventory: dict[str, float] = Field(default_factory=dict)
    commons_inventory_max: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)
    stats_max: dict[str, float] = Field(default_factory=dict)


class ResourceLimitsConfig(Config):
    """Resource limits configuration.

    Supports dynamic limits via modifiers: the effective limit is
    base_limit + sum(modifier_bonus * quantity_held) for each modifier item.

    Example:
        ResourceLimitsConfig(
            resources=["battery"],
            limit=0,  # base limit is 0
            modifiers={"gear": 1}  # each gear adds +1 battery capacity
        )
    """

    limit: int
    resources: list[str]
    modifiers: dict[str, int] = Field(
        default_factory=dict,
        description="Modifiers that increase the limit. Maps item name to bonus per item held.",
    )


class InventoryConfig(Config):
    """Inventory configuration for agents and chests."""

    default_limit: int = Field(default=65535, ge=0, description="Default resource limit")
    limits: dict[str, ResourceLimitsConfig] = Field(
        default_factory=dict,
        description="Resource-specific limits",
    )
    initial: dict[str, int] = Field(default_factory=dict, description="Initial inventory")
    regen_amounts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Vibe-dependent inventory regeneration. Maps vibe name to resource amounts. "
            "Use 'default' for fallback when agent's vibe isn't specified. "
            "Example: {'default': {'energy': 1}, 'weapon': {'energy': 2}}"
        ),
    )

    def get_limit(self, resource_name: str) -> int:
        """Get the resource limit for a given resource name."""
        for limit_config in self.limits.values():
            if resource_name in limit_config.resources:
                return limit_config.limit
        return self.default_limit


class DamageConfig(Config):
    """Damage configuration for agents.

    When an agent's inventory items reach or exceed all threshold values, one random
    resource from the resources map is destroyed (weighted by quantity above minimum)
    and the threshold amounts are subtracted from inventory.
    """

    threshold: dict[str, int] = Field(
        default_factory=dict,
        description="Map of resource names to threshold values. All must be reached to trigger damage.",
    )
    resources: dict[str, int] = Field(
        default_factory=dict,
        description="Map of resources that can be destroyed, with minimum values. "
        "Only resources listed here can be destroyed. Resources at or below minimum are protected.",
    )

    @model_validator(mode="after")
    def _validate_distinct_keys(self) -> "DamageConfig":
        """Ensure that threshold and resources keys don't overlap."""
        threshold_keys = set(self.threshold.keys())
        resources_keys = set(self.resources.keys())
        overlapping_keys = threshold_keys.intersection(resources_keys)

        if overlapping_keys:
            raise ValueError(
                f"Resources cannot appear in both threshold and resources maps. "
                f"Overlapping keys: {sorted(overlapping_keys)}"
            )

        return self


# TODO: this should probably subclass GridObjectConfig
class AgentConfig(Config):
    """Python agent configuration."""

    inventory: InventoryConfig = Field(default_factory=InventoryConfig, description="Inventory configuration")
    rewards: AgentRewards = Field(default_factory=AgentRewards)
    freeze_duration: int = Field(default=10, ge=-1, description="Duration agent remains frozen after certain actions")
    team_id: int = Field(default=0, ge=0, description="Team identifier for grouping agents")
    tags: list[str] = Field(default_factory=lambda: ["agent"], description="Tags for this agent instance")
    commons: Optional[str] = Field(
        default=None,
        description="Name of commons this agent belongs to. Adds 'commons:{name}' tag automatically.",
    )
    diversity_tracked_resources: list[str] = Field(
        default_factory=list,
        description="Resource names that contribute to inventory diversity metrics",
    )
    initial_vibe: int = Field(default=0, ge=0, description="Initial vibe value for this agent instance")
    damage: Optional[DamageConfig] = Field(
        default=None,
        description="Damage config: when all threshold stats are reached, remove one random resource from inventory",
    )
    handlers: list["ActivationHandler"] = Field(
        default_factory=list,
        description="Handlers triggered when another agent moves onto this agent",
    )

    @model_validator(mode="after")
    def _add_commons_tag(self) -> "AgentConfig":
        # Add commons tag if commons is set
        if self.commons:
            commons_tag = f"commons:{self.commons}"
            if commons_tag not in self.tags:
                self.tags = self.tags + [commons_tag]
        return self


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

    Align is triggered by move when the agent's vibe matches the configured vibe.
    It aligns the target's commons to the actor's commons, or sets it to none if set_to_none=True.
    """

    action_handler: str = Field(default="align")
    vibe: str = Field(
        default="",
        description="Vibe name that triggers align on move",
    )
    cost: dict[str, int] = Field(
        default_factory=dict,
        description="Cost to the actor for aligning (resource name -> amount)",
    )
    commons_cost: dict[str, int] = Field(
        default_factory=dict,
        description="Cost deducted from the actor's commons (resource name -> amount)",
    )
    set_to_none: bool = Field(
        default=False,
        description="If true, sets target's commons to none instead of actor's commons (scramble mode)",
    )

    def _actions(self) -> list[Action]:
        # Align doesn't create standalone actions - it's triggered by move
        return []


# ===== Activation Handler System =====
# Data-driven system for configuring what happens when an agent activates a target object


ActivationTarget = Literal["actor", "target", "actor_commons", "target_commons"]


class ActivationFilter(Config):
    """Base class for activation filters. All filters in a handler must pass."""

    target: Literal["actor", "target", "actor_commons", "target_commons"] = Field(
        default="actor",
        description="Entity to check the filter against",
    )


class VibeFilter(ActivationFilter):
    """Filter that checks if the target entity has a specific vibe."""

    filter_type: Literal["vibe"] = "vibe"
    vibe: str = Field(description="Vibe name that must match")


class ResourceFilter(ActivationFilter):
    """Filter that checks if the target entity has required resources."""

    filter_type: Literal["resource"] = "resource"
    resources: dict[str, int] = Field(
        default_factory=dict,
        description="Minimum resource amounts required",
    )


class AlignmentFilter(ActivationFilter):
    """Filter that checks the alignment status of a target.

    Can check if target is aligned/unaligned, or if it's aligned to
    the same/different commons as the actor.
    """

    filter_type: Literal["alignment"] = "alignment"
    target: Literal["actor", "target"] = Field(
        default="target",
        description="Entity to check the filter against (only actor/target for alignment)",
    )
    alignment: Literal["aligned", "unaligned", "same_commons", "different_commons", "not_same_commons"] = Field(
        description=(
            "Alignment condition to check: "
            "'aligned' = target has any commons, "
            "'unaligned' = target has no commons, "
            "'same_commons' = target has same commons as actor, "
            "'different_commons' = target has different commons than actor (but is aligned), "
            "'not_same_commons' = target is not aligned to actor (unaligned OR different_commons)"
        ),
    )


AnyActivationFilter = Annotated[
    Union[
        Annotated[VibeFilter, Tag("vibe")],
        Annotated[ResourceFilter, Tag("resource")],
        Annotated[AlignmentFilter, Tag("alignment")],
    ],
    Discriminator("filter_type"),
]


class ActivationMutation(Config):
    """Base class for activation mutations."""

    pass


class ResourceDeltaMutation(ActivationMutation):
    """Apply resource deltas to a target entity."""

    mutation_type: Literal["resource_delta"] = "resource_delta"
    target: ActivationTarget = Field(description="Entity to apply deltas to")
    deltas: dict[str, int] = Field(
        default_factory=dict,
        description="Resource changes (positive = gain, negative = lose)",
    )


class ResourceTransferMutation(ActivationMutation):
    """Transfer resources from one entity to another."""

    mutation_type: Literal["resource_transfer"] = "resource_transfer"
    from_target: ActivationTarget = Field(description="Entity to take resources from")
    to_target: ActivationTarget = Field(description="Entity to give resources to")
    resources: dict[str, int] = Field(
        default_factory=dict,
        description="Resources to transfer (amount, -1 = all available)",
    )


class AlignmentMutation(ActivationMutation):
    """Update the commons alignment of a target."""

    mutation_type: Literal["alignment"] = "alignment"
    target: Literal["target"] = Field(
        default="target",
        description="Entity to align (only 'target' supported)",
    )
    align_to: Literal["actor_commons", "none"] = Field(
        description="What to align the target to",
    )


class FreezeMutation(ActivationMutation):
    """Freeze an entity for a duration."""

    mutation_type: Literal["freeze"] = "freeze"
    target: Literal["actor", "target"] = Field(description="Entity to freeze")
    duration: int = Field(description="Freeze duration in ticks")


class ClearInventoryMutation(ActivationMutation):
    """Clear all resources in a limit group from inventory (set to 0)."""

    mutation_type: Literal["clear_inventory"] = "clear_inventory"
    target: ActivationTarget = Field(description="Entity to clear inventory from")
    limit_name: str = Field(description="Name of the resource limit group to clear (e.g., 'gear')")


class AttackMutation(ActivationMutation):
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
    on_success: list["AnyActivationMutation"] = Field(
        default_factory=list,
        description="Mutations to apply when attack succeeds",
    )


AnyActivationMutation = Annotated[
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

# Update forward reference for AttackMutation.on_success
AttackMutation.model_rebuild()


class ActivationHandler(Config):
    """Configuration for a target activation handler.

    When an agent moves onto a target, handlers are checked in registration order.
    The first handler where all filters pass has its mutations applied.
    """

    name: str = Field(description="Handler name for debugging/stats")
    filters: list[AnyActivationFilter] = Field(
        default_factory=list,
        description="All filters must pass for handler to trigger",
    )
    mutations: list[AnyActivationMutation] = Field(
        default_factory=list,
        description="Mutations applied when handler triggers",
    )


# ===== Helper Filter Functions =====
# Factory functions for creating common filter configurations


def isAligned() -> AlignmentFilter:
    """Filter: target is aligned to actor (same commons)."""
    return AlignmentFilter(target="target", alignment="same_commons")


def hasCommons() -> AlignmentFilter:
    """Filter: target has any commons alignment."""
    return AlignmentFilter(target="target", alignment="aligned")


def isNeutral() -> AlignmentFilter:
    """Filter: target has no alignment (unaligned)."""
    return AlignmentFilter(target="target", alignment="unaligned")


def isNotAligned() -> AlignmentFilter:
    """Filter: target is NOT aligned to actor (unaligned OR different commons)."""
    return AlignmentFilter(target="target", alignment="not_same_commons")


def isEnemy() -> AlignmentFilter:
    """Filter: target is aligned to a different commons than actor."""
    return AlignmentFilter(target="target", alignment="different_commons")


def ActorHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: actor has at least the specified resources."""
    return ResourceFilter(target="actor", resources=resources)


def TargetHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: target has at least the specified resources."""
    return ResourceFilter(target="target", resources=resources)


def ActorCommonsHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: actor's commons has at least the specified resources."""
    return ResourceFilter(target="actor_commons", resources=resources)


def TargetCommonsHas(resources: dict[str, int]) -> ResourceFilter:
    """Filter: target's commons has at least the specified resources."""
    return ResourceFilter(target="target_commons", resources=resources)


# ===== Helper Mutation Functions =====
# Factory functions for creating common mutation configurations


def Align() -> AlignmentMutation:
    """Mutation: align target to actor's commons."""
    return AlignmentMutation(target="target", align_to="actor_commons")


def RemoveAlignment() -> AlignmentMutation:
    """Mutation: remove target's alignment (set commons to none)."""
    return AlignmentMutation(target="target", align_to="none")


def Pickup(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from target to actor.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(from_target="target", to_target="actor", resources=resources)


def Drop(resources: dict[str, int]) -> ResourceTransferMutation:
    """Mutation: transfer resources from actor to target.

    Args:
        resources: Map of resource name to amount. Use -1 for "all available".
    """
    return ResourceTransferMutation(from_target="actor", to_target="target", resources=resources)


def UpdateTarget(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to target.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target="target", deltas=deltas)


def UpdateActor(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to actor.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target="actor", deltas=deltas)


def UpdateTargetCommons(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to target's commons.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target="target_commons", deltas=deltas)


def UpdateActorCommons(deltas: dict[str, int]) -> ResourceDeltaMutation:
    """Mutation: apply resource deltas to actor's commons.

    Args:
        deltas: Map of resource name to delta (positive = gain, negative = lose).
    """
    return ResourceDeltaMutation(target="actor_commons", deltas=deltas)


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: NoopActionConfig = Field(default_factory=lambda: NoopActionConfig())
    move: MoveActionConfig = Field(default_factory=lambda: MoveActionConfig())
    attack: AttackActionConfig = Field(default_factory=lambda: AttackActionConfig(enabled=False))
    transfer: TransferActionConfig = Field(default_factory=lambda: TransferActionConfig(enabled=False))
    align: AlignActionConfig | None = Field(default=None)
    scramble: AlignActionConfig | None = Field(default=None)
    change_vibe: ChangeVibeActionConfig = Field(default_factory=lambda: ChangeVibeActionConfig())

    def actions(self) -> list[Action]:
        action_configs = [self.noop, self.move, self.attack, self.transfer, self.change_vibe]
        if self.align:
            action_configs.append(self.align)
        if self.scramble:
            action_configs.append(self.scramble)
        return sum([action.actions() for action in action_configs], [])


class GlobalObsConfig(Config):
    """Global observation configuration."""

    episode_completion_pct: bool = Field(default=True)

    # Controls whether the last_action global token is included
    last_action: bool = Field(default=True)

    last_reward: bool = Field(default=True)

    # Compass token that points toward the assembler/hub center
    compass: bool = Field(default=False)

    # Goal tokens that indicate rewarding resources
    goal_obs: bool = Field(default=False)


class AOEEffectConfig(Config):
    """Configuration for Area of Effect (AOE) resource effects.

    When attached to a grid object, objects with inventory within range receive the resource_deltas each tick.

    Target filtering:
    - target_tags: If set, only objects with at least one matching tag are affected.
                   If None or empty, all HasInventory objects are affected.
                   Agents are always checked every tick (they move).
                   Static objects are registered/unregistered with the AOE for efficiency.

    Commons filtering:
    - members_only: If True, effect only applies to objects with the same commons as the source object
    - ignore_members: If True, effect is skipped for objects with the same commons as the source object
    """

    range: int = Field(default=1, ge=0, description="Radius of effect (Manhattan distance)")
    resource_deltas: dict[str, int] = Field(
        default_factory=dict,
        description="Resource changes per tick for objects in range. Positive = gain, negative = lose.",
    )
    target_tags: Optional[list[str]] = Field(
        default=None,
        description="If set, only objects with at least one matching tag are affected. "
        "If None, all HasInventory objects are affected.",
    )
    members_only: bool = Field(
        default=False,
        description="If True, effect only applies to objects with the same commons as the source object",
    )
    ignore_members: bool = Field(
        default=False,
        description="If True, effect is skipped for objects with the same commons as the source object",
    )


class GridObjectConfig(Config):
    """Base configuration for all grid objects.

    Python uses only names. Numeric type_ids are an internal C++ detail and are
    computed during Python→C++ conversion; they are never part of Python config
    or observations.
    """

    name: str = Field(description="Canonical type_name (human-readable)")
    map_name: str = Field(default="", description="Stable key used by maps to select this config")
    render_name: str = Field(default="", description="Stable display-class identifier for theming")
    render_symbol: str = Field(default="❓", description="Symbol used for rendering (e.g., emoji)")
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")
    vibe: int = Field(default=0, ge=0, le=255, description="Vibe value for this object instance")
    commons: Optional[str] = Field(
        default=None,
        description="Name of commons this object belongs to. Adds 'commons:{name}' tag automatically.",
    )
    aoes: list[AOEEffectConfig] = Field(
        default_factory=list,
        description="List of AOE effects this object emits to agents within range each tick",
    )
    handlers: list[ActivationHandler] = Field(
        default_factory=list,
        description="Handlers triggered when an agent moves onto this object",
    )

    @model_validator(mode="after")
    def _defaults_from_name(self) -> "GridObjectConfig":
        if not self.map_name:
            self.map_name = self.name
        if not self.render_name:
            self.render_name = self.name
        # If no tags, inject a default kind tag so the object is visible in observations
        if not self.tags:
            self.tags = [self.render_name]
        # Add commons tag if commons is set
        if self.commons:
            commons_tag = f"commons:{self.commons}"
            if commons_tag not in self.tags:
                self.tags = self.tags + [commons_tag]
        return self


class WallConfig(GridObjectConfig):
    """Python wall/block configuration."""

    # This is used to discriminate between different GridObjectConfig subclasses in Pydantic.
    # See AnyGridObjectConfig.
    # Please don't use this for anything game related.
    pydantic_type: Literal["wall"] = "wall"
    name: str = Field(default="wall")


class ProtocolConfig(Config):
    # Note that `vibes` implicitly also sets a minimum number of agents. So min_agents is useful
    # when you want to set a minimum that's higher than the number of vibes.
    min_agents: int = Field(default=0, ge=0, description="Number of agents required to use this protocol")
    vibes: list[str] = Field(default_factory=list)
    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    cooldown: int = Field(ge=0, default=0)


class AssemblerConfig(GridObjectConfig):
    """Python assembler configuration."""

    # This is used to discriminate between different GridObjectConfig subclasses in Pydantic.
    # See AnyGridObjectConfig.
    # Please don't use this for anything game related.
    pydantic_type: Literal["assembler"] = "assembler"
    # No default name -- we want to make sure that meaningful names are provided.
    protocols: list[ProtocolConfig] = Field(
        default_factory=list,
        description="Protocols in reverse order of priority.",
    )
    allow_partial_usage: bool = Field(
        default=False,
        description=(
            "Allow assembler to be used during cooldown with scaled resource requirements/outputs. "
            "This makes less sense if the assembler has multiple protocols."
        ),
    )
    max_uses: int = Field(default=0, ge=0, description="Maximum number of uses (0 = unlimited)")
    clip_immune: bool = Field(
        default=False, description="If true, this assembler cannot be clipped by the Clipper system"
    )
    start_clipped: bool = Field(
        default=False, description="If true, this assembler starts in a clipped state at the beginning of the game"
    )
    chest_search_distance: int = Field(
        default=0,
        ge=0,
        description="Distance within which assembler can use inventories from chests",
    )


class ChestConfig(GridObjectConfig):
    """Python chest configuration for multi-resource chests."""

    # This is used to discriminate between different GridObjectConfig subclasses in Pydantic.
    # See AnyGridObjectConfig.
    # Please don't use this for anything game related.
    pydantic_type: Literal["chest"] = "chest"
    name: str = Field(default="chest")

    # Vibe-based transfers: vibe -> resource -> delta
    vibe_transfers: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Map from vibe to resource deltas. "
            "E.g., {'carbon': {'carbon': 10, 'energy': -5}} deposits 10 carbon and withdraws 5 energy when "
            "showing carbon vibe"
        ),
    )

    inventory: InventoryConfig = Field(default_factory=InventoryConfig, description="Inventory configuration")


class CommonsChestConfig(ChestConfig):
    """Python commons chest configuration - like chest but uses commons inventory."""

    pydantic_type: Literal["commons_chest"] = "commons_chest"
    name: str = Field(default="commons_chest")


class ClipperConfig(Config):
    """
    Global clipper that probabilistically clips assemblers each tick.

    The clipper system uses a spatial diffusion process where clipping spreads
    based on distance from already-clipped buildings. The length_scale parameter
    controls the exponential decay: weight ~= exp(-distance / length_scale).
    """

    unclipping_protocols: list[ProtocolConfig] = Field(default_factory=list)
    length_scale: int = Field(
        default=0,
        ge=0,
        description="Controls spatial spread rate: weight ~= exp(-distance / length_scale). "
        "If <= 0, automatically calculated at runtime based on the sparsity of the grid.",
    )
    scaled_cutoff_distance: int = Field(
        default=3,
        ge=1,
        description="Maximum distance in units of length_scale for infection weight calculations.",
    )
    clip_period: int = Field(
        default=0, ge=0, description="Approximate timesteps between clipping events (0 = disabled)"
    )


class CommonsConfig(Config):
    """
    Configuration for a shared inventory (Commons).

    Commons provides a shared inventory that multiple grid objects can access.
    Objects are associated with a commons via tags of the form "commons:{name}".
    Grid objects can specify commons="name" in their config to automatically add
    this tag.
    """

    name: str = Field(description="Unique name for this commons")
    inventory: InventoryConfig = Field(default_factory=InventoryConfig, description="Inventory configuration")


AnyGridObjectConfig = SerializeAsAny[
    Annotated[
        Union[
            Annotated[WallConfig, Tag("wall")],
            Annotated[AssemblerConfig, Tag("assembler")],
            Annotated[ChestConfig, Tag("chest")],
            Annotated[CommonsChestConfig, Tag("commons_chest")],
        ],
        Discriminator("pydantic_type"),
    ]
]


class GameConfig(Config):
    """Python game configuration.

    Note: Type IDs are automatically assigned during validation when the GameConfig
    is constructed. If you need to add objects after construction, create a new
    GameConfig instance rather than modifying the objects dict post-construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _resolved_type_ids: bool = PrivateAttr(default=False)

    resource_names: list[str] = Field(
        default=[
            "ore_red",
            "ore_blue",
            "ore_green",
            "battery_red",
            "battery_blue",
            "battery_green",
            "heart",
            "armor",
            "laser",
            "blueprint",
        ]
    )
    vibe_names: list[str] = Field(default_factory=list)
    num_agents: int = Field(ge=1, default=24)
    # max_steps = zero means "no limit"
    max_steps: int = Field(ge=0, default=10000)
    # default is that we terminate / use "done" vs truncation
    episode_truncates: bool = Field(default=False)
    obs: ObsConfig = Field(default_factory=ObsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    actions: ActionsConfig = Field(default_factory=lambda: ActionsConfig())
    global_obs: GlobalObsConfig = Field(default_factory=GlobalObsConfig)
    objects: dict[str, AnyGridObjectConfig] = Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place  where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None

    # Inventory regeneration interval (global check timing)
    inventory_regen_interval: int = Field(
        default=0, ge=0, description="Interval in timesteps between regenerations (0 = disabled)"
    )

    # Global clipper system
    clipper: Optional[ClipperConfig] = Field(default=None, description="Global clipper configuration")

    # Commons - shared inventories that grid objects can belong to
    commons: list[CommonsConfig] = Field(
        default_factory=list,
        description="List of commons (shared inventories) that grid objects can belong to",
    )

    # Map builder configuration - accepts any MapBuilder config
    map_builder: AnyMapBuilderConfig = Field(default_factory=lambda: RandomMapBuilder.Config(agents=24))

    # Note that if this is False, agents won't be able to see how to unclip assemblers.
    protocol_details_obs: bool = Field(
        default=True, description="Objects show their protocol inputs and outputs when observed"
    )

    reward_estimates: Optional[dict[str, float]] = Field(default=None)

    @model_validator(mode="after")
    def _compute_feature_ids(self) -> "GameConfig":
        self.vibe_names = [vibe.name for vibe in self.actions.change_vibe.vibes]
        return self

    def id_map(self) -> "IdMap":
        """Get the observation feature ID map for this configuration."""
        return IdMap(self)


class MettaGridConfig(Config):
    """Environment configuration."""

    label: str = Field(default="mettagrid")
    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)

    def with_ascii_map(self, map_data: list[list[str]], char_to_map_name: dict[str, str]) -> "MettaGridConfig":
        self.game.map_builder = AsciiMapBuilder.Config(
            map_data=map_data,
            char_to_map_name=char_to_map_name,
        )
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "MettaGridConfig":
        """Create an empty room environment configuration."""
        map_builder = RandomMapBuilder.Config(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(move=MoveActionConfig(), change_vibe=ChangeVibeActionConfig())
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(render_symbol="⬛")
        return MettaGridConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

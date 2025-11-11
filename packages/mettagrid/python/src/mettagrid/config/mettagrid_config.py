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
from mettagrid.map_builder.random import RandomMapBuilder
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
    stats: dict[str, float] = Field(default_factory=dict)
    stats_max: dict[str, float] = Field(default_factory=dict)


class AgentConfig(Config):
    """Python agent configuration."""

    default_resource_limit: int = Field(default=255, ge=0)
    resource_limits: dict[str | tuple[str, ...], int] = Field(
        default_factory=dict,
        description="Resource limits - keys can be single resource names or tuples of names for shared limits",
    )
    freeze_duration: int = Field(default=10, ge=-1)
    rewards: AgentRewards = Field(default_factory=AgentRewards)
    action_failure_penalty: float = Field(default=0, ge=0)
    initial_inventory: dict[str, int] = Field(default_factory=dict)
    team_id: int = Field(default=0, ge=0, description="Team identifier for grouping agents")
    tags: list[str] = Field(default_factory=list, description="Tags for this agent instance")
    soul_bound_resources: list[str] = Field(
        default_factory=list, description="Resources that cannot be stolen during attacks"
    )
    shareable_resources: list[str] = Field(
        default_factory=list, description="Resources that will be shared when we use another agent"
    )
    inventory_regen_amounts: dict[str, int] = Field(
        default_factory=dict, description="Resources to regenerate and their amounts per regeneration interval"
    )
    diversity_tracked_resources: list[str] = Field(
        default_factory=list,
        description="Resource names that contribute to inventory diversity metrics",
    )
    initial_vibe: int = Field(default=0, ge=0, description="Initial vibe value for this agent instance")


class ActionConfig(Config):
    """Python action configuration."""

    action_handler: str
    enabled: bool = Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: dict[str, int] = Field(default_factory=dict)
    consumed_resources: dict[str, float] = Field(default_factory=dict)

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
    number_of_vibes: int = Field(default=0, ge=0, le=255)

    def _actions(self) -> list[Action]:
        return [self.ChangeVibe(vibe) for vibe in VIBES[: self.number_of_vibes]]

    def ChangeVibe(self, vibe: Vibe) -> Action:
        return Action(name=f"change_vibe_{vibe.name}")


class AttackActionConfig(ActionConfig):
    """Python attack action configuration."""

    action_handler: str = Field(default="attack")
    defense_resources: dict[str, int] = Field(default_factory=dict)
    target_locations: list[Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]] = Field(
        default_factory=lambda: ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    )

    def _actions(self) -> list[Action]:
        return [self.Attack(location) for location in self.target_locations]

    def Attack(self, location: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]) -> Action:
        return Action(name=f"attack_{location}")


class ResourceModActionConfig(ActionConfig):
    """Resource mod action configuration."""

    action_handler: str = Field(default="resource_mod")
    modifies: dict[str, float] = Field(default_factory=dict)
    agent_radius: int = Field(default=0, ge=0, le=255)
    scales: bool = Field(default=False)

    def _actions(self) -> list[Action]:
        return [self.ResourceMod()]

    def ResourceMod(self) -> Action:
        return Action(name="resource_mod")


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: NoopActionConfig = Field(default_factory=lambda: NoopActionConfig())
    move: MoveActionConfig = Field(default_factory=lambda: MoveActionConfig())
    attack: AttackActionConfig = Field(default_factory=lambda: AttackActionConfig(enabled=False))
    change_vibe: ChangeVibeActionConfig = Field(default_factory=lambda: ChangeVibeActionConfig())
    resource_mod: ResourceModActionConfig = Field(default_factory=lambda: ResourceModActionConfig(enabled=False))

    def actions(self) -> list[Action]:
        return sum(
            [action.actions() for action in [self.noop, self.move, self.attack, self.change_vibe, self.resource_mod]],
            [],
        )


class GlobalObsConfig(Config):
    """Global observation configuration."""

    episode_completion_pct: bool = Field(default=True)

    # Controls whether the last_action global token is included
    last_action: bool = Field(default=True)

    last_reward: bool = Field(default=True)

    # Controls whether visitation counts are included in observations
    visitation_counts: bool = Field(default=False)

    # Compass token that points toward the assembler/hub center
    compass: bool = Field(default=False)


class GridObjectConfig(Config):
    """Base configuration for all grid objects.

    Python uses only names. Numeric type_ids are an internal C++ detail and are
    computed during Python→C++ conversion; they are never part of Python config
    or observations.
    """

    name: str = Field(description="Canonical type_name (human-readable)")
    map_name: str = Field(default="", description="Stable key used by maps to select this config")
    render_name: str = Field(default="", description="Stable display-class identifier for theming")
    map_char: str = Field(default="?", description="Character used in ASCII maps")
    render_symbol: str = Field(default="❓", description="Symbol used for rendering (e.g., emoji)")
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")
    vibe: int = Field(default=0, ge=0, le=255, description="Vibe value for this object instance")

    @model_validator(mode="after")
    def _defaults_from_name(self) -> "GridObjectConfig":
        if not self.map_name:
            self.map_name = self.name
        if not self.render_name:
            self.render_name = self.name
        # If no tags, inject a default kind tag so the object is visible in observations
        if not self.tags:
            self.tags = [self.render_name]
        return self


class WallConfig(GridObjectConfig):
    """Python wall/block configuration."""

    # This is used to discriminate between different GridObjectConfig subclasses in Pydantic.
    # See AnyGridObjectConfig.
    # Please don't use this for anything game related.
    pydantic_type: Literal["wall"] = "wall"
    name: str = Field(default="wall")
    swappable: bool = Field(default=False)


class ProtocolConfig(Config):
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
    exhaustion: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Exhaustion rate - cooldown multiplier grows by (1 + exhaustion) after each use (0 = no exhaustion)"
        ),
    )
    clip_immune: bool = Field(
        default=False, description="If true, this assembler cannot be clipped by the Clipper system"
    )
    start_clipped: bool = Field(
        default=False, description="If true, this assembler starts in a clipped state at the beginning of the game"
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

    # Initial inventory for each resource
    initial_inventory: dict[str, int] = Field(
        default_factory=dict, description="Initial inventory for each resource type"
    )

    # Resource limits for the chest's inventory
    resource_limits: dict[str, int] = Field(
        default_factory=dict, description="Maximum amount per resource (uses inventory system's built-in limits)"
    )


class ClipperConfig(Config):
    """
    Global clipper that probabilistically clips assemblers each tick.

    The clipper system uses a spatial diffusion process where clipping spreads
    based on distance from already-clipped buildings. The length_scale parameter
    controls the exponential decay: weight = exp(-distance / length_scale).

    If length_scale is <= 0 (default 0.0), it will be automatically calculated
    at runtime in C++ using percolation based on the actual grid size and
    number of buildings placed. Set length_scale > 0 to use a manual value instead.

    If cutoff_distance is <= 0 (default 0.0), it will be automatically set to
    3 * length_scale at runtime. At this distance, exp(-3) ≈ 0.05, making weights
    negligible. Set cutoff_distance > 0 to use a manual cutoff.
    """

    unclipping_protocols: list[ProtocolConfig] = Field(default_factory=list)
    length_scale: float = Field(
        default=0.0,
        description="Controls spatial spread rate: weight = exp(-distance / length_scale). "
        "If <= 0, automatically calculated using percolation at runtime.",
    )
    cutoff_distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum distance for infection weight calculations. "
        "If <= 0, automatically set to 3 * length_scale at runtime.",
    )
    clip_rate: float = Field(default=0.0, ge=0.0, le=1.0)


AnyGridObjectConfig = SerializeAsAny[
    Annotated[
        Union[
            Annotated[WallConfig, Tag("wall")],
            Annotated[AssemblerConfig, Tag("assembler")],
            Annotated[ChestConfig, Tag("chest")],
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
    max_steps: int = Field(ge=0, default=1000)
    # default is that we terminate / use "done" vs truncation
    episode_truncates: bool = Field(default=False)
    obs: ObsConfig = Field(default_factory=ObsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    actions: ActionsConfig = Field(default_factory=lambda: ActionsConfig())
    global_obs: GlobalObsConfig = Field(default_factory=GlobalObsConfig)
    objects: dict[str, AnyGridObjectConfig] = Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None

    resource_loss_prob: float = Field(default=0.0, description="Probability of resource loss per step")

    # Inventory regeneration interval (global check timing)
    inventory_regen_interval: int = Field(
        default=0, ge=0, description="Interval in timesteps between regenerations (0 = disabled)"
    )

    # Global clipper system
    clipper: Optional[ClipperConfig] = Field(default=None, description="Global clipper configuration")

    # Map builder configuration - accepts any MapBuilder config
    map_builder: AnyMapBuilderConfig = Field(default_factory=lambda: RandomMapBuilder.Config(agents=24))

    protocol_details_obs: bool = Field(
        default=False, description="Objects show their protocol inputs and outputs when observed"
    )

    reward_estimates: Optional[dict[str, float]] = Field(default=None)

    @model_validator(mode="after")
    def _compute_feature_ids(self) -> "GameConfig":
        self._populate_vibe_names()
        # Note that this validation only runs once by default, so later changes by the user can cause this to no
        # longer be true.
        if not self.actions.change_vibe.number_of_vibes == len(self.vibe_names):
            raise ValueError("number_of_vibes must match the number of vibe names")
        return self

    def _populate_vibe_names(self) -> None:
        """Populate vibe_names from change_vibe action config if not already set."""
        if not self.vibe_names:
            num_vibes = self.actions.change_vibe.number_of_vibes
            self.vibe_names = [vibe.name for vibe in VIBES[:num_vibes]]

    def id_map(self) -> "IdMap":
        """Get the observation feature ID map for this configuration."""
        return IdMap(self)


class EnvSupervisorConfig(Config):
    """Environment supervisor configuration."""

    policy: Optional[str] = Field(default=None)
    policy_data_path: Optional[str] = Field(default=None)


class MettaGridConfig(Config):
    """Environment configuration."""

    label: str = Field(default="mettagrid")
    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)

    def with_ascii_map(self, map_data: list[list[str]]) -> "MettaGridConfig":
        self.game.map_builder = AsciiMapBuilder.Config(
            map_data=map_data,
            char_to_map_name={o.map_char: o.map_name for o in self.game.objects.values()},
        )
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "MettaGridConfig":
        """Create an empty room environment configuration."""
        map_builder = RandomMapBuilder.Config(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(
            move=MoveActionConfig(),
        )
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(map_char="#", render_symbol="⬛", swappable=False)
        return MettaGridConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

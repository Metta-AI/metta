from __future__ import annotations

import abc
import typing

import pydantic

import mettagrid.base_config
from . import id_map as id_map_module
from . import obs_config as obs_config_module
from . import vibes as vibes_module
import mettagrid.map_builder.ascii
import mettagrid.map_builder.map_builder
import mettagrid.map_builder.random
import mettagrid.simulator

# ===== Python Configuration Models =====

Direction = typing.Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
Directions = list(typing.get_args(Direction))

# Order must match C++ expectations: north, south, west, east
CardinalDirection = typing.Literal["north", "south", "west", "east"]
CardinalDirections = list(typing.get_args(CardinalDirection))


class AgentRewards(mettagrid.base_config.Config):
    """Agent reward configuration with separate inventory and stats rewards."""

    # inventory rewards get merged into stats rewards in the C++ environment. The advantage of using inventory rewards
    # is that it's easier for us to assert that these inventory items exist, and thus catch typos.
    inventory: dict[str, float] = pydantic.Field(default_factory=dict)
    inventory_max: dict[str, float] = pydantic.Field(default_factory=dict)
    stats: dict[str, float] = pydantic.Field(default_factory=dict)
    stats_max: dict[str, float] = pydantic.Field(default_factory=dict)


class AgentConfig(mettagrid.base_config.Config):
    """Python agent configuration."""

    default_resource_limit: int = pydantic.Field(default=255, ge=0)
    resource_limits: dict[str | tuple[str, ...], int] = pydantic.Field(
        default_factory=dict,
        description="Resource limits - keys can be single resource names or tuples of names for shared limits",
    )
    freeze_duration: int = pydantic.Field(default=10, ge=-1)
    rewards: AgentRewards = pydantic.Field(default_factory=AgentRewards)
    action_failure_penalty: float = pydantic.Field(default=0, ge=0)
    initial_inventory: dict[str, int] = pydantic.Field(default_factory=dict)
    team_id: int = pydantic.Field(default=0, ge=0, description="Team identifier for grouping agents")
    tags: list[str] = pydantic.Field(default_factory=list, description="Tags for this agent instance")
    soul_bound_resources: list[str] = pydantic.Field(
        default_factory=list, description="Resources that cannot be stolen during attacks"
    )
    shareable_resources: list[str] = pydantic.Field(
        default_factory=list, description="Resources that will be shared when we use another agent"
    )
    inventory_regen_amounts: dict[str, int] = pydantic.Field(
        default_factory=dict, description="Resources to regenerate and their amounts per regeneration interval"
    )
    diversity_tracked_resources: list[str] = pydantic.Field(
        default_factory=list,
        description="Resource names that contribute to inventory diversity metrics",
    )
    initial_vibe: int = pydantic.Field(default=0, ge=0, description="Initial vibe value for this agent instance")


class ActionConfig(mettagrid.base_config.Config):
    """Python action configuration."""

    action_handler: str
    enabled: bool = pydantic.Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: dict[str, int] = pydantic.Field(default_factory=dict)
    consumed_resources: dict[str, float] = pydantic.Field(default_factory=dict)

    def actions(self) -> list[mettagrid.simulator.Action]:
        if self.enabled:
            return self._actions()
        return []

    @abc.abstractmethod
    def _actions(self) -> list[mettagrid.simulator.Action]: ...


class NoopActionConfig(ActionConfig):
    """Noop action configuration."""

    action_handler: str = pydantic.Field(default="noop")

    def _actions(self) -> list[mettagrid.simulator.Action]:
        return [self.Noop()]

    def Noop(self) -> mettagrid.simulator.Action:
        return mettagrid.simulator.Action(name="noop")


class MoveActionConfig(ActionConfig):
    """Move action configuration."""

    action_handler: str = pydantic.Field(default="move")
    allowed_directions: list[Direction] = pydantic.Field(default_factory=lambda: CardinalDirections)

    def _actions(self) -> list[mettagrid.simulator.Action]:
        return [self.Move(direction) for direction in self.allowed_directions]

    def Move(self, direction: Direction) -> mettagrid.simulator.Action:
        return mettagrid.simulator.Action(name=f"move_{direction}")


class ChangeVibeActionConfig(ActionConfig):
    """Change vibe action configuration."""

    action_handler: str = pydantic.Field(default="change_vibe")
    number_of_vibes: int = pydantic.Field(default=0, ge=0, le=255)

    def _actions(self) -> list[mettagrid.simulator.Action]:
        return [self.ChangeVibe(vibe) for vibe in vibes_module.VIBES[: self.number_of_vibes]]

    def ChangeVibe(self, vibe: vibes_module.Vibe) -> mettagrid.simulator.Action:
        return mettagrid.simulator.Action(name=f"change_vibe_{vibe.name}")


class AttackActionConfig(ActionConfig):
    """Python attack action configuration."""

    action_handler: str = pydantic.Field(default="attack")
    defense_resources: dict[str, int] = pydantic.Field(default_factory=dict)
    target_locations: list[typing.Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]] = pydantic.Field(
        default_factory=lambda: ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    )

    def _actions(self) -> list[mettagrid.simulator.Action]:
        return [self.Attack(location) for location in self.target_locations]

    def Attack(
        self, location: typing.Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ) -> mettagrid.simulator.Action:
        return mettagrid.simulator.Action(name=f"attack_{location}")


class ResourceModActionConfig(ActionConfig):
    """Resource mod action configuration."""

    action_handler: str = pydantic.Field(default="resource_mod")
    modifies: dict[str, float] = pydantic.Field(default_factory=dict)
    agent_radius: int = pydantic.Field(default=0, ge=0, le=255)
    scales: bool = pydantic.Field(default=False)

    def _actions(self) -> list[mettagrid.simulator.Action]:
        return [self.ResourceMod()]

    def ResourceMod(self) -> mettagrid.simulator.Action:
        return mettagrid.simulator.Action(name="resource_mod")


class ActionsConfig(mettagrid.base_config.Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: NoopActionConfig = pydantic.Field(default_factory=lambda: NoopActionConfig())
    move: MoveActionConfig = pydantic.Field(default_factory=lambda: MoveActionConfig())
    attack: AttackActionConfig = pydantic.Field(default_factory=lambda: AttackActionConfig(enabled=False))
    change_vibe: ChangeVibeActionConfig = pydantic.Field(default_factory=lambda: ChangeVibeActionConfig())
    resource_mod: ResourceModActionConfig = pydantic.Field(
        default_factory=lambda: ResourceModActionConfig(enabled=False)
    )

    def actions(self) -> list[mettagrid.simulator.Action]:
        return sum(
            [action.actions() for action in [self.noop, self.move, self.attack, self.change_vibe, self.resource_mod]],
            [],
        )


class GlobalObsConfig(mettagrid.base_config.Config):
    """Global observation configuration."""

    episode_completion_pct: bool = pydantic.Field(default=True)

    # Controls whether the last_action global token is included
    last_action: bool = pydantic.Field(default=True)

    last_reward: bool = pydantic.Field(default=True)

    # Controls whether visitation counts are included in observations
    visitation_counts: bool = pydantic.Field(default=False)


class GridObjectConfig(mettagrid.base_config.Config):
    """Base configuration for all grid objects.

    Type IDs are automatically assigned if not explicitly provided. Auto-assignment
    is deterministic (sorted by object name) and fills gaps in the 1-255 range.
    Type ID 0 is reserved for agents.

    Explicit type_ids are optional and primarily useful for:
    - Ensuring stable IDs across config changes
    - Matching specific C++ expectations
    - Debugging and development

    In most cases, omit type_id and let the system auto-assign.
    """

    name: str = pydantic.Field(default="", description="Object name (used for identification)")
    type_id: typing.Optional[int] = pydantic.Field(
        default=None, ge=0, le=255, description="Numeric type ID for C++ runtime (auto-assigned if None)"
    )
    map_char: str = pydantic.Field(default="?", description="Character used in ASCII maps")
    render_symbol: str = pydantic.Field(default="❓", description="Symbol used for rendering (e.g., emoji)")
    tags: list[str] = pydantic.Field(default_factory=list, description="Tags for this object instance")
    vibe: int = pydantic.Field(default=0, ge=0, le=255, description="Vibe value for this object instance")


class WallConfig(GridObjectConfig):
    """Python wall/block configuration."""

    type: typing.Literal["wall"] = pydantic.Field(default="wall")
    swappable: bool = pydantic.Field(default=False)


class ProtocolConfig(mettagrid.base_config.Config):
    vibes: list[str] = pydantic.Field(default_factory=list)
    input_resources: dict[str, int] = pydantic.Field(default_factory=dict)
    output_resources: dict[str, int] = pydantic.Field(default_factory=dict)
    cooldown: int = pydantic.Field(ge=0, default=0)


class AssemblerConfig(GridObjectConfig):
    """Python assembler configuration."""

    type: typing.Literal["assembler"] = pydantic.Field(default="assembler")
    protocols: list[ProtocolConfig] = pydantic.Field(
        default_factory=list,
        description="Protocols in reverse order of priority.",
    )
    allow_partial_usage: bool = pydantic.Field(
        default=False,
        description=(
            "Allow assembler to be used during cooldown with scaled resource requirements/outputs. "
            "This makes less sense if the assembler has multiple protocols."
        ),
    )
    max_uses: int = pydantic.Field(default=0, ge=0, description="Maximum number of uses (0 = unlimited)")
    exhaustion: float = pydantic.Field(
        default=0.0,
        ge=0.0,
        description=(
            "Exhaustion rate - cooldown multiplier grows by (1 + exhaustion) after each use (0 = no exhaustion)"
        ),
    )
    clip_immune: bool = pydantic.Field(
        default=False, description="If true, this assembler cannot be clipped by the Clipper system"
    )
    start_clipped: bool = pydantic.Field(
        default=False, description="If true, this assembler starts in a clipped state at the beginning of the game"
    )


class ChestConfig(GridObjectConfig):
    """Python chest configuration for multi-resource chests."""

    type: typing.Literal["chest"] = pydantic.Field(default="chest")

    # Vibe-based transfers: vibe -> resource -> delta
    vibe_transfers: dict[str, dict[str, int]] = pydantic.Field(
        default_factory=dict,
        description=(
            "Map from vibe to resource deltas. "
            "E.g., {'carbon': {'carbon': 10, 'energy': -5}} deposits 10 carbon and withdraws 5 energy when "
            "showing carbon vibe"
        ),
    )

    # Initial inventory for each resource
    initial_inventory: dict[str, int] = pydantic.Field(
        default_factory=dict, description="Initial inventory for each resource type"
    )

    # Resource limits for the chest's inventory
    resource_limits: dict[str, int] = pydantic.Field(
        default_factory=dict, description="Maximum amount per resource (uses inventory system's built-in limits)"
    )


class ClipperConfig(mettagrid.base_config.Config):
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

    unclipping_protocols: list[ProtocolConfig] = pydantic.Field(default_factory=list)
    length_scale: float = pydantic.Field(
        default=0.0,
        description="Controls spatial spread rate: weight = exp(-distance / length_scale). "
        "If <= 0, automatically calculated using percolation at runtime.",
    )
    cutoff_distance: float = pydantic.Field(
        default=0.0,
        ge=0.0,
        description="Maximum distance for infection weight calculations. "
        "If <= 0, automatically set to 3 * length_scale at runtime.",
    )
    clip_rate: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)


AnyGridObjectConfig = pydantic.SerializeAsAny[
    typing.Annotated[
        typing.Union[
            typing.Annotated[WallConfig, pydantic.Tag("wall")],
            typing.Annotated[AssemblerConfig, pydantic.Tag("assembler")],
            typing.Annotated[ChestConfig, pydantic.Tag("chest")],
        ],
        pydantic.Discriminator("type"),
    ]
]


class GameConfig(mettagrid.base_config.Config):
    """Python game configuration.

    Note: Type IDs are automatically assigned during validation when the GameConfig
    is constructed. If you need to add objects after construction, create a new
    GameConfig instance rather than modifying the objects dict post-construction,
    as type_id assignment only happens at validation time.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _resolved_type_ids: bool = pydantic.PrivateAttr(default=False)

    resource_names: list[str] = pydantic.Field(
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
    vibe_names: list[str] = pydantic.Field(default_factory=list)
    num_agents: int = pydantic.Field(ge=1, default=24)
    # max_steps = zero means "no limit"
    max_steps: int = pydantic.Field(ge=0, default=1000)
    # default is that we terminate / use "done" vs truncation
    episode_truncates: bool = pydantic.Field(default=False)
    obs: obs_config_module.ObsConfig = pydantic.Field(default_factory=obs_config_module.ObsConfig)
    agent: AgentConfig = pydantic.Field(default_factory=AgentConfig)
    agents: list[AgentConfig] = pydantic.Field(default_factory=list)
    actions: ActionsConfig = pydantic.Field(default_factory=lambda: ActionsConfig())
    global_obs: GlobalObsConfig = pydantic.Field(default_factory=GlobalObsConfig)
    objects: dict[str, AnyGridObjectConfig] = pydantic.Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: typing.Optional[typing.Any] = None

    resource_loss_prob: float = pydantic.Field(default=0.0, description="Probability of resource loss per step")

    # Inventory regeneration interval (global check timing)
    inventory_regen_interval: int = pydantic.Field(
        default=0, ge=0, description="Interval in timesteps between regenerations (0 = disabled)"
    )

    # Global clipper system
    clipper: typing.Optional[ClipperConfig] = pydantic.Field(default=None, description="Global clipper configuration")

    # Map builder configuration - accepts any MapBuilder config
    map_builder: mettagrid.map_builder.map_builder.AnyMapBuilderConfig = pydantic.Field(
        default_factory=lambda: mettagrid.map_builder.random.RandomMapBuilder.Config(agents=24)
    )

    # Feature Flags
    track_movement_metrics: bool = pydantic.Field(
        default=True, description="Enable movement metrics tracking (sequential rotations)"
    )
    protocol_details_obs: bool = pydantic.Field(
        default=False, description="Objects show their protocol inputs and outputs when observed"
    )

    reward_estimates: typing.Optional[dict[str, float]] = pydantic.Field(default=None)

    @pydantic.model_validator(mode="after")
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
            self.vibe_names = [vibe.name for vibe in vibes_module.VIBES[:num_vibes]]

    def _ensure_type_ids_assigned(self) -> None:
        """Ensure type IDs are assigned if they haven't been yet."""
        if not self._resolved_type_ids:
            id_map_module.IdMap.assign_type_ids(self)
            self._resolved_type_ids = True

    def __getattribute__(self, name: str):
        """Intercept attribute access to ensure type IDs are assigned when accessing objects."""
        if name == "objects":
            self._ensure_type_ids_assigned()
        return super().__getattribute__(name)

    def id_map(self) -> id_map_module.IdMap:
        """Get the observation feature ID map for this configuration."""
        # Create a minimal MettaGridConfig wrapper
        wrapper = MettaGridConfig(game=self)
        return id_map_module.IdMap(wrapper)


class TeacherConfig(mettagrid.base_config.Config):
    """Teacher configuration."""

    enabled: bool = pydantic.Field(default=False)
    use_actions: bool = pydantic.Field(default=False)
    policy: str = pydantic.Field(default="baseline")


class MettaGridConfig(mettagrid.base_config.Config):
    """Environment configuration."""

    label: str = pydantic.Field(default="mettagrid")
    game: GameConfig = pydantic.Field(default_factory=GameConfig)
    desync_episodes: bool = pydantic.Field(default=True)
    teacher: TeacherConfig = pydantic.Field(default_factory=TeacherConfig)

    def id_map(self) -> id_map_module.IdMap:
        """Get the observation feature ID map for this configuration."""
        return id_map_module.IdMap(self)

    def with_ascii_map(self, map_data: list[list[str]]) -> "MettaGridConfig":
        self.game.map_builder = mettagrid.map_builder.ascii.AsciiMapBuilder.Config(
            map_data=map_data,
            char_to_name_map={o.map_char: o.name for o in self.game.objects.values()},
        )
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "MettaGridConfig":
        """Create an empty room environment configuration."""
        map_builder = mettagrid.map_builder.random.RandomMapBuilder.Config(
            agents=num_agents, width=width, height=height, border_width=border_width
        )
        actions = ActionsConfig(
            move=MoveActionConfig(),
        )
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(name="wall", map_char="#", render_symbol="⬛", swappable=False)
        return MettaGridConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

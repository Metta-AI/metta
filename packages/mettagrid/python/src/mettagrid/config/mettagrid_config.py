from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    SerializeAsAny,
    Tag,
    field_validator,
    model_validator,
)

from mettagrid.base_config import Config
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig
from mettagrid.map_builder.random import RandomMapBuilder

# ===== Python Configuration Models =====

# Left to right, top to bottom.
FixedPosition = Literal["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
Position = FixedPosition | Literal["Any"]


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


class ActionConfig(Config):
    """Python action configuration."""

    enabled: bool = Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: dict[str, int] = Field(default_factory=dict)
    consumed_resources: dict[str, float] = Field(default_factory=dict)


class AttackActionConfig(ActionConfig):
    """Python attack action configuration."""

    defense_resources: dict[str, int] = Field(default_factory=dict)


class ChangeGlyphActionConfig(ActionConfig):
    """Change glyph action configuration."""

    number_of_glyphs: int = Field(default=0, ge=0, le=255)


class ResourceModActionConfig(ActionConfig):
    """Resource mod action configuration."""

    modifies: dict[str, float] = Field(default_factory=dict)
    agent_radius: int = Field(default=0, ge=0, le=255)
    converter_radius: int = Field(default=0, ge=0, le=255)
    scales: bool = Field(default=False)


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: ActionConfig = Field(default_factory=lambda: ActionConfig())
    move: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))  # Default movement action
    rotate: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    put_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    get_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    attack: AttackActionConfig = Field(default_factory=lambda: AttackActionConfig(enabled=False))
    swap: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    change_glyph: ChangeGlyphActionConfig = Field(default_factory=lambda: ChangeGlyphActionConfig(enabled=False))
    resource_mod: ResourceModActionConfig = Field(default_factory=lambda: ResourceModActionConfig(enabled=False))


class GlobalObsConfig(Config):
    """Global observation configuration."""

    episode_completion_pct: bool = Field(default=True)

    # Controls whether the last_action global token is included
    last_action: bool = Field(default=True)

    last_reward: bool = Field(default=True)

    # Controls whether visitation counts are included in observations
    visitation_counts: bool = Field(default=False)


class GridObjectConfig(Config):
    """Base configuration for all grid objects."""

    name: str = Field(default="", description="Object name (used for identification)")
    type_id: int = Field(ge=0, le=255, description="Numeric type ID for C++ runtime")
    map_char: str = Field(default="?", description="Character used in ASCII maps")
    render_symbol: str = Field(default="❓", description="Symbol used for rendering (e.g., emoji)")
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")


class WallConfig(GridObjectConfig):
    """Python wall/block configuration."""

    type: Literal["wall"] = Field(default="wall")
    swappable: bool = Field(default=False)


class ConverterConfig(GridObjectConfig):
    """Python converter configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: Literal["converter"] = Field(default="converter")
    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    max_output: int = Field(ge=-1, default=5)
    max_conversions: int = Field(default=-1)
    conversion_ticks: int = Field(ge=0, default=1)
    cooldown: list[int] = Field(default_factory=lambda: [0])
    initial_resource_count: int = Field(ge=0, default=0)

    @field_validator("cooldown", mode="before")
    @classmethod
    def normalize_cooldown(cls, value: Any) -> list[int]:
        if value is None:
            return [0]
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = [int(item) for item in value]
            if not values:
                return [0]
            return values
        raise TypeError("cooldown must be an int or iterable of ints")


class RecipeConfig(Config):
    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    cooldown: int = Field(ge=0, default=0)


class AssemblerConfig(GridObjectConfig):
    """Python assembler configuration."""

    type: Literal["assembler"] = Field(default="assembler")
    recipes: list[tuple[list[Position], RecipeConfig]] = Field(
        default_factory=list,
        description="Recipes in reverse order of priority.",
    )
    allow_partial_usage: bool = Field(
        default=False,
        description=(
            "Allow assembler to be used during cooldown with scaled resource requirements/outputs. "
            "This makes less sense if the assembler has multiple recipes."
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
    fully_overlapping_recipes_allowed: bool = Field(default=False, description="Allow recipes to fully overlap")


class ChestConfig(GridObjectConfig):
    """Python chest configuration."""

    type: Literal["chest"] = Field(default="chest")
    resource_type: str = Field(description="Resource type that this chest can store")
    position_deltas: list[tuple[FixedPosition, int]] = Field(
        default_factory=list,
        description=(
            "List of (position, delta) tuples. "
            "Positive delta = deposit, negative = withdraw (e.g., (E, 1) deposits 1, (N, -5) withdraws 5)"
        ),
    )
    initial_inventory: int = Field(default=0, ge=0, description="Initial amount of resource_type in the chest")
    max_inventory: int = Field(
        default=255,
        ge=-1,
        description="Maximum inventory (resources are destroyed when depositing beyond this, -1 = unlimited)",
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

    unclipping_recipes: list[RecipeConfig] = Field(default_factory=list)
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
            Annotated[ConverterConfig, Tag("converter")],
            Annotated[AssemblerConfig, Tag("assembler")],
            Annotated[ChestConfig, Tag("chest")],
        ],
        Discriminator("type"),
    ]
]


class GameConfig(Config):
    """Python game configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    num_agents: int = Field(ge=1, default=24)
    # max_steps = zero means "no limit"
    max_steps: int = Field(ge=0, default=1000)
    # default is that we terminate / use "done" vs truncation
    episode_truncates: bool = Field(default=False)
    obs_width: Literal[3, 5, 7, 9, 11, 13, 15] = Field(default=11)
    obs_height: Literal[3, 5, 7, 9, 11, 13, 15] = Field(default=11)
    num_observation_tokens: int = Field(ge=1, default=200)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    actions: ActionsConfig = Field(default_factory=lambda: ActionsConfig(noop=ActionConfig()))
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
    map_builder: AnyMapBuilderConfig = RandomMapBuilder.Config(agents=24)

    # Feature Flags
    track_movement_metrics: bool = Field(
        default=True, description="Enable movement metrics tracking (sequential rotations)"
    )
    recipe_details_obs: bool = Field(
        default=False, description="Converters show their recipe inputs and outputs when observed"
    )
    allow_diagonals: bool = Field(default=False, description="Enable actions to be aware of diagonal orientations")

    reward_estimates: Optional[dict[str, float]] = Field(default=None)


class MettaGridConfig(Config):
    """Environment configuration."""

    label: str = Field(default="mettagrid")
    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_fields(self) -> "MettaGridConfig":
        return self

    def with_ascii_map(self, map_data: list[list[str]]) -> "MettaGridConfig":
        self.game.map_builder = AsciiMapBuilder.Config(
            map_data=map_data,
            char_to_name_map={o.map_char: o.name for o in self.game.objects.values()},
        )
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "MettaGridConfig":
        """Create an empty room environment configuration."""
        map_builder = RandomMapBuilder.Config(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(
            move=ActionConfig(),
        )
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛", swappable=False)
        return MettaGridConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from mettagrid.config.config import Config

if TYPE_CHECKING:
    from mettagrid.map_builder.map_builder import AnyMapBuilderConfig


def _default_map_builder_config() -> "AnyMapBuilderConfig":
    from mettagrid.map_builder.random import RandomMapBuilder

    return RandomMapBuilder.Config(agents=24)


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
    resource_limits: dict[str, int] = Field(default_factory=dict)
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


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    move: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=True))  # Default movement action
    rotate: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    put_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=True))
    get_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=True))
    attack: AttackActionConfig = Field(default_factory=lambda: AttackActionConfig(enabled=False))
    swap: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    change_color: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    change_glyph: ChangeGlyphActionConfig = Field(default_factory=lambda: ChangeGlyphActionConfig(enabled=False))


class GlobalObsConfig(Config):
    """Global observation configuration."""

    episode_completion_pct: bool = Field(default=True)

    # Controls both last_action and last_action_arg
    last_action: bool = Field(default=True)

    last_reward: bool = Field(default=True)

    # Controls whether visitation counts are included in observations
    visitation_counts: bool = Field(default=False)


class WallConfig(Config):
    """Python wall/block configuration."""

    type_id: int
    swappable: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")


class ConverterConfig(Config):
    """Python converter configuration."""

    name: str = Field(default="converter")
    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    type_id: int = Field(default=0, ge=0, le=255)
    max_output: int = Field(ge=-1, default=5)
    max_conversions: int = Field(default=-1)
    conversion_ticks: int = Field(ge=0, default=1)
    cooldown: int = Field(ge=0)
    initial_resource_count: int = Field(ge=0, default=0)
    color: int = Field(default=0, ge=0, le=255)
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")


class RecipeConfig(Config):
    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    cooldown: int = Field(ge=0, default=0)
    max_use: Optional[int] = Field(ge=0, default=None)


class AssemblerConfig(Config):
    """Python assembler configuration."""

    name: str = Field(default="assembler")
    type_id: int = Field(default=0, ge=0, le=255)
    recipes: list[tuple[list[Position], RecipeConfig]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")
    allow_partial_usage: bool = Field(
        default=False,
        description=(
            "Allow assembler to be used during cooldown with scaled resource requirements/outputs. "
            "This makes less sense if the assembler has multiple recipes."
        ),
    )
    max_uses: int = Field(default=0, ge=0, description="Maximum number of uses (0 = unlimited)")


class ChestConfig(Config):
    """Python chest configuration."""

    type_id: int = Field(default=0, ge=0, le=255)
    resource_type: str = Field(description="Resource type that this chest can store")
    deposit_positions: list[FixedPosition] = Field(
        default_factory=list, description="Positions where agents can deposit resources"
    )
    withdrawal_positions: list[FixedPosition] = Field(
        default_factory=list, description="Positions where agents can withdraw resources"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for this object instance")


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
    objects: dict[str, ConverterConfig | WallConfig | AssemblerConfig | ChestConfig] = Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None

    resource_loss_prob: float = Field(default=0.0, description="Probability of resource loss per step")

    # Inventory regeneration settings
    inventory_regen_amounts: dict[str, int] = Field(
        default_factory=dict, description="Resources to regenerate and their amounts per regeneration interval"
    )
    inventory_regen_interval: int = Field(
        default=0, ge=0, description="Interval in timesteps between regenerations (0 = disabled)"
    )

    # Map builder configuration - accepts any MapBuilder config
    map_builder: "AnyMapBuilderConfig" = Field(default_factory=_default_map_builder_config)

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
        from mettagrid.map_builder.ascii import AsciiMapBuilder

        self.game.map_builder = AsciiMapBuilder.Config(map_data=map_data)
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "MettaGridConfig":
        """Create an empty room environment configuration."""
        from mettagrid.map_builder.random import RandomMapBuilder

        map_builder = RandomMapBuilder.Config(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(
            move=ActionConfig(),
            rotate=ActionConfig(enabled=False),  # Disabled for unified movement system
        )
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(type_id=1, swappable=False)
        return MettaGridConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

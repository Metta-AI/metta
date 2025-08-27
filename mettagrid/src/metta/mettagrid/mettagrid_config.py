from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.common.config import Config
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.map_builder.random import RandomMapBuilder

if TYPE_CHECKING:
    from metta.sim.simulation_config import SimulationConfig
from metta.mettagrid.map_builder.map_builder import AnyMapBuilderConfig

# ===== Python Configuration Models =====


class InventoryRewards(Config):
    """Inventory-based reward configuration."""

    ore_red: float = Field(default=0)
    ore_blue: float = Field(default=0)
    ore_green: float = Field(default=0)
    ore_red_max: int = Field(default=255)
    ore_blue_max: int = Field(default=255)
    ore_green_max: int = Field(default=255)
    battery_red: float = Field(default=0)
    battery_blue: float = Field(default=0)
    battery_green: float = Field(default=0)
    battery_red_max: int = Field(default=255)
    battery_blue_max: int = Field(default=255)
    battery_green_max: int = Field(default=255)
    heart: float = Field(default=1)
    heart_max: int = Field(default=255)
    armor: float = Field(default=0)
    armor_max: int = Field(default=255)
    laser: float = Field(default=0)
    laser_max: int = Field(default=255)
    blueprint: float = Field(default=0)
    blueprint_max: int = Field(default=255)


class StatsRewards(Config):
    """Agent stats-based reward configuration.

    Maps stat names to reward values. Stats are tracked by the StatsTracker
    and can include things like 'action.attack.agent', 'inventory.armor.gained', etc.
    Each entry can have:
    - stat_name: reward_per_unit
    - stat_name_max: maximum cumulative reward for this stat
    """

    model_config = ConfigDict(extra="allow")  # Allow any stat names to be added dynamically


class AgentRewards(Config):
    """Agent reward configuration with separate inventory and stats rewards."""

    inventory: InventoryRewards = Field(default_factory=InventoryRewards)
    stats: StatsRewards = Field(default_factory=StatsRewards)


class AgentConfig(Config):
    """Python agent configuration."""

    default_resource_limit: int = Field(default=255, ge=0)
    resource_limits: dict[str, int] = Field(default_factory=dict)
    freeze_duration: int = Field(default=10, ge=-1)
    rewards: AgentRewards = Field(default_factory=AgentRewards)
    action_failure_penalty: float = Field(default=0, ge=0)
    initial_inventory: dict[str, int] = Field(default_factory=dict)


class GroupConfig(Config):
    """Python group configuration."""

    id: int = Field(default=0)
    sprite: Optional[int] = Field(default=None)
    # group_reward_pct values outside of [0.0,1.0] are probably mistakes, and are probably
    # unstable. If you want to use values outside this range, please update this comment!
    group_reward_pct: float = Field(default=0, ge=0, le=1)
    props: Optional[AgentConfig] = Field(default=None)


class ActionConfig(Config):
    """Python action configuration."""

    enabled: bool = Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: dict[str, int] = Field(default_factory=dict)
    consumed_resources: dict[str, int] = Field(default_factory=dict)


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
    put_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    place_box: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
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

    # Controls whether resource rewards are included in observations
    resource_rewards: bool = Field(default=False)

    # Controls whether visitation counts are included in observations
    visitation_counts: bool = Field(default=False)


class WallConfig(Config):
    """Python wall/block configuration."""

    type_id: int
    swappable: bool = Field(default=False)


class BoxConfig(Config):
    """Python box configuration."""

    type_id: int = Field(default=0, ge=0, le=255)
    resources_to_create: dict[str, int] = Field(default_factory=dict)


class ConverterConfig(Config):
    """Python converter configuration."""

    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    type_id: int = Field(default=0, ge=0, le=255)
    max_output: int = Field(ge=-1, default=5)
    max_conversions: int = Field(default=-1)
    conversion_ticks: int = Field(ge=0, default=1)
    cooldown: int = Field(ge=0)
    initial_resource_count: int = Field(ge=0, default=0)
    color: int = Field(default=0, ge=0, le=255)


class GameConfig(Config):
    """Python game configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inventory_item_names: list[str] = Field(
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
    # Every agent must be in a group, so we need at least one group
    groups: dict[str, GroupConfig] = Field(default_factory=lambda: {"agent": GroupConfig()}, min_length=1)
    actions: ActionsConfig = Field(default_factory=lambda: ActionsConfig(noop=ActionConfig()))
    global_obs: GlobalObsConfig = Field(default_factory=GlobalObsConfig)
    objects: dict[str, ConverterConfig | WallConfig | BoxConfig] = Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None

    resource_loss_prob: float = Field(default=0.0, description="Probability of resource loss per step")

    # Map builder configuration - accepts any MapBuilder config
    map_builder: AnyMapBuilderConfig = RandomMapBuilder.Config(agents=24)

    # Feature Flags
    track_movement_metrics: bool = Field(
        default=True, description="Enable movement metrics tracking (sequential rotations)"
    )
    no_agent_interference: bool = Field(
        default=False, description="Enable agents to move through and not observe each other"
    )
    recipe_details_obs: bool = Field(
        default=False, description="Converters show their recipe inputs and outputs when observed"
    )
    allow_diagonals: bool = Field(default=False, description="Enable actions to be aware of diagonal orientations")


class EnvConfig(Config):
    """Environment configuration."""

    label: str = Field(default="mettagrid")
    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_fields(self) -> "EnvConfig":
        return self

    def to_sim(self, name: str) -> "SimulationConfig":
        from metta.sim.simulation_config import SimulationConfig

        return SimulationConfig(
            name=name,
            env=self,
        )

    def with_ascii_map(self, map_data: list[list[str]]) -> "EnvConfig":
        self.game.map_builder = AsciiMapBuilder.Config(map_data=map_data)
        return self

    @staticmethod
    def EmptyRoom(
        num_agents: int, width: int = 10, height: int = 10, border_width: int = 1, with_walls: bool = False
    ) -> "EnvConfig":
        """Create an empty room environment configuration."""
        map_builder = RandomMapBuilder.Config(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(
            move=ActionConfig(),
            rotate=ActionConfig(),
        )
        objects = {}
        if border_width > 0 or with_walls:
            objects["wall"] = WallConfig(type_id=1, swappable=False)
        return EnvConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

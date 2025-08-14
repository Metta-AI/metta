from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from metta.common.util.config import Config
from metta.sim.simulation_config import SimulationConfig

if TYPE_CHECKING:
    from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.map_builder import MapBuilderConfigUnion
from metta.mettagrid.map_builder.random import RandomMapBuilderConfig

# ===== Python Configuration Models =====


class InventoryRewards(Config):
    """Inventory-based reward configuration."""

    ore_red: Optional[float] = Field(default=None)
    ore_blue: Optional[float] = Field(default=None)
    ore_green: Optional[float] = Field(default=None)
    ore_red_max: Optional[int] = Field(default=None)
    ore_blue_max: Optional[int] = Field(default=None)
    ore_green_max: Optional[int] = Field(default=None)
    battery_red: Optional[float] = Field(default=None)
    battery_blue: Optional[float] = Field(default=None)
    battery_green: Optional[float] = Field(default=None)
    battery_red_max: Optional[int] = Field(default=None)
    battery_blue_max: Optional[int] = Field(default=None)
    battery_green_max: Optional[int] = Field(default=None)
    heart: Optional[float] = Field(default=None)
    heart_max: Optional[int] = Field(default=None)
    armor: Optional[float] = Field(default=None)
    armor_max: Optional[int] = Field(default=None)
    laser: Optional[float] = Field(default=None)
    laser_max: Optional[int] = Field(default=None)
    blueprint: Optional[float] = Field(default=None)
    blueprint_max: Optional[int] = Field(default=None)


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

    inventory: Optional[InventoryRewards] = Field(default_factory=InventoryRewards)
    stats: Optional[StatsRewards] = Field(default_factory=StatsRewards)

    # For backward compatibility, handle old format
    def __init__(self, **data):
        # If we have direct inventory reward keys, move them to inventory
        inventory_keys = [
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
        inventory_max_keys = [f"{k}_max" for k in inventory_keys]

        # Check if this is the old format (has direct reward keys)
        if any(k in data for k in inventory_keys + inventory_max_keys):
            # Old format - move to inventory
            if "inventory" not in data:
                data["inventory"] = {}

            for key in list(data.keys()):
                if key in inventory_keys or key in inventory_max_keys:
                    data["inventory"][key] = data.pop(key)

        super().__init__(**data)


class AgentConfig(Config):
    """Python agent configuration."""

    default_resource_limit: Optional[int] = Field(default=0, ge=0)
    resource_limits: Optional[dict[str, int]] = Field(default_factory=dict)
    freeze_duration: Optional[int] = Field(default=10, ge=-1)
    rewards: Optional[AgentRewards] = Field(default_factory=AgentRewards)
    action_failure_penalty: Optional[float] = Field(default=0, ge=0)
    initial_inventory: Optional[dict[str, int]] = Field(default_factory=dict)


class GroupConfig(Config):
    """Python group configuration."""

    id: int
    sprite: Optional[int] = Field(default=None)
    # group_reward_pct values outside of [0.0,1.0] are probably mistakes, and are probably
    # unstable. If you want to use values outside this range, please update this comment!
    group_reward_pct: float = Field(default=0, ge=0, le=1)
    props: AgentConfig = Field(default_factory=AgentConfig)


class ActionConfig(Config):
    """Python action configuration."""

    enabled: bool = Field(default=True)
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: Optional[dict[str, int]] = Field(default=None)
    consumed_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class AttackActionConfig(ActionConfig):
    """Python attack action configuration."""

    defense_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class ChangeGlyphActionConfig(ActionConfig):
    """Change glyph action configuration."""

    number_of_glyphs: int = Field(default=0, ge=0, le=255)


class ActionsConfig(Config):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    move: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    move_8way: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    move_cardinal: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    rotate: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    put_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    place_box: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
    get_items: ActionConfig = Field(default_factory=lambda: ActionConfig(enabled=False))
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
    groups: dict[str, GroupConfig] = Field(
        default_factory=lambda: {"agent": GroupConfig(id=0, sprite=0, props=AgentConfig())}, min_length=1
    )
    actions: ActionsConfig = Field(default_factory=lambda: ActionsConfig(noop=ActionConfig()))
    global_obs: GlobalObsConfig = Field(default_factory=GlobalObsConfig)
    recipe_details_obs: bool = Field(default=False)
    objects: dict[str, ConverterConfig | WallConfig | BoxConfig] = Field(default_factory=dict)
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None

    map_builder: MapBuilderConfigUnion = RandomMapBuilderConfig(agents=24)

    # Movement metrics configuration
    track_movement_metrics: bool = Field(
        default=False, description="Enable movement metrics tracking (sequential rotations)"
    )
    no_agent_interference: bool = Field(
        default=False, description="Enable agents to move through and not observe each other"
    )
    resource_loss_prob: float = Field(default=0.0, description="Probability of resource loss per step")


class EnvConfig(Config):
    """Environment configuration."""

    game: GameConfig = Field(default_factory=GameConfig)
    desync_episodes: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_fields(self) -> "EnvConfig":
        return self

    def to_curriculum_cfg(self) -> "CurriculumConfig":
        from metta.cogworks.curriculum.curriculum import CurriculumConfig
        from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig

        return CurriculumConfig(
            task_generator=SingleTaskGeneratorConfig(env=self),
        )

    def to_curriculum(self):
        from metta.cogworks.curriculum.curriculum import Curriculum

        return Curriculum(self.to_curriculum_cfg())

    def to_sim(self, name: str) -> SimulationConfig:
        from metta.sim.simulation_config import SimulationConfig

        return SimulationConfig(
            name=name,
            env=self,
        )

    @staticmethod
    def EmptyRoom(num_agents: int, width: int = 10, height: int = 10, border_width: int = 1) -> "EnvConfig":
        """Create an empty room environment configuration."""
        map_builder = RandomMapBuilderConfig(agents=num_agents, width=width, height=height, border_width=border_width)
        actions = ActionsConfig(
            move_8way=ActionConfig(),
            rotate=ActionConfig(),
        )
        objects = {}
        if border_width > 0:
            objects["wall"] = WallConfig(type_id=1, swappable=False)
        return EnvConfig(
            game=GameConfig(map_builder=map_builder, actions=actions, num_agents=num_agents, objects=objects)
        )

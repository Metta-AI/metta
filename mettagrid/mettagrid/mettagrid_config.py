from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, RootModel


class AgentRewards(BaseModel):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = 0.0
    ore_red: Optional[float] = Field(default=0.005, alias="ore.red")
    ore_blue: Optional[float] = Field(default=0.005, alias="ore.blue")
    ore_green: Optional[float] = Field(default=0.005, alias="ore.green")
    battery_red: Optional[float] = Field(default=0.01, alias="battery.red")
    battery_blue: Optional[float] = Field(default=0.01, alias="battery.blue")
    battery_green: Optional[float] = Field(default=0.01, alias="battery.green")
    battery_red_max: Optional[float] = Field(default=5.0, alias="battery.red_max")
    battery_blue_max: Optional[float] = Field(default=5.0, alias="battery.blue_max")
    battery_green_max: Optional[float] = Field(default=5.0, alias="battery.green_max")
    heart: Optional[float] = 1.0
    heart_max: Optional[float] = 1000.0


class AgentConfig(BaseModel):
    """Agent configuration."""

    default_item_max: Optional[int] = 50
    freeze_duration: Optional[int] = 10
    inventory_size: Optional[int] = 0
    rewards: Optional[AgentRewards] = None


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModel):
    """Group configuration."""

    id: int
    sprite: Optional[int] = 0
    group_reward_pct: Optional[float] = 0.0
    props: Optional[GroupProps] = Field(default_factory=lambda: GroupProps({}))


class ActionConfig(BaseModel):
    """Action configuration."""

    enabled: bool = True


class ActionsConfig(BaseModel):
    """Actions configuration."""

    noop: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    move: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    rotate: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    put_items: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    get_items: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    attack: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    swap: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    change_color: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())


class WallConfig(BaseModel):
    """Wall/Block configuration."""

    swappable: Optional[bool] = False


class ConverterConfig(BaseModel):
    """Converter configuration for objects that convert items."""

    # Input items (e.g., "input_ore.red": 3)
    input_ore_red: Optional[int] = Field(default=None, alias="input_ore.red")
    input_ore_blue: Optional[int] = Field(default=None, alias="input_ore.blue")
    input_ore_green: Optional[int] = Field(default=None, alias="input_ore.green")
    input_battery_red: Optional[int] = Field(default=None, alias="input_battery.red")
    input_battery_blue: Optional[int] = Field(default=None, alias="input_battery.blue")
    input_battery_green: Optional[int] = Field(default=None, alias="input_battery.green")
    input_heart: Optional[int] = Field(default=None, alias="input_heart")
    input_blueprint: Optional[int] = Field(default=None, alias="input_blueprint")

    # Output items (e.g., "output_ore.red": 1)
    output_ore_red: Optional[int] = Field(default=None, alias="output_ore.red")
    output_ore_blue: Optional[int] = Field(default=None, alias="output_ore.blue")
    output_ore_green: Optional[int] = Field(default=None, alias="output_ore.green")
    output_battery_red: Optional[int] = Field(default=None, alias="output_battery.red")
    output_battery_blue: Optional[int] = Field(default=None, alias="output_battery.blue")
    output_battery_green: Optional[int] = Field(default=None, alias="output_battery.green")
    output_heart: Optional[int] = Field(default=None, alias="output_heart")
    output_armor: Optional[int] = Field(default=None, alias="output_armor")
    output_laser: Optional[int] = Field(default=None, alias="output_laser")
    output_blueprint: Optional[int] = Field(default=None, alias="output_blueprint")

    # Converter properties
    max_output: int
    conversion_ticks: int
    cooldown: int
    initial_items: int
    color: Optional[int] = 0


class ObjectsConfig(BaseModel):
    """Objects configuration."""

    altar: Optional[ConverterConfig] = None
    mine_red: Optional[ConverterConfig] = Field(default=None, alias="mine.red")
    mine_blue: Optional[ConverterConfig] = Field(default=None, alias="mine.blue")
    mine_green: Optional[ConverterConfig] = Field(default=None, alias="mine.green")
    generator_red: Optional[ConverterConfig] = Field(default=None, alias="generator.red")
    generator_blue: Optional[ConverterConfig] = Field(default=None, alias="generator.blue")
    generator_green: Optional[ConverterConfig] = Field(default=None, alias="generator.green")
    armory: Optional[ConverterConfig] = None
    lasery: Optional[ConverterConfig] = None
    lab: Optional[ConverterConfig] = None
    factory: Optional[ConverterConfig] = None
    temple: Optional[ConverterConfig] = None
    wall: Optional[WallConfig] = None
    block: Optional[WallConfig] = None


class RewardSharingGroup(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig(BaseModel):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup]] = None


class GameConfig(BaseModel):
    """Game configuration."""

    num_agents: int
    max_steps: int
    obs_width: int
    obs_height: int
    num_observation_tokens: int
    agent: AgentConfig
    groups: Dict[str, GroupConfig]
    actions: ActionsConfig
    objects: ObjectsConfig
    reward_sharing: Optional[RewardSharingConfig] = None

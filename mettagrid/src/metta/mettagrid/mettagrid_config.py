from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, RootModel


class AgentRewards(BaseModel):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = None
    ore_red: Optional[float] = Field(default=None, alias="ore.red")
    ore_blue: Optional[float] = Field(default=None, alias="ore.blue")
    ore_green: Optional[float] = Field(default=None, alias="ore.green")
    battery_red: Optional[float] = Field(default=None, alias="battery.red")
    battery_blue: Optional[float] = Field(default=None, alias="battery.blue")
    battery_green: Optional[float] = Field(default=None, alias="battery.green")
    battery_red_max: Optional[float] = Field(default=None, alias="battery.red_max")
    battery_blue_max: Optional[float] = Field(default=None, alias="battery.blue_max")
    battery_green_max: Optional[float] = Field(default=None, alias="battery.green_max")
    heart: Optional[float] = None
    heart_max: Optional[float] = None


class AgentConfig(BaseModel):
    """Agent configuration."""

    default_item_max: Optional[int] = None
    freeze_duration: Optional[int] = None
    rewards: Optional[AgentRewards] = None


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModel):
    """Group configuration."""

    id: int
    sprite: Optional[int] = None
    group_reward_pct: Optional[float] = None
    props: Optional[GroupProps] = None


class ActionConfig(BaseModel):
    """Action configuration."""

    enabled: bool


class ActionsConfig(BaseModel):
    """Actions configuration."""

    noop: ActionConfig
    move: ActionConfig
    rotate: ActionConfig
    put_items: ActionConfig
    get_items: ActionConfig
    attack: ActionConfig
    swap: ActionConfig
    change_color: ActionConfig


class WallConfig(BaseModel):
    """Wall/Block configuration."""

    swappable: Optional[bool] = None


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
    color: Optional[int] = None


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

    class Config:
        extra = "forbid"

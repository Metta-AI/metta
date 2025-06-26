from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel


class BaseModelWithForbidExtra(BaseModel):
    model_config = dict(extra="forbid")


class AgentRewards(BaseModelWithForbidExtra):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = Field(default=None, ge=0)
    ore_red: Optional[float] = Field(default=None, alias="ore.red")
    ore_blue: Optional[float] = Field(default=None, alias="ore.blue")
    ore_green: Optional[float] = Field(default=None, alias="ore.green")
    ore_red_max: Optional[float] = Field(default=None, alias="ore.red_max")
    ore_blue_max: Optional[float] = Field(default=None, alias="ore.blue_max")
    ore_green_max: Optional[float] = Field(default=None, alias="ore.green_max")
    battery_red: Optional[float] = Field(default=None, alias="battery.red")
    battery_blue: Optional[float] = Field(default=None, alias="battery.blue")
    battery_green: Optional[float] = Field(default=None, alias="battery.green")
    battery_red_max: Optional[float] = Field(default=None, alias="battery.red_max")
    battery_blue_max: Optional[float] = Field(default=None, alias="battery.blue_max")
    battery_green_max: Optional[float] = Field(default=None, alias="battery.green_max")
    heart: Optional[float] = Field(default=None)
    heart_max: Optional[float] = Field(default=None)
    armor: Optional[float] = Field(default=None)
    armor_max: Optional[float] = Field(default=None)
    laser: Optional[float] = Field(default=None)
    laser_max: Optional[float] = Field(default=None)
    blueprint: Optional[float] = Field(default=None)
    blueprint_max: Optional[float] = Field(default=None)


class AgentConfig(BaseModelWithForbidExtra):
    """Agent configuration."""

    default_item_max: Optional[int] = Field(default=None, ge=0)
    freeze_duration: Optional[int] = Field(default=None, ge=0)
    rewards: Optional[AgentRewards] = None
    ore_red_max: Optional[int] = Field(default=None, alias="ore.red_max")
    ore_blue_max: Optional[int] = Field(default=None, alias="ore.blue_max")
    ore_green_max: Optional[int] = Field(default=None, alias="ore.green_max")
    battery_red_max: Optional[int] = Field(default=None, alias="battery.red_max")
    battery_blue_max: Optional[int] = Field(default=None, alias="battery.blue_max")
    battery_green_max: Optional[int] = Field(default=None, alias="battery.green_max")
    heart_max: Optional[int] = Field(default=None)
    armor_max: Optional[int] = Field(default=None)
    laser_max: Optional[int] = Field(default=None)
    blueprint_max: Optional[int] = Field(default=None)


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModelWithForbidExtra):
    """Group configuration."""

    id: int
    sprite: Optional[int] = Field(default=None)
    # Values outside of 0 and 1 are probably mistakes, and are probably
    # unstable. If you want to use values outside this range, please update this comment!
    group_reward_pct: Optional[float] = Field(default=None, ge=0, le=1)
    props: Optional[GroupProps] = None


class ActionConfig(BaseModelWithForbidExtra):
    """Action configuration."""

    enabled: bool


class ActionsConfig(BaseModelWithForbidExtra):
    """Actions configuration."""

    noop: ActionConfig
    move: ActionConfig
    rotate: ActionConfig
    put_items: ActionConfig
    get_items: ActionConfig
    attack: ActionConfig
    swap: ActionConfig
    change_color: ActionConfig


class WallConfig(BaseModelWithForbidExtra):
    """Wall/Block configuration."""

    swappable: Optional[bool] = None


class ConverterConfig(BaseModelWithForbidExtra):
    """Converter configuration for objects that convert items."""

    # Input items (e.g., "input_ore.red": 3)
    input_ore_red: Optional[int] = Field(default=None, alias="input_ore.red", ge=0, le=255)
    input_ore_blue: Optional[int] = Field(default=None, alias="input_ore.blue", ge=0, le=255)
    input_ore_green: Optional[int] = Field(default=None, alias="input_ore.green", ge=0, le=255)
    input_battery_red: Optional[int] = Field(default=None, alias="input_battery.red", ge=0, le=255)
    input_battery_blue: Optional[int] = Field(default=None, alias="input_battery.blue", ge=0, le=255)
    input_battery_green: Optional[int] = Field(default=None, alias="input_battery.green", ge=0, le=255)
    input_heart: Optional[int] = Field(default=None, alias="input_heart", ge=0, le=255)
    input_armor: Optional[int] = Field(default=None, alias="input_armor", ge=0, le=255)
    input_laser: Optional[int] = Field(default=None, alias="input_laser", ge=0, le=255)
    input_blueprint: Optional[int] = Field(default=None, alias="input_blueprint", ge=0, le=255)

    # Output items (e.g., "output_ore.red": 1)
    output_ore_red: Optional[int] = Field(default=None, alias="output_ore.red", ge=0, le=255)
    output_ore_blue: Optional[int] = Field(default=None, alias="output_ore.blue", ge=0, le=255)
    output_ore_green: Optional[int] = Field(default=None, alias="output_ore.green", ge=0, le=255)
    output_battery_red: Optional[int] = Field(default=None, alias="output_battery.red", ge=0, le=255)
    output_battery_blue: Optional[int] = Field(default=None, alias="output_battery.blue", ge=0, le=255)
    output_battery_green: Optional[int] = Field(default=None, alias="output_battery.green", ge=0, le=255)
    output_heart: Optional[int] = Field(default=None, alias="output_heart", ge=0, le=255)
    output_armor: Optional[int] = Field(default=None, alias="output_armor", ge=0, le=255)
    output_laser: Optional[int] = Field(default=None, alias="output_laser", ge=0, le=255)
    output_blueprint: Optional[int] = Field(default=None, alias="output_blueprint", ge=0, le=255)

    # Converter properties
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_items: int = Field(ge=0)
    color: Optional[int] = Field(default=None, ge=0, le=255)


class RewardSharingGroup(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig(BaseModelWithForbidExtra):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup]] = None


class GameConfig(BaseModelWithForbidExtra):
    """Game configuration."""

    inventory_item_names: List[str]
    num_agents: int = Field(ge=1)
    # zero means "no limit"
    max_steps: int = Field(ge=0)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    agent: AgentConfig
    # Every agent must be in a group, so we need at least one group
    groups: Dict[str, GroupConfig] = Field(min_length=1)
    actions: ActionsConfig
    objects: Dict[str, ConverterConfig | WallConfig]
    reward_sharing: Optional[RewardSharingConfig] = None

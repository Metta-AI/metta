from typing import Any, Dict, List, Optional

from pydantic import Field, RootModel

from metta.common.util.typed_config import BaseModelWithForbidExtra


class AgentRewards(BaseModelWithForbidExtra):
    """Agent reward configuration."""

    ore_red: Optional[float] = Field(default=None)
    ore_blue: Optional[float] = Field(default=None)
    ore_green: Optional[float] = Field(default=None)
    ore_red_max: Optional[float] = Field(default=None)
    ore_blue_max: Optional[float] = Field(default=None)
    ore_green_max: Optional[float] = Field(default=None)
    battery_red: Optional[float] = Field(default=None)
    battery_blue: Optional[float] = Field(default=None)
    battery_green: Optional[float] = Field(default=None)
    battery_red_max: Optional[float] = Field(default=None)
    battery_blue_max: Optional[float] = Field(default=None)
    battery_green_max: Optional[float] = Field(default=None)
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

    default_resource_limit: Optional[int] = Field(default=None, ge=0)
    resource_limits: Optional[Dict[str, int]] = Field(default_factory=dict)
    freeze_duration: Optional[int] = Field(default=None, ge=-1)
    rewards: Optional[AgentRewards] = None
    action_failure_penalty: Optional[float] = Field(default=None, ge=0)


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
    # defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: Optional[Dict[str, int]] = None
    consumed_resources: Optional[Dict[str, int]] = Field(default_factory=dict)


class AttackActionConfig(ActionConfig):
    """Attack action configuration."""

    defense_resources: Optional[Dict[str, int]] = Field(default_factory=dict)


class ActionsConfig(BaseModelWithForbidExtra):
    """Actions configuration."""

    noop: ActionConfig
    move: ActionConfig
    rotate: ActionConfig
    put_items: ActionConfig
    get_items: ActionConfig
    attack: AttackActionConfig
    swap: ActionConfig
    change_color: ActionConfig


class WallConfig(BaseModelWithForbidExtra):
    """Wall/Block configuration."""

    type_id: int
    swappable: Optional[bool] = None


class ConverterConfig(BaseModelWithForbidExtra):
    """Converter configuration for objects that convert items."""

    # Input items (e.g., "input_ore_red": 3)
    input_ore_red: Optional[int] = Field(default=None, ge=0, le=255)
    input_ore_blue: Optional[int] = Field(default=None, ge=0, le=255)
    input_ore_green: Optional[int] = Field(default=None, ge=0, le=255)
    input_battery_red: Optional[int] = Field(default=None, ge=0, le=255)
    input_battery_blue: Optional[int] = Field(default=None, ge=0, le=255)
    input_battery_green: Optional[int] = Field(default=None, ge=0, le=255)
    input_heart: Optional[int] = Field(default=None, ge=0, le=255)
    input_armor: Optional[int] = Field(default=None, ge=0, le=255)
    input_laser: Optional[int] = Field(default=None, ge=0, le=255)
    input_blueprint: Optional[int] = Field(default=None, ge=0, le=255)

    # Output items (e.g., "output_ore_red": 1)
    output_ore_red: Optional[int] = Field(default=None, ge=0, le=255)
    output_ore_blue: Optional[int] = Field(default=None, ge=0, le=255)
    output_ore_green: Optional[int] = Field(default=None, ge=0, le=255)
    output_battery_red: Optional[int] = Field(default=None, ge=0, le=255)
    output_battery_blue: Optional[int] = Field(default=None, ge=0, le=255)
    output_battery_green: Optional[int] = Field(default=None, ge=0, le=255)
    output_heart: Optional[int] = Field(default=None, ge=0, le=255)
    output_armor: Optional[int] = Field(default=None, ge=0, le=255)
    output_laser: Optional[int] = Field(default=None, ge=0, le=255)
    output_blueprint: Optional[int] = Field(default=None, ge=0, le=255)

    # Converter properties
    type_id: int
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_items: int = Field(ge=0)
    color: Optional[int] = Field(default=None, ge=0, le=255)


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

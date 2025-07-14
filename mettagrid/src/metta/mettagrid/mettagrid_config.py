from typing import Any, Literal, Optional

from pydantic import Field, RootModel

from metta.common.util.typed_config import BaseModelWithForbidExtra


class AgentRewards(BaseModelWithForbidExtra):
    """Agent reward configuration."""

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


class AgentConfig(BaseModelWithForbidExtra):
    """Agent configuration."""

    default_resource_limit: Optional[int] = Field(default=None, ge=0)
    resource_limits: Optional[dict[str, int]] = Field(default_factory=dict)
    freeze_duration: Optional[int] = Field(default=None, ge=-1)
    rewards: Optional[AgentRewards] = Field(default=None)
    action_failure_penalty: Optional[float] = Field(default=None, ge=0)


class GroupProps(RootModel[dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModelWithForbidExtra):
    """Group configuration."""

    id: int
    sprite: Optional[int] = Field(default=None)
    # Values outside of 0 and 1 are probably mistakes, and are probably
    # unstable. If you want to use values outside this range, please update this comment!
    group_reward_pct: Optional[float] = Field(default=None, ge=0, le=1)
    props: Optional[GroupProps] = Field(default=None)


class ActionConfig(BaseModelWithForbidExtra):
    """Action configuration."""

    enabled: bool
    # defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: Optional[dict[str, int]] = Field(default=None)
    consumed_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class AttackActionConfig(ActionConfig):
    """Attack action configuration."""

    defense_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class ChangeGlyphActionConfig(ActionConfig):
    """Change glyph action configuration."""

    number_of_glyphs: int = Field(default=0, ge=0, le=255)


class ActionsConfig(BaseModelWithForbidExtra):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: Optional[ActionConfig] = None
    move: Optional[ActionConfig] = None
    rotate: Optional[ActionConfig] = None
    put_items: Optional[ActionConfig] = None
    get_items: Optional[ActionConfig] = None
    attack: Optional[AttackActionConfig] = None
    swap: Optional[ActionConfig] = None
    change_color: Optional[ActionConfig] = None
    change_glyph: Optional[ChangeGlyphActionConfig] = None


class WallConfig(BaseModelWithForbidExtra):
    """Wall/Block configuration."""

    type_id: int
    swappable: bool = Field(default=False)


class ConverterConfig(BaseModelWithForbidExtra):
    """Converter configuration for objects that convert items."""

    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    type_id: int = Field(default=0, ge=0, le=255)
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_resource_count: int = Field(ge=0)
    color: int = Field(default=0, ge=0, le=255)


class GameConfig(BaseModelWithForbidExtra):
    """Game configuration."""

    inventory_item_names: list[str]
    num_agents: int = Field(ge=1)
    # zero means "no limit"
    max_steps: int = Field(ge=0)
    obs_width: Literal[1, 3, 5, 7, 9, 11, 13, 15]

    num_observation_tokens: int = Field(ge=1)
    agent: AgentConfig
    # Every agent must be in a group, so we need at least one group
    groups: dict[str, GroupConfig] = Field(min_length=1)
    actions: ActionsConfig
    objects: dict[str, ConverterConfig | WallConfig]

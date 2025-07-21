from typing import Optional

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra

# ===== Python Configuration Models =====


class PyAgentRewards(BaseModelWithForbidExtra):
    """Python agent reward configuration."""

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


class PyAgentConfig(BaseModelWithForbidExtra):
    """Python agent configuration."""

    default_resource_limit: Optional[int] = Field(default=0, ge=0)
    resource_limits: Optional[dict[str, int]] = Field(default_factory=dict)
    freeze_duration: Optional[int] = Field(default=0, ge=-1)
    rewards: Optional[PyAgentRewards] = Field(default_factory=PyAgentRewards)
    action_failure_penalty: Optional[float] = Field(default=0, ge=0)


class PyGroupConfig(BaseModelWithForbidExtra):
    """Python group configuration."""

    id: int
    sprite: Optional[int] = Field(default=None)
    # group_reward_pct values outside of [0.0,1.0] are probably mistakes, and are probably
    # unstable. If you want to use values outside this range, please update this comment!
    group_reward_pct: float = Field(default=0, ge=0, le=1)
    props: PyAgentConfig = Field(default_factory=PyAgentConfig)


class PyActionConfig(BaseModelWithForbidExtra):
    """Python action configuration."""

    enabled: bool
    # required_resources defaults to consumed_resources. Otherwise, should be a superset of consumed_resources.
    required_resources: Optional[dict[str, int]] = Field(default=None)
    consumed_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class PyAttackActionConfig(PyActionConfig):
    """Python attack action configuration."""

    defense_resources: Optional[dict[str, int]] = Field(default_factory=dict)


class PyChangeGlyphActionConfig(PyActionConfig):
    """Change glyph action configuration."""

    number_of_glyphs: int = Field(default=0, ge=0, le=255)


class PyActionsConfig(BaseModelWithForbidExtra):
    """
    Actions configuration.

    Omitted actions are disabled by default.
    """

    noop: Optional[PyActionConfig] = None
    move: Optional[PyActionConfig] = None
    rotate: Optional[PyActionConfig] = None
    put_items: Optional[PyActionConfig] = None
    get_items: Optional[PyActionConfig] = None
    attack: Optional[PyAttackActionConfig] = None
    swap: Optional[PyActionConfig] = None
    change_color: Optional[PyActionConfig] = None
    change_glyph: Optional[PyChangeGlyphActionConfig] = None


class PyWallConfig(BaseModelWithForbidExtra):
    """Python wall/block configuration."""

    type_id: int
    swappable: bool = Field(default=False)


class PyConverterConfig(BaseModelWithForbidExtra):
    """Python converter configuration."""

    input_resources: dict[str, int] = Field(default_factory=dict)
    output_resources: dict[str, int] = Field(default_factory=dict)
    type_id: int = Field(default=0, ge=0, le=255)
    max_output: int = Field(ge=-1)
    max_conversions: int = Field(default=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_resource_count: int = Field(ge=0)
    color: int = Field(default=0, ge=0, le=255)


class PyGameConfig(BaseModelWithForbidExtra):
    """Python game configuration."""

    inventory_item_names: list[str]
    num_agents: int = Field(ge=1)
    # max_steps = zero means "no limit"
    max_steps: int = Field(ge=0)
    # default is that we terminate / use "done" vs truncation
    episode_truncates: bool = Field(default=False)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    agent: PyAgentConfig
    # Every agent must be in a group, so we need at least one group
    groups: dict[str, PyGroupConfig] = Field(min_length=1)
    actions: PyActionsConfig
    objects: dict[str, PyConverterConfig | PyWallConfig]

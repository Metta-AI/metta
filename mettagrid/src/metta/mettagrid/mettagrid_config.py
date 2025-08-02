from typing import Any, Literal, Optional

from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra

# ===== Python Configuration Models =====


class PyInventoryRewards(BaseModelWithForbidExtra):
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


class PyStatsRewards(BaseModelWithForbidExtra):
    """Agent stats-based reward configuration.

    Maps stat names to reward values. Stats are tracked by the StatsTracker
    and can include things like 'action.attack.agent', 'inventory.armor.gained', etc.
    Each entry can have:
    - stat_name: reward_per_unit
    - stat_name_max: maximum cumulative reward for this stat
    """

    class Config:
        extra = "allow"  # Allow any stat names to be added dynamically


class PyAgentRewards(BaseModelWithForbidExtra):
    """Agent reward configuration with separate inventory and stats rewards."""

    inventory: Optional[PyInventoryRewards] = Field(default_factory=PyInventoryRewards)
    stats: Optional[PyStatsRewards] = Field(default_factory=PyStatsRewards)

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
    move_8way: Optional[PyActionConfig] = None
    move_cardinal: Optional[PyActionConfig] = None
    rotate: Optional[PyActionConfig] = None
    put_items: Optional[PyActionConfig] = None
    get_items: Optional[PyActionConfig] = None
    attack: Optional[PyAttackActionConfig] = None
    swap: Optional[PyActionConfig] = None
    change_color: Optional[PyActionConfig] = None
    change_glyph: Optional[PyChangeGlyphActionConfig] = None


class PyGlobalObsConfig(BaseModelWithForbidExtra):
    """Global observation configuration."""

    episode_completion_pct: bool = Field(default=True)

    # Controls both last_action and last_action_arg
    last_action: bool = Field(default=True)
    last_reward: bool = Field(default=True)

    # Controls whether resource rewards are included in observations
    resource_rewards: bool = Field(default=False)


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
    obs_width: Literal[3, 5, 7, 9, 11, 13, 15]
    obs_height: Literal[3, 5, 7, 9, 11, 13, 15]
    num_observation_tokens: int = Field(ge=1)
    agent: PyAgentConfig
    # Every agent must be in a group, so we need at least one group
    groups: dict[str, PyGroupConfig] = Field(min_length=1)
    actions: PyActionsConfig
    global_obs: PyGlobalObsConfig = Field(default_factory=PyGlobalObsConfig)
    recipe_details_obs: bool = Field(default=False)
    objects: dict[str, PyConverterConfig | PyWallConfig]
    # these are not used in the C++ code, but we allow them to be set for other uses.
    # E.g., templates can use params as a place where values are expected to be written,
    # and other parts of the template can read from there.
    params: Optional[Any] = None
    map_builder: Optional[Any] = None

    # Movement metrics configuration
    track_movement_metrics: bool = Field(
        default=False, description="Enable movement metrics tracking (sequential rotations)"
    )


class PyPolicyGameConfig(PyGameConfig):
    obs_width: Literal[11]
    obs_height: Literal[11]

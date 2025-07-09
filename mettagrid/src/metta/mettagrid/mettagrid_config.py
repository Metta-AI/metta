# mettagrid_config.py - Python configuration models and conversion logic

import copy
from typing import Any, Dict, List, Optional

from pydantic import Field, RootModel

from metta.common.util.typed_config import BaseModelWithForbidExtra

# Import C++ types from the extension module
from metta.mettagrid.mettagrid_c import (
    ActionConfig,
    AgentConfig,
    AttackActionConfig,
    ConverterConfig,
    GameConfig,
    WallConfig,
)

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

    default_resource_limit: Optional[int] = Field(default=None, ge=0)
    resource_limits: Optional[Dict[str, int]] = Field(default_factory=dict)
    freeze_duration: Optional[int] = Field(default=None, ge=-1)
    rewards: Optional[PyAgentRewards] = None
    action_failure_penalty: Optional[float] = Field(default=None, ge=0)


class PyGroupProps(RootModel[Dict[str, Any]]):
    """Python group properties configuration."""

    pass


class PyGroupConfig(BaseModelWithForbidExtra):
    """Python group configuration."""

    id: int
    sprite: Optional[int] = Field(default=None)
    group_reward_pct: Optional[float] = Field(default=None, ge=0, le=1)
    props: Optional[PyGroupProps] = None


class PyActionConfig(BaseModelWithForbidExtra):
    """Python action configuration."""

    enabled: bool
    required_resources: Optional[Dict[str, int]] = None
    consumed_resources: Optional[Dict[str, int]] = Field(default_factory=dict)


class PyAttackActionConfig(PyActionConfig):
    """Python attack action configuration."""

    defense_resources: Optional[Dict[str, int]] = Field(default_factory=dict)


class PyActionsConfig(BaseModelWithForbidExtra):
    """Python actions configuration."""

    noop: PyActionConfig
    move: PyActionConfig
    rotate: PyActionConfig
    put_items: PyActionConfig
    get_items: PyActionConfig
    attack: PyAttackActionConfig
    swap: PyActionConfig
    change_color: PyActionConfig


class PyWallConfig(BaseModelWithForbidExtra):
    """Python wall/block configuration."""

    type_id: int
    swappable: Optional[bool] = None


class PyConverterConfig(BaseModelWithForbidExtra):
    """Python converter configuration."""

    input_resources: Dict[str, int] = Field(default_factory=dict)
    output_resources: Dict[str, int] = Field(default_factory=dict)
    type_id: int
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_resource_count: int = Field(ge=0)
    color: Optional[int] = Field(default=0, ge=0, le=255)


class PyGameConfig(BaseModelWithForbidExtra):
    """Python game configuration."""

    inventory_item_names: List[str]
    num_agents: int = Field(ge=1)
    max_steps: int = Field(ge=0)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    agent: PyAgentConfig
    groups: Dict[str, PyGroupConfig] = Field(min_length=1)
    actions: PyActionsConfig
    objects: Dict[str, PyConverterConfig | PyWallConfig]


# ===== Conversion Functions =====


def from_mettagrid_config(mettagrid_config_dict: dict[str, Any]) -> GameConfig:
    """Convert a Python PyGameConfig to a C++ GameConfig."""

    mettagrid_config = PyGameConfig(**mettagrid_config_dict)

    resource_names = list(mettagrid_config.inventory_item_names)
    resource_ids = dict((name, i) for i, name in enumerate(resource_names))

    object_configs = {}

    # These are the baseline settings for all agents
    agent_default_config_dict = mettagrid_config.agent.model_dump(by_alias=True, exclude_unset=True)

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in mettagrid_config.groups.items():
        group_config_dict = group_config.model_dump(by_alias=True, exclude_unset=True)
        merged_config = copy.deepcopy(agent_default_config_dict)

        # Update, but in a nested way
        for key, value in group_config_dict.get("props", {}).items():
            if isinstance(value, dict):
                # At the time of writing, this should only be the rewards field
                merged_config[key] = value
            else:
                merged_config[key] = value

        default_resource_limit = merged_config.get("default_resource_limit", 0)

        # Handle rewards conversion
        rewards_dict = {}
        if merged_config.get("rewards"):
            if isinstance(merged_config["rewards"], PyAgentRewards):
                rewards_dict = merged_config["rewards"].model_dump(exclude_none=True)
            elif isinstance(merged_config["rewards"], dict):
                rewards_dict = merged_config["rewards"]

        agent_group_config = {
            "freeze_duration": merged_config.get("freeze_duration", 0),
            "group_id": group_config.id,
            "group_name": group_name,
            "action_failure_penalty": merged_config.get("action_failure_penalty", 0),
            "resource_limits": dict(
                (resource_id, merged_config.get("resource_limits", {}).get(resource_name, default_resource_limit))
                for (resource_id, resource_name) in enumerate(resource_names)
            ),
            "resource_rewards": dict((resource_ids[k], v) for k, v in rewards_dict.items() if not k.endswith("_max")),
            "resource_reward_max": dict(
                (resource_ids[k[:-4]], v) for k, v in rewards_dict.items() if k.endswith("_max") and v is not None
            ),
            "group_reward_pct": group_config.group_reward_pct or 0,
        }

        # HardCodedConfig
        agent_group_config["type_id"] = 0
        agent_group_config["type_name"] = "agent"
        object_configs["agent." + group_name] = AgentConfig(**agent_group_config)

    # Convert other objects
    for object_type, object_config in mettagrid_config.objects.items():
        if isinstance(object_config, PyConverterConfig):
            converter_config = ConverterConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                input_resources=dict((resource_ids[k], v) for k, v in object_config.input_resources.items()),
                output_resources=dict((resource_ids[k], v) for k, v in object_config.output_resources.items()),
                max_output=object_config.max_output,
                conversion_ticks=object_config.conversion_ticks,
                cooldown=object_config.cooldown,
                initial_resource_count=object_config.initial_resource_count,
                color=object_config.color or 0,
            )
            object_configs[object_type] = converter_config
        elif isinstance(object_config, PyWallConfig):
            wall_config = WallConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                swappable=object_config.swappable or False,
            )
            object_configs[object_type] = wall_config
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_config = mettagrid_config.model_dump(by_alias=True, exclude_none=True)

    # Convert actions
    actions_config = {}
    for action_name, action_config in game_config["actions"].items():
        action_config_params = {}
        action_config_params["consumed_resources"] = dict(
            (resource_ids[k], v) for k, v in action_config["consumed_resources"].items()
        )
        if action_config.get("required_resources", None) is not None:
            action_config_params["required_resources"] = dict(
                (resource_ids[k], v) for k, v in action_config["required_resources"].items()
            )
        else:
            action_config_params["required_resources"] = action_config_params["consumed_resources"]

        if action_name == "attack":
            action_config_params["defense_resources"] = dict(
                (resource_ids[k], v) for k, v in action_config["defense_resources"].items()
            )
            actions_config[action_name] = AttackActionConfig(**action_config_params)
        else:
            actions_config[action_name] = ActionConfig(**action_config_params)

    game_config["actions"] = actions_config

    del game_config["agent"]
    del game_config["groups"]
    game_config["objects"] = object_configs

    return GameConfig(**game_config)

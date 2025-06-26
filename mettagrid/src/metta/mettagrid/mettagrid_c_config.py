import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel, conint

from metta.mettagrid.mettagrid_config import ConverterConfig as ConverterConfig_py
from metta.mettagrid.mettagrid_config import GameConfig as GameConfig_py
from metta.mettagrid.mettagrid_config import WallConfig as WallConfig_py

Byte = conint(ge=0, le=255)
FeatureId = Byte


class BaseModelWithForbidExtra(BaseModel):
    model_config = dict(extra="forbid")


class AgentGroupConfig_cpp(BaseModelWithForbidExtra):
    """Agent group configuration."""

    freeze_duration: int = Field(ge=0)
    action_failure_penalty: float = Field(default=0, ge=0)
    max_items_per_type: Dict[FeatureId, int] = Field(default_factory=dict)
    resource_rewards: Dict[FeatureId, float] = Field(default_factory=dict)
    resource_reward_max: Dict[FeatureId, float] = Field(default_factory=dict)
    group_name: str
    group_id: int
    group_reward_pct: float = Field(ge=0, le=1)


class ActionConfig_cpp(BaseModelWithForbidExtra):
    """Action configuration."""

    enabled: bool


class ActionsConfig_cpp(BaseModelWithForbidExtra):
    """Actions configuration."""

    noop: ActionConfig_cpp
    move: ActionConfig_cpp
    rotate: ActionConfig_cpp
    put_items: ActionConfig_cpp
    get_items: ActionConfig_cpp
    attack: ActionConfig_cpp
    swap: ActionConfig_cpp
    change_color: ActionConfig_cpp
    laser_item_id: FeatureId
    armor_item_id: FeatureId


class WallConfig_cpp(BaseModelWithForbidExtra):
    """Wall/Block configuration."""

    swappable: Optional[bool] = None


class ConverterConfig_cpp(BaseModelWithForbidExtra):
    """Converter configuration for objects that convert items."""

    recipe_input: Dict[FeatureId, int] = Field(default_factory=dict)
    recipe_output: Dict[FeatureId, int] = Field(default_factory=dict)
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_items: int = Field(ge=0)
    color: Byte = Field(default=0)


class ObjectsConfig_cpp(BaseModelWithForbidExtra):
    """Objects configuration."""

    altar: Optional[ConverterConfig_cpp] = None
    mine_red: Optional[ConverterConfig_cpp] = None
    mine_blue: Optional[ConverterConfig_cpp] = None
    mine_green: Optional[ConverterConfig_cpp] = None
    generator_red: Optional[ConverterConfig_cpp] = None
    generator_blue: Optional[ConverterConfig_cpp] = None
    generator_green: Optional[ConverterConfig_cpp] = None
    armory: Optional[ConverterConfig_cpp] = None
    lasery: Optional[ConverterConfig_cpp] = None
    lab: Optional[ConverterConfig_cpp] = None
    factory: Optional[ConverterConfig_cpp] = None
    temple: Optional[ConverterConfig_cpp] = None
    wall: Optional[WallConfig_cpp] = None
    block: Optional[WallConfig_cpp] = None


class RewardSharingGroup_cpp(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig_cpp(BaseModelWithForbidExtra):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup_cpp]] = None


class GameConfig_cpp(BaseModelWithForbidExtra):
    """Game configuration."""

    inventory_item_names: List[str]
    num_agents: int = Field(ge=1)
    max_steps: int = Field(ge=0)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    agent_groups: Dict[str, AgentGroupConfig_cpp] = Field(min_length=1)
    actions: ActionsConfig_cpp
    objects: ObjectsConfig_cpp
    reward_sharing: Optional[RewardSharingConfig_cpp] = None


def from_mettagrid_config(mettagrid_config: GameConfig_py) -> GameConfig_cpp:
    """Convert a mettagrid_config.GameConfig to a mettagrid_c_config.GameConfig."""

    inventory_item_names = list(mettagrid_config.inventory_item_names)
    inventory_item_ids = dict((name, i) for i, name in enumerate(inventory_item_names))

    agent_group_configs = {}

    # these are the baseline settings for all agents
    agent_default_config_dict = mettagrid_config.agent.model_dump(by_alias=True, exclude_unset=True)

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in mettagrid_config.groups.items():
        group_config_dict = group_config.model_dump(by_alias=True, exclude_unset=True)
        merged_config = copy.deepcopy(agent_default_config_dict)
        # update, but in a nested way
        for key, value in group_config_dict.get("props", {}).items():
            if isinstance(value, dict):
                # At the time of writing, this should only be the rewards field
                merged_config[key] = value
            else:
                merged_config[key] = value

        default_item_max = merged_config.get("default_item_max", 0)

        agent_group_config = {
            "freeze_duration": merged_config.get("freeze_duration", 0),
            "group_id": group_config.id,
            "group_name": group_name,
            "action_failure_penalty": merged_config.get("rewards", {}).get("action_failure_penalty", 0),
            "max_items_per_type": dict(
                (item_id, merged_config.get(item_name + "_max", default_item_max))
                for (item_id, item_name) in enumerate(inventory_item_names)
            ),
            "resource_rewards": dict(
                (inventory_item_ids[k], v)
                for k, v in merged_config.get("rewards", {}).items()
                if not k.endswith("_max") and k != "action_failure_penalty"
            ),
            "resource_reward_max": dict(
                (inventory_item_ids[k[:-4]], v)
                for k, v in merged_config.get("rewards", {}).items()
                if k.endswith("_max")
            ),
            "group_reward_pct": group_config.group_reward_pct or 0,
        }

        # these defaults should be moved elsewhere!
        for k in agent_group_config["resource_rewards"]:
            if k not in agent_group_config["resource_reward_max"]:
                agent_group_config["resource_reward_max"][k] = 1000

        agent_group_configs["agent." + group_name] = AgentGroupConfig_cpp(**agent_group_config)

    object_configs = {}
    for object_type, object_config in mettagrid_config.objects.items():
        if isinstance(object_config, ConverterConfig_py):
            converter_config_dict = object_config.model_dump(by_alias=True, exclude_unset=True)
            converter_config_cpp_dict = {
                "recipe_input": {},
                "recipe_output": {},
            }
            for k, v in converter_config_dict.items():
                if k.startswith("input_"):
                    converter_config_cpp_dict["recipe_input"][inventory_item_ids[k[6:]]] = v
                elif k.startswith("output_"):
                    converter_config_cpp_dict["recipe_output"][inventory_item_ids[k[7:]]] = v
                else:
                    converter_config_cpp_dict[k] = v
            object_configs[object_type] = ConverterConfig_cpp(**converter_config_cpp_dict)
        elif isinstance(object_config, WallConfig_py):
            object_configs[object_type] = WallConfig_cpp(**object_config.model_dump(by_alias=True, exclude_unset=True))
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_config = mettagrid_config.model_dump(by_alias=True, exclude_none=True)
    game_config["agent_groups"] = agent_group_configs
    del game_config["agent"]
    del game_config["groups"]
    game_config["objects"] = object_configs
    # TODO: make this configurable at a higher level
    game_config["actions"]["laser_item_id"] = inventory_item_ids["laser"]
    game_config["actions"]["armor_item_id"] = inventory_item_ids["armor"]

    return GameConfig_cpp(**game_config)


def cpp_config_dict(game_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a config dict and returns a config_c dict.

    In particular, this function converts from the style of config we have in yaml to the style of config we expect
    in cpp; and validates along the way.
    """
    game_config = GameConfig_py(**game_config_dict)

    return from_mettagrid_config(game_config).model_dump(by_alias=True, exclude_none=True)

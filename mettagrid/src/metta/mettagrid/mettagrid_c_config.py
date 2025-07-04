import copy
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, RootModel, conint

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.mettagrid.mettagrid_config import ConverterConfig as ConverterConfig_py
from metta.mettagrid.mettagrid_config import GameConfig as GameConfig_py
from metta.mettagrid.mettagrid_config import WallConfig as WallConfig_py

Byte = conint(ge=0, le=255)
FeatureId = Byte


class ActionConfig_cpp(BaseModelWithForbidExtra):
    """Action configuration."""

    # Required resources should be a superset of consumed resources.
    # E.g., maybe you need a laser and a battery to attack, but only consume the laser.
    required_resources: Dict[FeatureId, int]
    consumed_resources: Dict[FeatureId, int]
    enabled: bool


class AttackActionConfig_cpp(ActionConfig_cpp):
    """Attack action configuration."""

    # If there are no defense resources, the attack will always succeed.
    # Otherwise, you need to have enough defense resources to block the attack.
    defense_resources: Dict[FeatureId, int]


class ActionsConfig_cpp(BaseModelWithForbidExtra):
    """Actions configuration."""

    noop: ActionConfig_cpp
    move: ActionConfig_cpp
    rotate: ActionConfig_cpp
    put_items: ActionConfig_cpp
    get_items: ActionConfig_cpp
    attack: AttackActionConfig_cpp
    swap: ActionConfig_cpp
    change_color: ActionConfig_cpp


class ObjectConfig_cpp(BaseModelWithForbidExtra):
    """Object configuration."""

    object_type: Literal["agent", "converter", "wall"]
    # type_id is meant for consumption by the agents, and it should show up in features.
    type_id: int
    # type_name is meant for consumption by humans, and will be used in stats and the viewer.
    type_name: str


class AgentGroupConfig_cpp(ObjectConfig_cpp):
    """Agent group configuration."""

    object_type: Literal["agent"] = "agent"
    freeze_duration: int = Field(ge=-1)
    action_failure_penalty: float = Field(default=0, ge=0)
    max_items_per_type: Dict[FeatureId, int] = Field(default_factory=dict)
    resource_rewards: Dict[FeatureId, float] = Field(default_factory=dict)
    resource_reward_max: Dict[FeatureId, float] = Field(default_factory=dict)
    group_name: str
    group_id: int
    group_reward_pct: float = Field(ge=0, le=1)
    type_id: int = 0


class WallConfig_cpp(ObjectConfig_cpp):
    """Wall/Block configuration."""

    object_type: Literal["wall"] = "wall"
    swappable: Optional[bool] = None
    type_id: Byte


class ConverterConfig_cpp(ObjectConfig_cpp):
    """Converter configuration for objects that convert items."""

    object_type: Literal["converter"] = "converter"
    recipe_input: Dict[FeatureId, int] = Field(default_factory=dict)
    recipe_output: Dict[FeatureId, int] = Field(default_factory=dict)
    max_output: int = Field(ge=-1)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_items: int = Field(ge=0)
    color: Byte = Field(default=0)
    type_id: Byte


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
    actions: ActionsConfig_cpp
    objects: Dict[str, AgentGroupConfig_cpp | ConverterConfig_cpp | WallConfig_cpp]
    reward_sharing: Optional[RewardSharingConfig_cpp] = None


def from_mettagrid_config(mettagrid_config: GameConfig_py) -> GameConfig_cpp:
    """Convert a mettagrid_config.GameConfig to a mettagrid_c_config.GameConfig."""

    inventory_item_names = list(mettagrid_config.inventory_item_names)
    inventory_item_ids = dict((name, i) for i, name in enumerate(inventory_item_names))

    object_configs = {}

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

        # #HardCodedConfig
        # these defaults should be moved elsewhere!
        for k in agent_group_config["resource_rewards"]:
            if k not in agent_group_config["resource_reward_max"]:
                agent_group_config["resource_reward_max"][k] = 1000

        # #HardCodedConfig
        agent_group_config["type_id"] = 0
        agent_group_config["type_name"] = "agent"
        object_configs["agent." + group_name] = AgentGroupConfig_cpp(**agent_group_config)

    for object_type, object_config in mettagrid_config.objects.items():
        if isinstance(object_config, ConverterConfig_py):
            converter_config_dict = object_config.model_dump(by_alias=True, exclude_unset=True)
            converter_config_cpp_dict: dict[str, Any] = {
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
            converter_config_cpp_dict["type_name"] = object_type
            object_configs[object_type] = ConverterConfig_cpp(**converter_config_cpp_dict)
        elif isinstance(object_config, WallConfig_py):
            object_config_dict = object_config.model_dump(by_alias=True, exclude_unset=True)
            object_config_dict["type_name"] = object_type
            object_configs[object_type] = WallConfig_cpp(**object_config_dict)
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_config = mettagrid_config.model_dump(by_alias=True, exclude_none=True)

    # Add required and consumed resources to the attack action
    for action_name, action_config in game_config["actions"].items():
        game_config["actions"][action_name]["consumed_resources"] = dict(
            (inventory_item_ids[k], v) for k, v in action_config["consumed_resources"].items()
        )
        if action_config.get("required_resources", None) is not None:
            game_config["actions"][action_name]["required_resources"] = dict(
                (inventory_item_ids[k], v) for k, v in action_config["required_resources"].items()
            )
        else:
            game_config["actions"][action_name]["required_resources"] = game_config["actions"][action_name][
                "consumed_resources"
            ]
        if action_name == "attack":
            game_config["actions"][action_name]["defense_resources"] = dict(
                (inventory_item_ids[k], v) for k, v in action_config["defense_resources"].items()
            )

    del game_config["agent"]
    del game_config["groups"]
    game_config["objects"] = object_configs

    return GameConfig_cpp(**game_config)


def cpp_config_dict(game_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a config dict and returns a config_c dict.

    In particular, this function converts from the style of config we have in yaml to the style of config we expect
    in cpp; and validates along the way.
    """
    game_config = GameConfig_py(**game_config_dict)

    return from_mettagrid_config(game_config).model_dump(by_alias=True, exclude_none=True)

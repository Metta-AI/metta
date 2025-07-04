import copy
from typing import Any, ClassVar, Dict, List

from pydantic import ConfigDict, Field, conint

from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.mettagrid.mettagrid_c import ActionConfig as ActionConfig_cpp
from metta.mettagrid.mettagrid_c import AgentConfig as AgentConfig_cpp
from metta.mettagrid.mettagrid_c import AttackActionConfig as AttackActionConfig_cpp
from metta.mettagrid.mettagrid_c import ConverterConfig as ConverterConfig_cpp
from metta.mettagrid.mettagrid_c import WallConfig as WallConfig_cpp
from metta.mettagrid.mettagrid_config import ConverterConfig as ConverterConfig_py
from metta.mettagrid.mettagrid_config import GameConfig as GameConfig_py
from metta.mettagrid.mettagrid_config import WallConfig as WallConfig_py

Byte = conint(ge=0, le=255)
FeatureId = Byte


class ActionsConfig_cpp(BaseModelWithForbidExtra):
    """Actions configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    noop: ActionConfig_cpp
    move: ActionConfig_cpp
    rotate: ActionConfig_cpp
    put_items: ActionConfig_cpp
    get_items: ActionConfig_cpp
    attack: AttackActionConfig_cpp
    swap: ActionConfig_cpp
    change_color: ActionConfig_cpp

class GameConfig_cpp(BaseModelWithForbidExtra):
    """Game configuration."""

    inventory_item_names: List[str]
    num_agents: int = Field(ge=1)
    max_steps: int = Field(ge=0)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    actions: ActionsConfig_cpp
    objects: Dict[str, Any]


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
        object_configs["agent." + group_name] = AgentConfig_cpp(**agent_group_config)

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
            converter_config_cpp_dict["type_name"] = object_type
            object_configs[object_type] = ConverterConfig_cpp(**converter_config_cpp_dict)
        elif isinstance(object_config, WallConfig_py):
            wall_config = WallConfig_cpp(
                type_id=object_config.type_id,
                type_name=object_type,
                swappable=object_config.swappable,
            )
            object_configs[object_type] = wall_config
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_config = mettagrid_config.model_dump(by_alias=True, exclude_none=True)

    actions_config_cpp = {}
    # Add required and consumed resources to the attack action
    for action_name, action_config in game_config["actions"].items():
        action_config_cpp_params = {}
        action_config_cpp_params["consumed_resources"] = dict(
            (inventory_item_ids[k], v) for k, v in action_config["consumed_resources"].items()
        )
        if action_config.get("required_resources", None) is not None:
            action_config_cpp_params["required_resources"] = dict(
                (inventory_item_ids[k], v) for k, v in action_config["required_resources"].items()
            )
        else:
            action_config_cpp_params["required_resources"] = action_config_cpp_params["consumed_resources"]
        if action_name == "attack":
            action_config_cpp_params["defense_resources"] = dict(
                (inventory_item_ids[k], v) for k, v in action_config["defense_resources"].items()
            )
            actions_config_cpp[action_name] = AttackActionConfig_cpp(**action_config_cpp_params)
        else:
            actions_config_cpp[action_name] = ActionConfig_cpp(**action_config_cpp_params)

    game_config["actions"] = actions_config_cpp

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

import copy
from typing import Annotated, Any

from pydantic import Field

from metta.mettagrid.mettagrid_c import ActionConfig as ActionConfig_cpp
from metta.mettagrid.mettagrid_c import AgentConfig as AgentConfig_cpp
from metta.mettagrid.mettagrid_c import AttackActionConfig as AttackActionConfig_cpp
from metta.mettagrid.mettagrid_c import ConverterConfig as ConverterConfig_cpp
from metta.mettagrid.mettagrid_c import GameConfig as GameConfig_cpp
from metta.mettagrid.mettagrid_c import WallConfig as WallConfig_cpp
from metta.mettagrid.mettagrid_config import ConverterConfig as ConverterConfig_py
from metta.mettagrid.mettagrid_config import GameConfig as GameConfig_py
from metta.mettagrid.mettagrid_config import WallConfig as WallConfig_py

Byte = Annotated[int, Field(ge=0, le=255)]
FeatureId = Byte


def from_mettagrid_config(mettagrid_config_dict: dict[str, Any]) -> GameConfig_cpp:
    """Convert a mettagrid_config.GameConfig to a mettagrid_c_config.GameConfig."""

    mettagrid_config = GameConfig_py(**mettagrid_config_dict)

    resource_names = list(mettagrid_config.inventory_item_names)
    resource_ids = dict((name, i) for i, name in enumerate(resource_names))

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

        default_resource_limit = merged_config.get("default_resource_limit", 0)

        agent_group_config = {
            "freeze_duration": merged_config.get("freeze_duration", 0),
            "group_id": group_config.id,
            "group_name": group_name,
            "action_failure_penalty": merged_config.get("action_failure_penalty", 0),
            "resource_limits": dict(
                (resource_id, merged_config.get("resource_limits", {}).get(resource_name, default_resource_limit))
                for (resource_id, resource_name) in enumerate(resource_names)
            ),
            "resource_rewards": dict(
                (resource_ids[k], v) for k, v in merged_config.get("rewards", {}).items() if not k.endswith("_max")
            ),
            "resource_reward_max": dict(
                (resource_ids[k[:-4]], v)
                for k, v in merged_config.get("rewards", {}).items()
                if k.endswith("_max") and v is not None
            ),
            "group_reward_pct": group_config.group_reward_pct or 0,
        }

        # #HardCodedConfig
        agent_group_config["type_id"] = 0
        agent_group_config["type_name"] = "agent"
        object_configs["agent." + group_name] = AgentConfig_cpp(**agent_group_config)

    for object_type, object_config in mettagrid_config.objects.items():
        if isinstance(object_config, ConverterConfig_py):
            converter_config_cpp = ConverterConfig_cpp(
                type_id=object_config.type_id,
                type_name=object_type,
                input_resources=dict((resource_ids[k], v) for k, v in object_config.input_resources.items()),
                output_resources=dict((resource_ids[k], v) for k, v in object_config.output_resources.items()),
                max_output=object_config.max_output,
                conversion_ticks=object_config.conversion_ticks,
                cooldown=object_config.cooldown,
                initial_resource_count=object_config.initial_resource_count,
                color=object_config.color,
            )
            object_configs[object_type] = converter_config_cpp
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
            (resource_ids[k], v) for k, v in action_config["consumed_resources"].items()
        )
        if action_config.get("required_resources", None) is not None:
            action_config_cpp_params["required_resources"] = dict(
                (resource_ids[k], v) for k, v in action_config["required_resources"].items()
            )
        else:
            action_config_cpp_params["required_resources"] = action_config_cpp_params["consumed_resources"]
        if action_name == "attack":
            action_config_cpp_params["defense_resources"] = dict(
                (resource_ids[k], v) for k, v in action_config["defense_resources"].items()
            )
            actions_config_cpp[action_name] = AttackActionConfig_cpp(**action_config_cpp_params)
        else:
            actions_config_cpp[action_name] = ActionConfig_cpp(**action_config_cpp_params)

    game_config["actions"] = actions_config_cpp

    del game_config["agent"]
    del game_config["groups"]
    game_config["objects"] = object_configs

    return GameConfig_cpp(**game_config)

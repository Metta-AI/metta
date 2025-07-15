import copy
from typing import Any

from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import ChangeGlyphActionConfig as CppChangeGlyphActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import PyConverterConfig, PyGameConfig, PyWallConfig


def from_mettagrid_config(mettagrid_config_dict: dict[str, Any]) -> CppGameConfig:
    """Convert a PyGameConfig to a CppGameConfig."""

    game_config = PyGameConfig(**mettagrid_config_dict)

    resource_names = list(game_config.inventory_item_names)
    resource_name_to_id = {name: i for i, name in enumerate(resource_names)}

    objects_cpp_params = {}  # params for CppConverterConfig or CppWallConfig

    # These are the baseline settings for all agents
    default_agent_config_dict = game_config.agent.model_dump(by_alias=True, exclude_unset=True)
    default_resource_limit = default_agent_config_dict.get("default_resource_limit", 0)

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in game_config.groups.items():
        agent_group_props = copy.deepcopy(default_agent_config_dict)

        group_config_dict = group_config.model_dump(by_alias=True, exclude_unset=True)

        # Update, but in a nested way
        for key, value in group_config_dict.get("props", {}).items():
            agent_group_props[key] = value

        agent_cpp_params = {
            "freeze_duration": agent_group_props.get("freeze_duration", 0),
            "group_id": group_config.id,
            "group_name": group_name,
            "action_failure_penalty": agent_group_props.get("action_failure_penalty", 0),
            "resource_limits": {
                resource_id: agent_group_props.get("resource_limits", {}).get(resource_name, default_resource_limit)
                for resource_id, resource_name in enumerate(resource_names)
            },
            "resource_rewards": {
                resource_name_to_id[k]: v
                for k, v in agent_group_props.get("rewards", {}).items()
                if not k.endswith("_max")
            },
            "resource_reward_max": {
                resource_name_to_id[k[:-4]]: v
                for k, v in agent_group_props.get("rewards", {}).items()
                if k.endswith("_max") and v is not None
            },
            "group_reward_pct": group_config.group_reward_pct or 0,
            "type_id": 0,
            "type_name": "agent",
        }

        objects_cpp_params["agent." + group_name] = CppAgentConfig(**agent_cpp_params)

    # Convert other objects
    for object_type, object_config in game_config.objects.items():
        if isinstance(object_config, PyConverterConfig):
            cpp_converter_config = CppConverterConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                input_resources={resource_name_to_id[k]: v for k, v in object_config.input_resources.items() if v > 0},
                output_resources={
                    resource_name_to_id[k]: v for k, v in object_config.output_resources.items() if v > 0
                },
                max_output=object_config.max_output,
                conversion_ticks=object_config.conversion_ticks,
                cooldown=object_config.cooldown,
                initial_resource_count=object_config.initial_resource_count,
                color=object_config.color,
            )
            objects_cpp_params[object_type] = cpp_converter_config
        elif isinstance(object_config, PyWallConfig):
            cpp_wall_config = CppWallConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                swappable=object_config.swappable,
            )
            objects_cpp_params[object_type] = cpp_wall_config
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_cpp_params = game_config.model_dump(by_alias=True, exclude_none=True)
    del game_cpp_params["agent"]
    del game_cpp_params["groups"]

    actions_cpp_params = {}
    for action_name, action_config in game_cpp_params["actions"].items():
        if not action_config["enabled"]:
            continue

        action_cpp_params = {
            "consumed_resources": {resource_name_to_id[k]: v for k, v in action_config["consumed_resources"].items()},
            "required_resources": {
                resource_name_to_id[k]: v
                for k, v in (action_config.get("required_resources") or action_config["consumed_resources"]).items()
            },
        }

        if action_name == "attack":
            action_cpp_params["defense_resources"] = {
                resource_name_to_id[k]: v for k, v in action_config["defense_resources"].items()
            }
            actions_cpp_params[action_name] = CppAttackActionConfig(**action_cpp_params)
        elif action_name == "change_glyph":
            action_cpp_params["number_of_glyphs"] = action_config["number_of_glyphs"]
            actions_cpp_params[action_name] = CppChangeGlyphActionConfig(**action_cpp_params)
        else:
            actions_cpp_params[action_name] = CppActionConfig(**action_cpp_params)

    game_cpp_params["actions"] = actions_cpp_params
    game_cpp_params["objects"] = objects_cpp_params

    return CppGameConfig(**game_cpp_params)

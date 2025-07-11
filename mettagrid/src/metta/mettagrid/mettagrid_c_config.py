import copy
from typing import Any

from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import PyAgentRewards, PyConverterConfig, PyGameConfig, PyWallConfig


def from_mettagrid_config(mettagrid_config_dict: dict[str, Any]) -> CppGameConfig:
    """Convert a Python PyGameConfig to a CppGameConfig."""

    py_mettagrid_config = PyGameConfig(**mettagrid_config_dict)

    resource_names = list(py_mettagrid_config.inventory_item_names)
    resource_ids = dict((name, i) for i, name in enumerate(resource_names))

    cpp_object_configs = {}

    # These are the baseline settings for all agents
    agent_default_config_dict = py_mettagrid_config.agent.model_dump(by_alias=True, exclude_unset=True)

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in py_mettagrid_config.groups.items():
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
        cpp_object_configs["agent." + group_name] = CppAgentConfig(**agent_group_config)

    # Convert other objects
    for py_object_type, py_cfg in py_mettagrid_config.objects.items():
        if isinstance(py_cfg, PyConverterConfig):
            cpp_converter_config = CppConverterConfig(
                type_id=py_cfg.type_id,
                type_name=py_object_type,
                input_resources=dict((resource_ids[k], v) for k, v in py_cfg.input_resources.items() if v > 0),
                output_resources=dict((resource_ids[k], v) for k, v in py_cfg.output_resources.items() if v > 0),
                max_output=py_cfg.max_output,
                conversion_ticks=py_cfg.conversion_ticks,
                cooldown=py_cfg.cooldown,
                initial_resource_count=py_cfg.initial_resource_count,
                color=py_cfg.color,
            )
            cpp_object_configs[py_object_type] = cpp_converter_config
        elif isinstance(py_cfg, PyWallConfig):
            cpp_wall_config = CppWallConfig(
                type_id=py_cfg.type_id,
                type_name=py_object_type,
                swappable=py_cfg.swappable,
            )
            cpp_object_configs[py_object_type] = cpp_wall_config
        else:
            raise ValueError(f"Unknown object type: {py_object_type}")

    py_game_config = py_mettagrid_config.model_dump(by_alias=True, exclude_none=True)

    cpp_actions_config = {}
    # Add required and consumed resources to the attack action
    for py_action_name, py_cfg in py_game_config["actions"].items():
        params = {}
        params["consumed_resources"] = dict((resource_ids[k], v) for k, v in py_cfg["consumed_resources"].items())
        if py_cfg.get("required_resources", None) is not None:
            params["required_resources"] = dict((resource_ids[k], v) for k, v in py_cfg["required_resources"].items())
        else:
            params["required_resources"] = params["consumed_resources"]

        if py_action_name == "attack":
            params["defense_resources"] = dict((resource_ids[k], v) for k, v in py_cfg["defense_resources"].items())
            cpp_actions_config[py_action_name] = CppAttackActionConfig(**params)
        else:
            cpp_actions_config[py_action_name] = CppActionConfig(**params)

    py_game_config["actions"] = cpp_actions_config

    del py_game_config["agent"]
    del py_game_config["groups"]
    py_game_config["objects"] = cpp_object_configs

    return CppGameConfig(**py_game_config)
